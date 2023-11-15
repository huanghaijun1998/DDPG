#!/usr/bin/env python3
import tensorflow as tf
import os
import numpy as np
import math

# Hyper Parameters 超参数
LAYER1_SIZE = 400       # 第一层的大小
LAYER2_SIZE = 300       # 第二层的大小
LEARNING_RATE = 0.0001     #学习率
TAU = 0.001     #软更新参数

#定义了模型保存的路径
model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'critic1-3')

#tf.reset_default_graph()    #新加的，重置新的模型

#定义了一个名为 CriticNetwork 的类
class CriticNetwork:
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		# create q network  创建一个Q网络
		self.state_input, self.action_input, self.q_value_output,\
		self.net = self.create_q_network(state_dim,action_dim)

		# create target q network (the same structure with q network)    创建一个目标Q网络，结构与Q网络相同
		#self.net 是一个实例变量，用于存储神经网络的参数（权重和偏置）
		self.target_state_input, self.target_action_input, self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

		self.create_training_method()            #创建一个训练的方法 
		self.sess.run(tf.initialize_all_variables())
			
		self.update_target()      #更新目标
		self.load_network()       #导入网络
		#self.save_network()    #自己加的，然后又删掉
		#self.saver = tf.train.Saver()      #也是后面加的，跟load_network网络中一样
	 
      

	   #定义了训练方法和损失函数
	def create_training_method(self):
		self.y_input = tf.placeholder("float",[None,1])      #创建一个 TensorFlow 占位符，用于输入每个训练样本对应的目标输出值（也就是标签）。None 表示可以在训练时输入任意数量的样本，而 1 表示每个样本的标签是一个单独的浮点数值。
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output))      #这是损失函数的定义，用于衡量模型预测值与目标值之间的差距。self.y_input 是真实的标签值，而 self.q_value_output 是模型的预测输出值。tf.square 计算了预测值与真实值之差的平方，tf.reduce_mean 则求取了所有样本的平均值，得到一个标量值作为损失函数的输出。
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)      #这里使用 Adam 优化器来最小化损失函数 self.cost。优化器的目标是调整模型的参数，使得损失函数的值尽可能小，从而提高模型的预测准确性。LEARNING_RATE 是优化器的学习率，决定了每次参数更新的步长。
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)       #这里计算了损失函数 self.cost 相对于动作输入 self.action_input 的梯度。这个梯度在训练过程中将会用到，用于更新动作网络的参数，使得动作网络能够更好地逼近目标 Q 值网络的输出。
                  #tf.gradients() 函数用于计算一个张量（或一组张量）相对于另一个张量（或一组张量）的梯度。



	def create_q_network(self,state_dim,action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])                   #状态输入
		action_input = tf.placeholder("float",[None,action_dim])             #动作输入


         #这个 self.variable() 方法可以方便地在神经网络中创建和初始化参数，如权重矩阵和偏置向量。
		 # 通过调用这个方法，可以根据指定的形状和初始化范围创建相应的变量，并将其用于构建神经网络模型。
		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
		W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
		b2 = self.variable([layer2_size],layer1_size+action_dim)

		#通过调用 TensorFlow 的 tf.random_uniform() 函数创建一个均匀分布的随机张量作为变量的初始值。
		#然后，通过调用 TensorFlow 的 tf.Variable() 函数将随机张量包装成一个可训练的变量。
		W3 = tf.Variable(tf.random_uniform([layer2_size,1],-0.003,0.003))
		b3 = tf.Variable(tf.random_uniform([1],-0.003,0.003))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.matmul(layer2,W3) + b3

		return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]


      #创建一个名为create_target_q_network的目标Q网络
	def create_target_q_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])


         #ema 是一个指数移动平均（Exponential Moving Average，EMA）对象，通过 tf.train.ExponentialMovingAverage() 函数创建。它用于计算网络参数的移动平均值，以实现目标网络的更新。
         #decay=1-TAU 是指数移动平均的衰减因子，其中 TAU 是一个超参数，表示更新目标网络时新权重所占的比例。
         # target_update 是一个操作（operation），通过 ema.apply() 方法将指数移动平均应用到网络参数上，用于更新目标网络的参数。
         #target_net 是一个列表推导式，用于创建目标网络的参数副本。它遍历 net 列表中的每个参数，通过 ema.average(x) 方法获取其指数移动平均值，并将其存储在 target_net 列表中
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]



        #在代码中，首先计算第一层隐藏层的输出 layer1。它通过将状态输入 state_input 与目标网络的第一个权重矩阵 target_net[0] 相乘，并加上第一个偏置向量 target_net[1]，然后应用 ReLU 激活函数 tf.nn.relu() 得到隐藏层的输出
       #接下来，计算第二层隐藏层的输出 layer2。它通过将第一层隐藏层的输出 layer1 与目标网络的第二个权重矩阵 target_net[2] 相乘，再加上动作输入 action_input 与目标网络的第三个权重矩阵 target_net[3] 相乘，以及第二个偏置向量 target_net[4]，然后应用 ReLU 激活函数 tf.nn.relu() 得到隐藏层的输出。
      #最后，计算输出层的 Q 值 q_value_output。它通过将第二层隐藏层的输出 layer2 与目标网络的第四个权重矩阵 target_net[5] 相乘，再加上第三个偏置向量 target_net[6] 得到。
     #这段代码实现了目标演员网络的前向传播过程，根据输入的状态和动作，计算出相应的 Q 值输出。这些 Q 值输出将在强化学习算法中用于计算损失并进行网络的训练。
		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		q_value_output = tf.matmul(layer2,target_net[5]) + target_net[6]

		return state_input,action_input,q_value_output,target_update

	def update_target(self):
		self.sess.run(self.target_update)     #通过调用 self.sess.run(self.target_update)，目标网络的参数将从主网络复制，实现了目标网络的更新。这样，在训练过程中，目标网络将与主网络保持同步，并提供稳定的目标 Q 值用于更新主网络。这有助于加速强化学习的收敛过程，并提高学习的效率。

	def train(self,y_batch,state_batch,action_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={self.y_input:y_batch, self.state_input:state_batch, self.action_input:action_batch})
         #self.optimizer 是一个 TensorFlow 优化器，用于最小化损失函数。损失函数 self.cost 是目标 Q 值和主神经网络输出 Q 值之间的均方差。通过运行 self.sess.run(self.optimizer,feed_dict={...})，我们可以使用输入的批次数据进行一次梯度下降更新，从而更新主神经网络的参数，使其逼近目标 Q 值。

           #gradients 方法用于计算给定状态和动作批次的 Q 值关于动作的梯度。
	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={self.state_input:state_batch, self.action_input:action_batch})[0]
           #我们可以计算关于动作的 Q 值梯度。这个梯度用于在 DDPG 算法中更新 Actor 网络的参数，从而让 Actor 网络生成更优的动作策略。

       #target_q 方法用于计算目标 Q 值的估计。
	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={self.target_state_input:state_batch, self.target_action_input:action_batch})
        
		#q_value 方法用于计算 Critic 网络（Q 网络）在给定状态和动作下的 Q 值估计。
	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={self.state_input:state_batch, self.action_input:action_batch})

	# f fan-in size     这段代码中的 variable 方法用于创建一个可训练的 TensorFlow 变量（Variable）。在深度学习中，参数（权重和偏置）通常是可训练的变量，它们会根据损失函数进行反向传播和梯度下降来进行优化。
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	#下面是自己屏蔽的
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded critic network")
		else:
			print("Could not find old  critic-network weights")

	def save_network(self,time_step):
		print('save critic-network...',time_step)
		self.saver.save(self.sess, model_dir + 'critic-network', global_step=time_step)
		print('保存成功')