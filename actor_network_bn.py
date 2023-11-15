#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import os
import numpy as np
import math

# Hyper Parameters 超参数
LAYER1_SIZE = 400   # 第一层的大小
LAYER2_SIZE = 300   # 第二层的大小
LEARNING_RATE = 0.0001    #学习率
TAU = 0.001    #软更新参数

#定义了模型保存的路径
model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'actor1-3')

#tf.reset_default_graph()    #新加的，重置新的模型

#定义了一个名为 ActorNetwork 的类
#在初始化过程中，创建了演员网络、目标演员网络，并定义了训练规则。
# 同时初始化变量并加载已有的网络权重
class ActorNetwork:
	def __init__(self,sess,state_dim,action_dim):    #初始化方法，self是一个特殊的参数，用于表示类的实例对象自身。在类的方法中，通过 self 可以访问类的属性和方法。sess 是一个普通的参数，用于传递 TensorFlow 的会话对象接受 TensorFlow 的会话对象 sess、状态空间维度 state_dim 和动作空间维度 action_dim 作为输入参数。
		self.time_step = 0   #创建了一个名为 time_step 的实例变量，并将其初始化为0。这个变量用于跟踪训练的时间步数
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim     #将参数 state_dim 的值赋给类的实例变量 self.state_dim。这样做的原因是为了在整个类的各个方法中都能访问和使用这个值。
		# create actor network    建立演员网络
		#将返回的结果复制给多个变量
		self.state_input,self.action_output,self.net,self.is_training = self.create_network(state_dim, action_dim)

		# create target actor network    建立目标演员网络
		self.target_state_input,self.target_action_output,self.target_update,self.target_is_training = self.create_target_network(state_dim,action_dim,self.net)

		# define training rules      训练的规则
		self.create_training_method()          #训练的方法，后面有具体的训练方法
		self.sess.run(tf.initialize_all_variables())      #初始化变量的意思

		self.update_target()     #实现了更新目标网络的参数
		self.load_network()      #但要注意如果没有加载预训练的参数，演员网络将是随机初始化的，可能会导致初始训练的表现不稳定。随着训练的进行，演员网络的策略会逐渐改进，但初始阶段可能会出现不稳定的训练过程。
		#self.save_network()      #自己加的，然后又删掉

		#self.saver = tf.train.Saver()    #也是后面加的，跟load_network网络中一样

        #创建训练网络的方法。定义了训练规则，包括参数梯度计算和优化器的应用。
	def create_training_method(self):     #def 是 Python 中用于定义函数的关键字。此处定义了一个函数
	
		# self.q_gradient_input: 这是类实例的一个属性，应该是输入，用于存储 TensorFlow 占位符对象。
        # tf.placeholder("float", [None, self.action_dim]): 这是使用 TensorFlow 的 tf.placeholder() 函数创建一个占位符对象。占位符是在 TensorFlow 计算图中预留位置的特殊节点，可以在执行时用实际数据填充。
        # "float": 指定了占位符的数据类型，这里是浮点型。
       # [None, self.action_dim]: 指定了占位符的形状（shape）。这里使用了 None 来表示该维度可以是任意长度，而 self.action_dim 表示第二维的长度由类实例的 action_dim 属性决定。
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])  #
		
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		#它创建了一个 Adam 优化器，并将参数梯度与网络参数进行梯度更新。
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))
          #优化器是否可以更好的优化呢？

        #创建演员网络的方法。根据输入的状态空间维度和动作空间维度，构建了一个具有两个隐藏层的前馈神经网络。
		# 网络的每个隐藏层都经过批量归一化（Batch Normalization）和 ReLU 激活函数的处理。     
	def create_network(self,state_dim, action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)     #该占位符的数据类型是布尔型（tf.bool），用于指示当前是否处于训练阶段。

        #W1 是一个形状为 [state_dim, layer1_size] 的变量，用于表示模型的第一个权重矩阵。 variable为某一种方法
		# state_dim 是状态空间的维度， layer1_size 是第一层神经网络的大小（即隐藏层的神经元数量）。
		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)

        #    tf.Variable(): 这是 TensorFlow 的函数，用于创建一个可训练的变量。
        #tf.random_uniform([layer2_size, action_dim], -0.003, 0.003): 这是使用 TensorFlow 的 tf.random_uniform() 函数创建一个均匀分布的随机张量。
       # [layer2_size, action_dim]: 这是张量的形状（shape），其中 layer2_size 是表示第二层神经网络大小（即隐藏层的神经元数量），action_dim 是表示动作空间维度的值。
       # -0.003 和 0.003: 这是均匀分布的随机值的范围，即随机值的最小值和最大值。
		W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-0.003, 0.003))
		b3 = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))


         #计算了输入 state_input 与权重矩阵 W1 的矩阵乘法，并将结果与偏置向量 b1 相加，得到 layer1。matmul函数执行矩阵的运算
		layer1 = tf.matmul(state_input,W1) + b1      #batch_norm_layer在后面已定义
		# 该行代码的作用是将 layer1 输入应用于批量归一化层，并通过 ReLU 激活函数进行非线性变换。
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='batch_norm_1',activation=tf.nn.relu)
		
		layer2 = tf.matmul(layer1_bn,W2) + b2
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='batch_norm_2',activation=tf.nn.relu)
		
		#线速度和角速度的激活函数不一样
		action = tf.matmul(layer2_bn, W3) + b3
		action_linear = self.batch_norm_layer(action[:, None, 0],training_phase=is_training,scope_bn='action_linear',activation=tf.sigmoid)
		action_angular = self.batch_norm_layer(action[:, None, 1],training_phase=is_training,scope_bn='action_angular',activation=tf.tanh)
		# action_linear = tf.sigmoid(action[:, None, 0])
		# action_angular = tf.tanh(action[:, None, 1])
		#这段代码使用了 TensorFlow 的 `concat` 函数，将两个张量 `action_linear` 和 `action_angular` 沿着最后一个维度拼接起来，生成一个新的张量 `action`。具体来说，`action_linear` 表示线性动作，`action_angular` 表示角速度动作，而 `action` 则是将两者合并在一起的动作张量。
		action = tf.concat([action_linear, action_angular], axis=-1)

		return state_input, action, [W1,b1,W2,b2,W3,b3], is_training

       #创建目标演员网络的方法。
	   # 与演员网络类似，但在创建过程中使用了指数移动平均（Exponential Moving Average）来更新目标网络的参数。
	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)      #用来更新
		target_net = [ema.average(x) for x in net]     #最后，函数创建了一个名为 `target_net` 的列表，其中包含 `net` 列表中张量的指数移动平均值。这是使用列表推导式和 `ema` 对象的 `average` 方法完成的。生成的列表表示目标网络，它是主网络的冻结副本，其参数会随着时间缓慢更新。


           #这段代码实现了目标网络的前向传播过程。它接受输入 `state_input` 和目标网络的参数 `target_net`，并构建了一个两层的全连接神经网络。其中，`tf.matmul` 表示矩阵乘法，`+` 表示矩阵加法，`self.batch_norm_layer` 表示批归一化操作，`tf.nn.relu` 表示 ReLU 激活函数。
           #具体来说，第一层的输出为 `layer1`，计算公式为 `tf.matmul(state_input,target_net[0]) + target_net[1]`，其中 `target_net[0]` 和 `target_net[1]` 是目标网络的第一层参数（权重和偏置）。然后，对 `layer1` 进行批归一化操作，并使用 ReLU 激活函数得到 `layer1_bn`。
           #第二层的输出为 `layer2`，计算公式为 `tf.matmul(layer1_bn,target_net[2]) + target_net[3]`，其中 `target_net[2]` 和 `target_net[3]` 是目标网络的第二层参数。然后，对 `layer2` 进行批归一化操作，并使用 ReLU 激活函数得到 `layer2_bn`。
            #这个目标网络的结构和主网络的结构非常相似，都是两层全连接神经网络，不同的是目标网络的参数是从主网络的参数中通过指数移动平均得到的。因此，目标网络的输出是一个对主网络输出的一种“软剪枝”，通过缓慢地更新参数，使目标网络的输出更加平滑，从而提高算法的稳定性和泛化能力。
		layer1 = tf.matmul(state_input,target_net[0]) + target_net[1]
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='target_batch_norm_1',activation=tf.nn.relu)

		layer2 = tf.matmul(layer1_bn,target_net[2]) + target_net[3]
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='target_batch_norm_2',activation=tf.nn.relu)
		




		#这段代码计算了神经网络的原始输出 action 张量，然后通过批量归一化并应用特定的激活函数，提取了动作中的线性和角度部分。最终得到的 action_linear 和 action_angular 张量代表智能体在环境中将要执行的最终动作。
		#layer2_bn 是神经网络中某一层的输出张量。上述代码计算了动作张量 action。它执行了 layer2_bn 和最后一层的权重 target_net[4] 的矩阵乘法，然后加上偏置项 target_net[5]。得到的 action 张量是神经网络的原始输出，表示智能体应该采取的动作。
		action = tf.matmul(layer2_bn, target_net[4]) + target_net[5]
		#对 action 张量中的第一个元素（索引为0）进行批量归一化操作，该元素表示线性动作。批量归一化是一种用于归一化层输入的技术，有助于稳定和加速训练过程。training_phase 是一个占位符，用于指示神经网络是处于训练模式还是其他模式。scope_bn 是批量归一化的命名范围。activation=tf.sigmoid 表示在批量归一化后，该层使用了 sigmoid 激活函数。
		action_linear = self.batch_norm_layer(action[:, None, 0], training_phase=is_training, scope_bn='target_action_linear', activation=tf.sigmoid)
		#这行代码对 action 张量中的第二个元素（索引为1）进行批量归一化操作，该元素表示角动作。activation=tf.tanh 表示在批量归一化后，该层使用了双曲正切（tanh）激活函数。
		action_angular = self.batch_norm_layer(action[:, None, 1], training_phase=is_training, scope_bn='target_action_angular', activation=tf.tanh)
		# action_linear = tf.sigmoid(action[:, None, 0])
		# action_angular = tf.tanh(action[:, None, 1])
		action = tf.concat([action_linear, action_angular], axis=-1)      #将线速度和角速度合并为一个张量

		return state_input, action, target_update, is_training

      #定义update_target来更新目标网络
	def update_target(self):
		self.sess.run(self.target_update)

         #定义训练的方法，self.is_training: True表示当前是否处于训练阶段
		 #self.optimizer 表示演员网络的优化器，用于更新网络的参数以最小化损失函数
		 #{self.q_gradient_input: q_gradient_batch, self.state_input: state_batch, self.is_training: True} 是一个 feed_dict 字典，将 Q 梯度批次 q_gradient_batch 传递给 self.q_gradient_input 占位符，
		 # 将状态批次 state_batch 传递给 self.state_input 占位符，将训练阶段的标志 True 传递给 self.is_training 占位符。
	def train(self,q_gradient_batch,state_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={self.q_gradient_input:q_gradient_batch, self.state_input:state_batch, self.is_training: True})


         #此函数接受一个状态批次 state_batch 作为输入，用于获取神经网络的输出动作。
	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={self.state_input:state_batch, self.is_training: True})

        # action(self, state) 函数：此函数接受一个状态 state 作为输入，用于获取在给定状态下的动作。
	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={self.state_input:[state], self.is_training: False})[0]
   
        #此函数接受一个状态批次 state_batch 作为输入，用于获取目标网络的输出动作。
	def target_actions(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={self.target_state_input: state_batch, self.target_is_training: True})

	# f fan-in size
	#定义了一个名为variable的函数，
	#shape 是方法的一个参数，表示要创建的变量的形状（shape）。
    #f 是方法的另一个参数，用于计算变量初始化的范围。使用tf.random_uniform函数来随机初始化这个Variable，初始化时设置范围为[-1/sqrt(f), 1/sqrt(f)],这是一个比较常见的初始化方式
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


    #定义了一个方法batch_norm_layer，用于应用批量归一化（batch normalization）层。归一化层可以提高模型的收敛性和泛化能力，训练和测试是不一样的，有False和True
	def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
		return tf.cond(training_phase, 
		lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True, updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
		lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True, updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))


    #self.saver 是类中的一个实例变量，表示 TensorFlow 的 Saver 对象，用于保存和恢复模型的权重。
    #model_dir 是类中定义的一个变量，表示模型保存的路径。
	#if语句表示如果有路径检测点，则更新相关的权重，否则仍然用原来的模型权重。
	
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(model_dir)        #使用tf.train.get_checkpoint_state()函数,传入模型保存的目录model_dir,获取checkpoint文件信息,赋值给checkpoint变量。
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded actoe network")
		else:
			print("Could not find old actor-network weights")      #开始训练之前也要导入网络吗？？？
      
     #定义了一个名为 save_network 的方法，用于保存演员网络的权重
	def save_network(self,time_step):
		print('save actor-network...',time_step)
		self.saver.save(self.sess, model_dir + 'actor-network', global_step=time_step)   #保存的位置应该有点问题，
		print('保存成功')

 