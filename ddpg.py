#!/usr/bin/env python3
import gym
import tensorflow as tf
import numpy as np
from critic_network import CriticNetwork     #这3个是导入了自定义的模块
from actor_network_bn import ActorNetwork
from replay_buffer import ReplayBuffer

REPLAY_BUFFER_SIZE = 100000     #经验回放缓冲区的大小，它指定了可以存储的过去交互样本的数量。
REPLAY_START_SIZE = 10000    #经验回放开始训练的阈值。当经验回放缓冲区中的样本数量达到该阈值时，算法开始使用经验回放进行训练。
BATCH_SIZE = 128      #每个训练步骤中从经验回放缓冲区中随机采样的样本数量。这些样本将被用于更新Critic网络和Actor网络的参数。
GAMMA = 0.99     #折扣因子，用于计算未来奖励的衰减量。在计算目标Q值时，将当前奖励乘以折扣因子的幂次，以考虑未来奖励的影响。


class DDPG:
    def __init__(self, env, state_dim, action_dim):      #括号里面的为参数
        self.name = 'DDPG'   #将算法的名称设置为'DDPG'，存储在实例变量self.name中。
        self.environment = env    #存储环境对象env到实例变量self.environment中，以便在算法中使用。 
        self.time_step = 0     #初始化时间步self.time_step为0，用于跟踪算法运行的总步数。运行的总步数怎么计算
        self.state_dim = state_dim     #存储状态空间维度state_dim到实例变量self.state_dim中。
        self.action_dim = action_dim   #存储动作空间维度action_dim到实例变量self.action_dim中。
        self.sess = tf.InteractiveSession()   #创建一个交互式的TensorFlow会话对象，并将其存储在实例变量self.sess中。

          #通过创建这两个网络的实例，DDPG算法可以利用这些网络进行训练和测试，从而实现强化学习任务的目标。
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)    #创建了演员网络的实例。ActorNetwork是一个类，里面有3个参数
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)
        #这里是否需要做更高改？

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)     #创建了一个回放缓冲区对象。

    def train(self):    #class为类，def 是定义的关键字，后面的都表示相关的算法，此处为train方法。self代表类的实例对象本身
        #在训练过程中，首先从replay buffer中获取一个batch的数据样本，用于训练网络。每个样本包括当前状态、动作、奖励、下一个状态和完成标志。
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])       #用于更新critic网络    有点不太理解
        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])      #为了保持一致性，使用NumPy库的resize函数对action_batch进行调整大小操作，将其调整为指定的维度[BATCH_SIZE, self.action_dim]。

        next_action_batch = self.actor_network.target_actions(next_state_batch)     #next_action_batch是通过调用self.actor_network对象的target_actions方法获取的下一个状态的动作批次。
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)    #同理
        #与Q值网络有关
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        # Update critic by minimizing the loss L    通过最小化损失来更新critic L
        self.critic_network.train(y_batch, state_batch, action_batch)    #是调用评论者网络（Critic Network）的训练方法，用于更新评论者网络的参数。

        # Update the actor policy using the sampled gradient:  使用采样的渐变更新actor策略：
        #action_batch_for_gradients是用于计算梯度的动作批次。在DDPG算法中，用于更新演员网络的梯度是根据状态批次（state_batch）和演员网络（actor_network）计算的动作值。
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        #q_gradient_batch是根据评论家网络（critic_network）在给定状态批次（state_batch）和动作批次（action_batch_for_gradients）下计算得到的Q值梯度
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)
         
      #使用Q值梯度来更新演员网络的方法。
        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()           #更新目标网络的方法
        self.critic_network.update_target()
       
        #action(self, state)方法用于根据给定的状态(state)生成一个动作(action)。
    def action(self, state):
        action = self.actor_network.action(state)

        return action
        

        #方法用于接收一个完整的环境转换信息，并将其存储到回放缓冲器中，然后根据需要进行训练。
    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)   #将当前环境转换信息存储到回放缓冲器中
        if self.replay_buffer.count() == REPLAY_START_SIZE:       #检查回放缓冲器中存储的转换数量是否等于启动训练的阈值。如果是，则打印一条提示消息，表示开始训练。
            print('\n---------------Start training---------------')
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:     #如果是，则增加时间步数（self.time_step）并调用self.train()方法开始进行训练。
            self.time_step += 1
            self.train()

           #这部分代码用于定期保存训练过程中的模型权重。每1w个时间步保存一次！
        if self.time_step % 10000 == 0 and self.time_step > 0:
            self.actor_network.save_network(self.time_step)
            print('\n---------------保存actor网络---------------')                 #自己新加的
            self.critic_network.save_network(self.time_step)
            print('\n---------------保存critic网络---------------')                 #自己新加的

        return self.time_step










