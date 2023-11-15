#!/usr/bin/env python3
#这段代码实现了一个使用DDPG算法在Gazebo仿真环境中训练和测试智能体的主程序。
import rospy
import gym
import gym_gazebo
import numpy as np
import tensorflow as tf
from ddpg import *
from environment import Env

exploration_decay_start_step = 50000    #表示开始减小探索噪声方差的时间步数。
#state_dim = 14    #修改之后的
state_dim = 16    #状态维度为16
action_dim = 2    #动作维度为2
action_linear_max = 0.25  # m/s      #动作的最大线速度
action_angular_max = 0.5  # rad/s      #动作的最大角速度
is_training = False   


# 在定义网络函数之前，调用tf.reset_default_graph()，确保使用新的计算图     
#tf.reset_default_graph()                      #新加的


def main():
    rospy.init_node('ddpg_stage_1')     #初始化节点名为“ddpg_stage_1”的ROS节点
    env = Env(is_training)           #定义了一个环境，Env 类是自定义的环境类， “is_training”参数表示当前是否处于训练模式
    agent = DDPG(env, state_dim, action_dim)  #这一行代码创建了一个名为 agent 的智能体对象。DDPG 类是一个自定义的深度确定性策略梯度（Deep Deterministic Policy Gradient，DDPG）算法的实现。它接受环境对象 env，状态维度 state_dim 和动作维度 action_dim 作为参数。
    past_action = np.array([0., 0.])     #这一行代码创建了一个名为 past_action 的Numpy数组，用于存储先前的动作值。初始值为 [0., 0.]，表示在开始时没有先前的动作值。
    print('State Dimensions: ' + str(state_dim))   #State Dimensions是状态维度的意思，这句话就是输出状态维度是多少的意思
    print('Action Dimensions: ' + str(action_dim))    #相应的动作维度
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')   #动作的最大值

    if is_training:
        print('Training mode')
        avg_reward_his = []      #创建一个空列表 avg_reward_his，用于存储平均奖励的历史记录。
        total_reward = 0            #初始化一个变量 total_reward，用于累积每个回合（episode）中的总奖励。
        var = 1.    #初始化一个变量 var，用于控制动作选择时的探索程度（exploration）。在训练的早期阶段，智能体通常更加探索环境，随机选择动作，随着训练的进行，逐渐减小探索程度，更倾向于选择已知更好的动作。这里将 var 初始化为 1.0，表示最初具有完全探索的策略。
         
        #total_train_rounds= 0   # 新加的，，初始化回合计数器

        while True:
            state = env.reset()    #重置环境并获取初始状态。env.reset() 方法用于将环境重置为初始状态，并返回初始状态的观测值。
            one_round_step = 0    #初始化一个变量 one_round_step，用于计算当前回合（episode）的步数。初始值为 0。
            
           # done_or_max_steps = False  #  新加的     初始化一个变量 done_or_max_steps，用于判断是否达到终止条件或达到最大步数。
           
            while True:
                a = agent.action(state)   #调用智能体（agent）的 action 方法，根据当前状态 state 选择一个动作 a。智能体的 action 方法基于当前状态输入，通过神经网络计算出一个动作。
                #np.random.normal(a[0], var) 和 np.random.normal(a[1], var) 分别使用正态分布函数添加噪声到动作的线性和角度部分。这样做是为了在训练过程中增加一定的探索性，促使智能体探索更多的动作空间。
                #np.clip() 函数将添加噪声后的动作 a[0] 和 a[1] 限制在一定的范围内。线性部分 a[0] 被限制在 0 到 1 之间，角度部分 a[1] 被限制在 -0.5 到 0.5 之间。
                a[0] = np.clip(np.random.normal(a[0], var), 0., 1.)  
                a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)
                 #在训练过程中增加了一定的探索性，这个探索性是否可以优化？？？？


                 #state_, r, done, arrive = env.step(a, past_action) 调用环境的 step 方法，传入动作 a 和过去的动作 past_action，执行一步环境交互。返回新的状态 state_、奖励值 r、是否终止 done 和是否到达目标 arrive 的信息。
                 #调用智能体的 perceive 方法，将当前状态 state、动作 a、奖励值 r、新状态 state_ 和是否终止 done 作为输入，进行强化学习算法的训练和经验回放。
                state_, r, done, arrive = env.step(a, past_action)
                time_step = agent.perceive(state, a, r, state_, done)

                if arrive:
                    result = 'Success'
                else:
                    result = 'Fail'
                  
                  #在每个时间步 time_step 大于0时，累加当前的奖励值 r 到 total_reward 中。这样可以跟踪每个回合的奖励累计。
                if time_step > 0:
                    total_reward += r
                   
                   #在每个训练周期（每完成10000个时间步）结束时计算平均奖励，并将其添加到 avg_reward_his 列表中。
                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 10000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))     #进行四舍五入保留2为小数
                    print('Average Reward:',avg_reward_his)
                    total_reward = 0     #进行一轮结束之后，总奖励规0
                    
                    
                  #当时间步长达到5w时，开始减少探索因子，而是根据已有的经验来判断。后期所有的var=0
                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    var *= 0.9999
                  

                  #在每个时间步，将当前的动作 a 赋值给 past_action 变量，以便在下一个时间步使用。
                  #然后，将新的状态 state_ 赋值给 state 变量，以更新当前状态。
                  # 最后，增加 one_round_step 变量的值，用于计算在当前回合中经历的时间步数。这个变量会在每个回合结束后重置为0。它用于跟踪每个回合的持续时间，以便在输出中进行显示和记录。
                past_action = a
                state = state_
                one_round_step += 1
                   
                 #如果到达了目标（arrive 为真），则打印当前回合的步数、方差 var、时间步数 time_step 和结果。然后将 one_round_step 重置为0，以便开始下一个回合的计数。
                if arrive:
                    print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
                    one_round_step = 0
                     
                    #done_or_max_steps = True   #新加的

                   
                 #如果达到了终止条件（done 为真或者 one_round_step 达到了最大步数500），则打印当前回合的步数、方差 var、时间步数 time_step 和结果，并终止当前回合的循环。
                if done or one_round_step >= 500:
                    print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)

                    break
                       
                    
                   # done_or_max_steps = True   #新加的
                    
                # if done_or_max_steps:
                 #       total_train_rounds += 1
                  #      if total_train_rounds >= 2000:
                   #         break
                #done_or_max_steps = False
                #break
           
    else:
        print('Testing mode')
        while True:
            state = env.reset()      #重置环境并获取初始状态。env.reset() 方法用于将环境重置为初始状态，并返回初始状态的观测值。
            one_round_step = 0    #初始化一个变量 one_round_step，用于计算当前回合（episode）的步数。初始值为 0。

            while True:
                a = agent.action(state)     #智能体根据当前状态选择一个动作a
                a[0] = np.clip(a[0], 0., 1.)   #a[0]用于表示线速度，a[1]用于表示角速度#用 np.clip() 函数将动作的每个分量限制在一定的范围内，即将线速度动作限制在 [0, 1] 范围内
                a[1] = np.clip(a[1], -0.5, 0.5)     #角速度动作限制在 [-0.5, 0.5] 范围内。
                state_, r, done, arrive = env.step(a, past_action)   #接下来，智能体将动作 a 提交给环境，并观察环境对动作的响应。环境返回新的状态 state_，即时奖励 r，一个布尔值 done 表示是否达到终止条件，以及一个布尔值 arrive 表示是否到达目标。
                past_action = a   #将当前动作 a 存储在 past_action 变量中
                state = state_     #更新当前状态为新的状态 state_，并增加回合步数 one_round_step。这样就完成了一个回合的执行过程，然后进入下一个回合。
                one_round_step += 1


                if arrive:
                    print('Step: %3i' % one_round_step, '| Arrive!!!')
                    one_round_step = 0

                #if done:
                 #   print('Step: %3i' % one_round_step, '| Collision!!!')
                  #  break
                  #新改进的，没有到达或者时间步长大于500时，显示未到达。防止测试的时候遗址陷入死循环
                if done or one_round_step >= 500:
                    print('Step: %3i' % one_round_step, '| Collision!!!')
                    break
if __name__ == '__main__':
     main()
