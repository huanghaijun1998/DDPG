#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)      #表示目标点的对角线距离。开庚号的意思，3.6和3.8表示机器人平台在前向和侧向方向上的最大位移量。
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')               #表示目标文件的路径
        #通过这个代码，找到需要更改的环境以及目标物的环境

class Env():      #构造方法初始化了环境对象
    def __init__(self, is_training):      #一个初始化的方法
        self.position = Pose()                 #机器人当前的位置（Pose对象）
        self.goal_position = Pose()       #目标位置的位置（Pose对象）
        self.goal_position.position.x = 0.       #目标位置的x坐标，初始值为0。
        self.goal_position.position.y = 0.       #目标位置的y坐标，初始值为0。
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)    #创建了一个名为pub_cmd_vel的发布者对象，它在'cmd_vel'主题上发布Twist类型的消息。它用于发送速度命令来控制机器人。
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)      #创建了一个名为sub_odom的订阅者对象，它订阅'odom'主题。它监听Odometry类型的消息，并在接收到新消息时调用getOdometry方法。
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)       #这些行代码创建了服务代理对象，用于调用Gazebo提供的特定服务。它们用于重置仿真、暂停和恢复物理引擎的运行。
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)     #这些行代码创建了用于在Gazebo中添加和删除模型的服务代理对象。它们用于在仿真环境中添加和移除目标模型
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.past_distance = 0.       #将past_distance属性初始化为0。它用于跟踪到目标位置的上一次距离。
        if is_training:
            self.threshold_arrive = 0.2     #threshold_arrive表示机器人到达目标位置的距离阈值。在这距离之内，表示到达目标点
        else:
            self.threshold_arrive = 0.4     #训练的时候是0.2，测试的时候是0.4

    def getGoalDistace(self):       #方法计算当前机器人位置与目标位置之间的距离。
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance     #这个距离被存储在goal_distance变量中，并且还将goal_distance的值赋给self.past_distance，以便在后续使用中进行比较和更新。

        return goal_distance

    def getOdometry(self, odom):         #方法获取机器人的位置和方向，并计算机器人与目标点之间的角度。
        self.position = odom.pose.pose.position      #位置
        orientation = odom.pose.pose.orientation   #方向
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))    #通过将角度转换为度数并进行四舍五入，得到了机器人的方向角度，存储在yaw变量中。

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360     #确保角度在0～360之间

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)    #到了机器人当前位置与目标位置在 x 和 y 轴上的相对距离。

        # Calculate the angle between robot and target       计算机器人和目标之间的角度
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)     #将计算得到的相对角度 rel_theta 与当前角度 yaw 相减得到角度差，

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)       #保证角度差为正！

        self.rel_theta = rel_theta         #相对角度
        self.yaw = yaw                              #当前角度
        self.diff_angle = diff_angle     #角度差异

    def getState(self, scan):           #方法根据激光扫描数据计算机器人的状态，包括扫描数据、距离、角度等信息。
        scan_range = []      #创建一个空列表，用于存储扫描数据的距离值。
        yaw = self.yaw        #将类的成员变量 self.yaw 的值赋给局部变量 yaw，表示机器人的当前角度。以下两行同理
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2     #设置最小距离阈值为 0.2。
        done = False       #初始化一个布尔变量 done，表示任务是否完成，默认为 False。
        arrive = False

        for i in range(len(scan.ranges)):     #遍历激光扫描数据中的每个测量值
            if scan.ranges[i] == float('Inf'):     #如果测量值为正无穷大（inf），说明激光传感器未检测到障碍物或测量值超出了传感器的有效范围。在这种情况下，将该值替换为3.5（或其他合适的代表无穷远的值）。
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):     #如果测量值为NaN（不是一个数字），说明激光传感器未能正确读取到该测量值。在这种情况下，将该值替换为0或其他适当的代表无效值的数值。
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])     #如果测量值既不是无穷大也不是NaN，说明激光传感器成功读取到了有效的测量值。将该值直接添加到scan_range列表中。

        if min_range > min(scan_range) > 0:      #意味着机器人距离最近的障碍物的距离小于min_range，达到了终止状态。此时，将done标志设置为True，表示任务结束。
            done = True

        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        if current_distance <= self.threshold_arrive:      
            # done = True
            arrive = True      #使用欧式距离判断机器人是否到达目标位置。

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):               #方法根据机器人的状态和行动，计算奖励值。
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        distance_rate = (self.past_distance - current_distance)   #上一次距离与当前距离的差值，这个值可以用来评估机器人的行动效果。

        reward = 500.*distance_rate
        self.past_distance = current_distance      #用500乘以差值来解决这个问题，离目标越来越近，奖励一直会是正值

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())
            #表示机器人到了终止状态，例如碰到障碍物或超过了预设的步数限制。在这种情况下，将奖励值设置为 -100，以惩罚机器人执行导致终止状态的行动。

        if arrive:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())     #到达目标，对机器人进行奖励，发送一个空速度，机器人停止运动
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')      #清除新的目标，以便生成新的目标模型

            # Build the target      建立目标
            rospy.wait_for_service('/gazebo/spawn_sdf_model')      #用于确保在生成新的目标物体之前，模拟环境中的 spawn_sdf_model 服务已经启动并可用。
            try:      #尝试生成一个新的目标物体，并将其添加到Gazebo仿真环境中。
                goal_urdf = open(goal_model_dir, "r").read()    #从指定的目标物体文件中读取URDF模型的内容。
                target = SpawnModel    #创建一个SpawnModel对象来定义要生成的目标物体。
                target.model_name = 'target'  # the same with sdf name     设置目标物体的名称，与SDF文件中的名称相对应。
                target.model_xml = goal_urdf      #设置目标物体的XML描述，即从文件中读取的URDF模型内容。
                self.goal_position.position.x = random.uniform(-3.6, 3.6)
                self.goal_position.position.y = random.uniform(-3.6, 3.6)     #在指定的范围内随机生成目标物体的横，纵坐标。
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')    #有点不懂，加载目标环境
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")      #如果生成目标物体异常，打印错误信息
            rospy.wait_for_service('/gazebo/unpause_physics')       #服务可用，以便在生成目标物体后继续仿真物理。
            self.goal_distance = self.getGoalDistace()      #获取当前机器人位置与目标位置之间的距离。
            arrive = False

        return reward

    def step(self, action, past_action):      #方法执行机器人的行动，发布速度控制命令，并获取机器人的状态和奖励值。
        linear_vel = action[0]     #从动作中获取线速度
        ang_vel = action[1]         #从动作中获取角速度

        vel_cmd = Twist()       #创建一个Twist 的对象 vel_cmd。
        vel_cmd.linear.x = linear_vel / 4    #将线速度设置为 linear_vel 的四分之一。
        vel_cmd.angular.z = ang_vel    #角速度不变
        self.pub_cmd_vel.publish(vel_cmd)     #发布器发布 vel_cmd，将控制命令发送给机器人。

        data = None   #等待并接收激光扫描数据。
        while data is None:      #初始为 None，用于存储激光扫描数据。
            try: 
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)        #等待接收名为 'scan' 的激光扫描消息，数据类型为 LaserScan，超时时间为 5 秒。并将值附给data
            except:
                pass

        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)    #将激光扫描数据 data 作为参数传入，并返回机器人的状态信息。这些信息包括扫描数据、距离、角度等。
        state = [i / 3.5 for i in state]     #对获取到的扫描数据进行归一化处理。通过遍历 state 列表中的每个元素，将其除以 3.5，然后将处理后的结果重新赋值给 state。这个归一化操作可以将扫描数据的范围缩放到 0 到 1 之间，方便后续的处理和计算。

        for pa in past_action:     #将过去的动作信息 past_action 添加到状态信息 state 。考虑过去对当前动作的影响
            state.append(pa)     

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]       #进行规一化处理
        reward = self.setReward(done, arrive)      #调用 setReward 方法，根据状态信息 done 和 arrive 计算奖励值 reward。
        ##这段代码的作用是将过去的动作信息与当前的状态信息合并，并对一些状态值进行归一化处理，同时计算奖励值。这些操作为后续的训练和决策提供了准备。##这段代码的作用是将过去的动作信息与当前的状态信息合并，并对一些状态值进行归一化处理，同时计算奖励值。这些操作为后续的训练和决策提供了准备。

        return np.asarray(state), reward, done, arrive

    def reset(self):            #方法重置环境，重新生成目标点，并获取初始状态。
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')      #等待 ROS 服务 /gazebo/delete_model 可用，该服务用于删除模型。
        self.del_model('target')    #调用 ROS 服务 /gazebo/delete_model，删除名为 'target' 的模型，以清除先前生成的目标点。

        rospy.wait_for_service('gazebo/reset_simulation')     #等待 ROS 服务 'gazebo/reset_simulation' 可用，该服务用于重置仿真环境。
        try:
            self.reset_proxy()     #重置仿真环境
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")     #重置失败，打印出错误信息

        # Build the targetz   建立目标z    它和目标（target）有什么区别与联系
        rospy.wait_for_service('/gazebo/spawn_sdf_model')     #用于确保在生成新的目标物体之前，模拟环境中的 spawn_sdf_model 服务已经启动并可用。
        try:     #尝试生成一个新的目标物体，并将其添加到Gazebo仿真环境中。
            goal_urdf = open(goal_model_dir, "r").read()   ##从指定的目标物体文件中读取URDF模型的内容。
            target = SpawnModel            #创建一个SpawnModel对象来定义要生成的目标物体。
            target.model_name = 'target'  # the same with sdf name     设置目标物体的名称，与SDF文件中的名称相对应。
            target.model_xml = goal_urdf      ##设置目标物体的XML描述，即从文件中读取的URDF模型内容。
            self.goal_position.position.x = random.uniform(-3.6, 3.6)
            self.goal_position.position.y = random.uniform(-3.6, 3.6)      ##在指定的范围内随机生成目标物体的横，纵坐标。

            # if -0.3 < self.goal_position.position.x < 0.3 and -0.3 < self.goal_position.position.y < 0.3:
            #     self.goal_position.position.x += 1
            #     self.goal_position.position.y += 1

            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')  
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")      #如果生成目标物体异常，打印错误信息
        rospy.wait_for_service('/gazebo/unpause_physics')    #服务可用，以便在生成目标物体后继续仿真物理。
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)    #等待接收名为 'scan' 的激光扫描消息，数据类型为 LaserScan，超时时间为 5 秒。并将值附给data
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)      #将激光扫描数据 data 作为参数传入，并返回机器人的状态信息。这些信息包括扫描数据、距离、角度等。
        state = [i / 3.5 for i in state]

        state.append(0)
        state.append(0)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)