#实现了一个重放缓冲区（Replay Buffer），用于存储和提取训练样本。
from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):       #构造方法的定义，self是实例本身，buffer_size是参数
        self.buffer_size = buffer_size     #将传入的 buffer_size 值赋给类的实例变量 buffer_size，即self.buffer_size，用于指定重放缓冲区的最大容量。
        self.num_experiences = 0     #将经验计数器 num_experiences 初始化为0，用于记录当前缓冲区中存储的经验数量。
        self.buffer = deque()        #创建一个空的双端队列对象 buffer，用于存储经验。

    def get_batch(self, batch_size):     #get_batch 方法用于从回放缓冲区中随机抽取一个批次的经验样本。它接受一个参数 batch_size，表示要抽取的样本数量。
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)     #通过调用 random.sample 函数从回放缓冲区中随机选择 batch_size 个样本，并返回抽取的样本列表
                  #在优先经验回放中随即抽取样本，怎么的随机法是否可以优化

    def size(self):     #定义size函数，返回回放值的大小
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):       #add 方法用于向回放缓冲区中添加一条经验样本。它接受 state、action、reward、new_state 和 done 作为参数，表示一个完整的经验。
        experience = (state, action, reward, new_state, done)   #将这些经验数据封装成一个元组 experience。
        if self.num_experiences < self.buffer_size:     #如果数量没有达到最大容量，则继续往里加，达到了的话删除最早的经验
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):      #用于清空缓存队列
        self.buffer = deque()
        self.num_experiences = 0