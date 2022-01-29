# @author: Ariel
# @time: 2021/10/13 8:55

import numpy as np
import pandas as pd
import tensorflow as tf

class DeepQNetwork:
    # 初始化各种变量
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01, # α 学习率
                 reward_decay=0.9, # γ 衰减因子
                 e_greedy=0.9, # ε-贪婪法
                 replace_target_iter=200, # 每隔C步 更新一次目标网络Q的参数w'=w
                 memory_size=500, # 经验记忆库的大小 多少条n_features*2+2元组
                 batch_size=32, # 随机梯度下降 批次大小
                 e_greedy_increment=None, # 逐渐减小探索ε的概率
                 output_graph=False # 输出tensorboard
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0 # 记录已经学习了多少步
        self.memory = np.zeros((self.memory_size, n_features*2+2)) # 记忆库 shape[记忆库条数，n_features*2+2元组]

        # 建立神经网络
        self._build_net()
        # 建立tensorflow会话
        self.sess = tf.Session()
        # 激活global_variables_initializer
        self.sess.run(tf.global_variables_initializer())

        # 输出tensorboard
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 记录每一步的误差
        self.cost_his = []

    # 建立两个网络 当前网络Q-训练参数 目标网络Q'-计算目标Q值
    def _build_net(self):
        # ------------- build evaluate_net -------------
        # tf.placeholder(dtype, shape, name)
        # dtype：数据类型 常用tf.float32, tf.float64等数值类型
        # shape：数据形状 默认None 就是一维值 也可以是多维（比如[2, 3], [None, 3] 表示列是3行不定）
        # name：名称
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') # 当前状态s
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # Q现实 与Q估计做差 反向传播更新参数

        # 定义估计网络Q 名称为eval_net
        with tf.variable_scope('eval_net'): # 绝大部分情况下会和tf.get_variable()配合使用，实现变量共享的功能
            # 默认参数定义
            # c_names 搜集网络中的所有参数 默认加入所有的Variable对象 并且在分布式环境中共享
            # n_l1  第一层 输出10个神经元
            # w_initializer 权重 初始化器：随机生成具有正态分布的张量 tf.random_normal_initializer(mean=0.0, stddev=0.3, seed=None, dtype=tf.float32)
            # b_initializer 偏置 初始化器：生成具有常量值的张量 tf.constant_initializer(value=0.1, dtype, verify_shape)
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                                                          10, \
                                                          tf.random_normal_initializer(0., 0.3), \
                                                          tf.constant_initializer(0.1)

            # 定义第一层网络
            with tf.variable_scope('l1'):
                # tf.get_variable(name, shape=None, initializer=None, collections=None) 若变量存在，返回现有变量；若变量不存在，根据给定形状和初始值创建变量。collections指定变量w1所属的集合
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # 第一层网络输出 l1 = s*w1 + b1
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # 定义第二层网络
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # 第二层网络输出 q_eval = l1*w + b2
                self.q_eval = tf.matmul(l1, w2) + b2

            # 损失函数
            with tf.name_scope('loss'): # 为了更好的管理变量的命名空间
                self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval)) # 按一定方式计算张量中元素之和

            # 优化器
            with tf.name_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------- build target_net -------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # 新状态s_
        # 定义目标网络Q' 名称为target_net（与估计网络Q 结构相同 参数不同）
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 存储经验记忆
    def store_transition(self, s, a, r, s_):
        # 如果不存在变量memory_counter 说明存储的是第一条经验数据 则创建并设置其计数器值为0（初始化索引值为0）
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # np.hstack 水平方向上平铺为一行
        transition = np.hstack((s, [a, r], s_))
        # 计算插入位置的索引值
        index = self.memory_counter % self.memory_size
        # 插入经验记忆表
        self.memory[index,:] = transition

        # 计数器+1
        self.memory_counter += 1

    # 选择动作
    def choose_action(self, observation):
        # 输入时为一维数据 为了tensorflow能处理 增加一维变成二维
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon: # 10% 随机选择
            action = np.random.randint(0, self.n_actions)
        else: # 90% 选择最大值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)

        return action


    # 更新目标网络Q'的参数
    def _replace_target_params(self):
        # 返回各自的参数列表
        t_params = tf.get_collection('target_net_params') # 目标网络Q'
        e_params = tf.get_collection('eval_net_params') # 当前网络Q
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)]) # e赋值给t 因为本身是列表 所以可以zip打包后 再for遍历每一项 做赋值assign操作


    def learn(self):
        # 每replace_target_iter步 更新一次目标网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        # 调用经验记忆库中的记忆 → 采取样本数集
        # 随机选取batch_size批次个记忆样本 如果库中的记忆不够 就只抽取记忆库中已存取的所有记忆 np.random.choice(self.memory_size, size=self.batch_size)
        # batch_memory = self.memory.sample(self.batch_size) \
        #     if self.memory_counter > self.memory_size\
        #     else self.memory[:self.memory_counter].sample(self.batch_size, replace=True)
        batch_memory = self.memory[np.random.choice(self.memory_size, size=self.batch_size),:] \
            if self.memory_counter > self.memory_size\
        else self.memory[np.random.choice(self.memory_counter, size=self.batch_size),:]

        # 运行神经网络 → Q下一个 Q估计
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                # 存储记忆 RL.store_transition(observation, action, reward, observation_)
                self.s_: batch_memory[:,-self.n_features:], # 目标网络Q'输入：抽取记忆中 所有样本 每个样本的状态 后四个特征feature
                self.s: batch_memory[:,:self.n_features] # 当前网络Q输入：抽取记忆中 所有样本 每个样本的状态 后前个特征feature
            }
        )

        # 先复制 再赋值 → Q现实
        # 为了有效的反向传播 q_next选择出来的对应动作 应以 q_eval选出来的对应动作 为基准
        q_target = q_eval.copy()
        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features + 1]
        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_target[np.arange(self.batch_size, dtype=np.int32),
                 batch_memory[:, self.n_features].astype(int)] \
            = batch_memory[:, self.n_features + 1] + self.gamma * np.max(q_next, axis=1)

        # 优化器 损失函数 → 计算Q估计和Q现实的误差 更新参数
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # 增加epsilon的值 前期探索 后期保守
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # 增加学习步数
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
