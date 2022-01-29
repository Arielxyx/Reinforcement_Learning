"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v1'

###############################  DDPG  ####################################

class DDPG(object):
    # 初始化 DDPG
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # 定义 Actor
        with tf.variable_scope('Actor'):
            # Actor 当前网络
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            # Actor 目标网络
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        # 定义 Critic
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            # Critic 当前网络
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            # Critic 目标网络
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # 网络参数
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # 软更新
        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # 计算 Q现实 (q_target = R + gamma * q_)
        q_target = self.R + GAMMA * q_
        # 均方差损失 (1/m∑(target_q - q)^2)
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        # 优化器训练 Critic
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        # 反馈 Q 值 取反
        a_loss = - tf.reduce_mean(q)    # maximize the q
        # 优化器训练 Actor
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        # 激活所有全局参数
        self.sess.run(tf.global_variables_initializer())

    # 选择动作
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # 软更新
        # soft target replacement
        self.sess.run(self.soft_replace)

        # 记忆库 → 取出 BATCH_SIZE 个样本
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE) # 随机取样n个样本的索引
        bt = self.memory[indices, :] # n个样本索引 对应的数据
        bs = bt[:, :self.s_dim] # 拆分样本各部分信息 → b_s b_a b_r b_s_
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # 训练 Actor
        self.sess.run(self.atrain, {self.S: bs})
        # 训练 Critic
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    # 存储每次的记忆到 numpy array
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    # 定义 Actor 网络结构
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
    # 定义 Critic 网络结构
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

# 状态 有多少个
s_dim = env.observation_space.shape[0]
# 可选动作 有多少个
a_dim = env.action_space.shape[0]
# 可选动作 限制其范围
a_bound = env.action_space.high

# 初始化 DDPG
ddpg = DDPG(a_dim, s_dim, a_bound)

# 噪声系数
var = 3  # control exploration
t1 = time.time()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        # 基于当前状态 s → 选择动作 a
        a = ddpg.choose_action(s)
        # 对选出的动作 a → 增加噪声 N
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        # 执行动作 a → 新状态 s_ 奖励 r
        s_, r, done, info = env.step(a)

        # 记忆库 → 存储经验数据
        ddpg.store_transition(s, a, r / 10, s_)

        # 记忆库头一次满了之后
        if ddpg.pointer > MEMORY_CAPACITY:
            # 逐渐降低探索性
            var *= .9995    # decay the action randomness
            # 训练网络 DDPG
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)