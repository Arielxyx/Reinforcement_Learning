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


np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount

REPLACEMENT = [
    # 软更新：系数tau 选择多少进行更新
    dict(name='soft', tau=0.01),
    # 硬更新：Actor 600轮更新一次参数 θ； Critic 500轮更新一次参数 w
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies

MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v1'

###############################  Actor  ####################################

# Actor 选择动作
class Actor(object):

    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        # 定义 Actor
        with tf.variable_scope('Actor'):

            # Actor当前网络：选择 当前动作 a → 供 Critic 计算 Q估计 q (Q(s, a, w)) *** → 及时更新参数 θ
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # Actor目标网络：选择 下一个动作 a_ → 供 Critic 计算 Q现实 target_q (Q(s_, a_, w)) ***
            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        # 定义硬更新：更新全部参数 θ' = θ
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        # 定义软更新：更新部分参数 θ' = tau*θ + (1-tau)*θ'
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    # 定义网络结构
    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            # s → 30
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)

            with tf.variable_scope('a'):
                # 30 → a
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                # 限制 输出动作 范围 ∈(-action_bound, action_bound)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    # 调用 训练 Actor
    def learn(self, s):   # batch update
        # 调用 优化器训练 Actor →
        # 计算 确定性策略损失 (dQ/da * da/dparams) →
        # 使用 Q 梯度 (dQ/da)
        self.sess.run(self.train_op, feed_dict={S: s})

        # soft：每次 都更新部分参数
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        # hard：每隔一定的步数 更新全部参数
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    # 选择动作
    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        # 基于当前状态 s → 选择动作 a
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    # 执行 训练 Actor
    def add_grad_to_graph(self, a_grads):
        # *** 确定性策略损失 Actor
        with tf.variable_scope('policy_grads'):
            # ys = policy; 动作 a
            # xs = policy's parameters; 对 参数 θ 求导
            # a_grads = the gradients of the policy to get more Q; 由 Critic 求得的 动作a的初始梯度值 dQ/da
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dQ/da * da/dparams
            # 计算 dQ/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        # 优化器训练 Actor
        with tf.variable_scope('A_train'):
            # 负的学习率为了使我们计算的梯度往上升 和 Policy Gradient 中的方式一个性质
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

# Critic 计算 Q 值
class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        # 定义 Critic
        with tf.variable_scope('Critic'):

            # Critic当前网络：计算 Q估计 q_eval → 使用到了 Actor计算的 a *** → 及时更新参数 w
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Critic目标网络：计算 Q下一个 q_next → 使用到了 Actor计算的 a_ ***
            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        # 计算 Q现实 target_q
        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        # 均方差损失 TD-error
        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        # 优化器训练 Critic
        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # *** 计算 Q梯度 dQ/da
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        # 定义硬更新：更新全部参数 w' = w
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        # 定义软更新：更新部分参数 w' = tau*w + (1-tau)*w'
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    # 定义网络结构
    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            # s、a → 30
            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            # 30 → q
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    # 调用 训练 Critic
    def learn(self, s, a, r, s_):
        # 调用 优化器训练 Critic →
        # 计算 均方差损失 TD-error (1/m∑(target_q - q)^2) →
        # 计算 Q现实 (target_q = R + gamma * q_ → _build_net('eval_net')) & Q估计 (q → _build_net('target_net'))
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})

        # soft：每次 都更新部分参数
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        # hard：每隔一定的步数 更新全部参数
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory 记忆库  ####################

class Memory(object):
    # 初始化记忆库
    def __init__(self, capacity, dims):
        self.capacity = capacity # 容量
        self.data = np.zeros((capacity, dims)) # 容量*每条记忆的维度
        self.pointer = 0 # 指针指向第0条记忆

    # 存储每次的记忆到 numpy array
    def store_transition(self, s, a, r, s_):
        # 存储的记忆
        transition = np.hstack((s, a, [r], s_))
        # 存储的位置
        index = self.pointer % self.capacity  # replace the old memory with new memory
        # 执行存储
        self.data[index, :] = transition
        # 指针+1
        self.pointer += 1

    # 取样
    def sample(self, n):
        # 记忆库存满了才能继续学习 否则取样失败
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # 随机取样n个样本的索引
        indices = np.random.choice(self.capacity, size=n)
        # n个样本索引 对应的数据
        return self.data[indices, :]


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

# 状态 有多少个
state_dim = env.observation_space.shape[0]
# 可选动作 有多少个
action_dim = env.action_space.shape[0]
# 可选动作 限制其范围
action_bound = env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
# 将 actor 同它的 eval_net/target_net 产生的 a/a_ 传给 Critic
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
# 将 critic 产出的 dQ/da 加入到 Actor 的 Graph 中去
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

# 噪声系数
var = 3  # control exploration

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):
        # 显示模拟
        if RENDER:
            env.render()

        # Add exploration noise
        # 基于当前状态 s → 选择动作 a
        a = actor.choose_action(s)
        # 对选出的动作 a → 增加噪声 N
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        # 执行动作 a → 新状态 s_ 奖励 r
        s_, r, done, info = env.step(a)

        # 记忆库 → 存储经验数据
        M.store_transition(s, a, r / 10, s_)

        # 记忆库头一次满了之后
        if M.pointer > MEMORY_CAPACITY:
            # 逐渐降低探索性
            var *= .9995    # decay the action randomness
            # 记忆库 → 取出 BATCH_SIZE 个样本
            b_M = M.sample(BATCH_SIZE)
            # 拆分样本各部分信息 → b_s b_a b_r b_s_
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            # 训练 Critic（计算 Q 值）
            critic.learn(b_s, b_a, b_r, b_s_)
            # 训练 Actor （基于状态 选择动作）
            actor.learn(b_s)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break

print('Running time: ', time.time()-t1)