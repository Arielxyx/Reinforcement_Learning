"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

GAME = 'Pendulum-v1'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)

# 观测值个数
N_S = env.observation_space.shape[0]
# 行为值个数
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]

# 可以被调用生成一个 Global net 也能被调用生成一个 Worker net
# 因为他们的结构是一样的 所以这个 class 可以被重复利用
class ACNet(object):
    def __init__(self, scope, globalAC=None): # scope 用于确定生成什么网络

        # Global Net：仅关注参数
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        # Local Net：要进行训练
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A') # 初始化action 是一个[batch，1]的矩阵 第二个维度为1 格式类似于[[1],[2],[3]]
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                # 训练 Actor Critic
                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                # Critic：计算 TD-error = (v现实 - v估计)
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td)) # tf.square 加平方避免负数

                # 对网络训练选出的动作 其概率进行正态分布化
                with tf.name_scope('wrap_a_out'):
                    # 均值 标准差
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                # 正态分布
                normal_dist = tf.distributions.Normal(mu, sigma)
                # Actor：计算 损失值 = logprob * td + ENTROPY_BETA * entropy
                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his) # logprob对动作概率求对数
                    exp_v = log_prob * tf.stop_gradient(td) # td决定梯度下降的方向
                    entropy = normal_dist.entropy()  # c增加随机探索度
                    self.exp_v = ENTROPY_BETA * entropy + exp_v # logprob * td + ENTROPY_BETA * entropy
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                # Actor：计算 dloss/dparams 用于反向传播更新参数
                with tf.name_scope('local_grad'):
                    # 实现 a_loss 对 a_params 每一个参数的求导 返回一个 list
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    # 实现 c_loss 对 c_params 每一个参数的求导 返回一个 list
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

                # 执行选择动作 a
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])

            # 同步
            with tf.name_scope('sync'):
                # 从 Local Net 推送到 Global Net：更新全局参数
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                # 从 Global Net 拉取到 Local Net：同步最新信息
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]

    # 调用 选择动作 a
    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})

    # 构建网络结构
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        # Actor 选择动作
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        # Critic 计算 v 值
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        # 返回 均值 方差 状态价值 Actor参数 Critic参数
        return mu, sigma, v, a_params, c_params

    # 调用 push
    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    # 调用 pull
    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])


class Worker(object):
    def __init__(self, name, globalAC):
        # 自己的环境
        self.env = gym.make(GAME).unwrapped
        # 自己的名字
        self.name = name
        # 自己的 Local Net：要绑定上 Global Net
        self.AC = ACNet(name, globalAC)

    def work(self):
        # 两个全局变量 R是所有worker的总reward ep是所有worker的总episode
        global GLOBAL_RUNNING_R, GLOBAL_EP
        # 本 Worker 的总步数
        total_step = 1
        # s, a, r 的缓存, 用于 n_steps 更新
        buffer_s, buffer_a, buffer_r = [], [], []
        # 本循环一次是一个回合
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            # 初始化环境
            s = self.env.reset()
            # 本回合总的reward
            ep_r = 0

            for ep_t in range(MAX_EP_STEP):
                # 只有worker0才将动画图像显示
                # if self.name == 'W_0':
                #     self.env.render()

                # *** 基于状态 s → 选择动作 a
                a = self.AC.choose_action(s)
                # 执行动作 a → 新状态 s_ 奖励 r 是否完成 done
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False

                # 记录本回合总体reward
                ep_r += r
                # 添加各种缓存
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize

                # 每 UPDATE_GLOBAL_ITER 步 or 回合完了 进行 sync 操作
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    # 如果回合结束 设定下一个状态的价值为0
                    if done:
                        v_s_ = 0   # terminal
                    # *** 如果是中间步骤 训练AC网络 计算下一个状态的价值
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]

                    buffer_v_target = []
                    # 计算当前回合中 每个状态的价值：反向衰减传递得到每一步的v现实 = Rt + γRt+1 + γ^2Rt+2 + ... + γ^n-1Rt+n-1 + γ^nV(s')
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        # 将每一步的v现实都加入缓存中
                        buffer_v_target.append(v_s_)
                    # 再次反向 得到本系列操作每一步的v现实(v-target)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }

                    # *** 推送更新到 Global Net
                    self.AC.update_global(feed_dict)
                    # 清空缓存
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # *** 获取 Global Net 的最新参数
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    # 创建 tensorflow session 会话
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # 训练 Actor
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        # 训练 Critic
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        # 创建 Global AC：仅包含全局参数 不负责训练
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        # 创建 多个 Worker：每个负责自身的训练工作
        workers = []
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))  # 每个 worker 都有共享这个 global AC

    # 并行工具
    COORD = tf.train.Coordinator()
    # 激活所有全局参数
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    # 启动 多个 Worker
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    # Worker 加入线程调度：当所有 Worker 运行完时才能继续向下执行
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

