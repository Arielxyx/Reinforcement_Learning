"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # 分别用于存储当前回合的状态、动作、奖励值
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # 建立 policy 神经网络
        self._build_net()

        self.sess = tf.Session()

        # 是否输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            # 接收 observation
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            # 接收在这个回合中选过的 actions
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            # 接收每个 state-action 所对应的 value (通过 reward 计算）
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # fc1 全连接层
        layer = tf.layers.dense(
            inputs=self.tf_obs, # 输入=观测值
            units=10, # 输出单元数=10
            activation=tf.nn.tanh,  # tanh activation 激励函数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), # 返回一个生成具有正态分布的张量的初始化器，权重矩阵的初始化函数
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2 全连接层
        all_act = tf.layers.dense(
            inputs=layer, # 输入=上一级输出
            units=self.n_actions, # 输出单元数=动作个数
            activation=None, # 无激励函数 因为最后一层需要加上Softmax
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # 对fc2的结果all_act 加上softmax 将动作值转化为概率值
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        # 计算损失函数 loss
        with tf.name_scope('loss'):

            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
                # sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
                # 计算过程：自动计算softmax → 计算交叉熵代价函数
                # 注意事项：logits=all_act 而不是all_act_prob（因为会自动计算softmax logits必须是没有经过tf.nn.softmax函数处理的数据 否则训练结果有问题）
                # logits：为神经网络最后一层的输出（内部会进行softmax）
                # labels：为神经网络期望的输出

                # 和tf.nn.softmax_cross_entropy_with_logits函数比较明显的区别：参数labels的不同 tf.nn.sparse_*的参数label是非稀疏表示的
                # 比如：一个3分类的样本标签 稀疏表示的形式为[0,0,1] 表示这个样本为第3个分类；而非稀疏表示就表示为2（因为从0开始算，0,1,2,就能表示三类）
                # 同理：[0,1,0]就表示样本属于第二个分类；而其非稀疏表示为1
                # 应用：因为深度学习中 图片一般是用非稀疏的标签的 所以用tf.nn.sparse_softmax_cross_entropy_with_logits的频率高

            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
                # 仅为第一个的展开形式
                # 这里注意：输入self.all_act_prob是一个n维向量
                # tf.one_hot(self.tf_acts，self.n_actions）：self.tf_acts是n维向量（多少行） self.n_actions表示热编码向量的维度（多少列）
                # 会产生self.tf_acts个热编码向量（1个one_hot张量，self.tf_acts作为indices的每一个值都进行独热编码），互相相乘得到一个值，全部求和

            # 强化学习：假设按照x(state)做的动作action(y)永远是对的 (出来的动作永远是"正确标签") 也永远按照"正确标签"修改自己的参数
            # 为确保该动作真的是正确标签：loss在原本的 cross - entropy 形式上乘以 vt（告诉cross - entropy 算出来的梯度是不是一个值得信任的梯度）
            # 如果vt小, 或者是负的, 说明这个梯度下降是一个错误的方向, 我们应该向着另一个方向更新参数
            # 如果vt是正的, 或很大, vt就会称赞cross - entropy出来的梯度, 并朝着这个方向梯度下降
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            # (vt = 本reward + 衰减的未来reward) 引导参数的梯度下降
            # tf.reduce_mean取平均

        # 进行训练
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        # 所有action的概率：状态obs插入新的一行维度 作为NN输入 并得到softmax值（概率值）
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # 根据概率来选 action
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        # .ravel() .flatten()：将多维数据降到一维 而ravel的变化会改变原来矩阵 flatten的变化不会影响原来的矩阵
        # .shape：读取矩阵的形状 shape[1]：读取矩阵第二维度的长度 表示总共有几个动作可以选择
        # numpy.random.choice(a, size=None, replace=True, p=None) 从数组a中选取维度size个随机数 size=None则选取一个 replace=True表示可重复抽取 p是a中每个数出现的概率 https://www.jb51.net/article/162012.htm
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # 衰减并标准化这回合的reward 使其更适合被学习
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        # 清空回合 data
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    # 用gamma衰减未来的reward 为了一定程度上减小回合variance
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # numpy.zeros_like(a) 返回一个零矩阵 与给定的矩阵相同形状
        discounted_ep_rs = np.zeros_like(self.ep_rs)

        running_add = 0
        # reversed()：返回序列seq的反向访问的迭代子
        # 如果self.ep_rs列表中存储有5个reward值 长度为5 则这里t取值为43210 倒序是为了可以取未来值计算
        # 当前状态s采取动作a获得的即时奖励：self.ep_rs[t]
        # 当前状态s采取动作a所获得的真实奖励：即时奖励 + 未来直到episode结束的奖励求和（running_add） * 衰减值
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 标准化reward
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs