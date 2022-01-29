# @author: Ariel
# @time: 2021/10/12 10:09


import numpy as np
import pandas as pd

# Q Learning 与 Sarsa 公共部分
class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)  # 定义空的Q_table 列名为actions名（此处实际选取的是标号 0 1 2 3）

    # 一维中固定了长度
    # 二维中不知道有多少个state 需要检验下一个state 是已经经历过的还是从来没经历过的 若从没经历则加入Q_table
    def check_state_exists(self, state):
        # 状态state不存在 则添加一个状态行到Q_table
        if state not in self.q_table.index:  # .index 查看所有索引行 .columns 查看所有列名称
            self.q_table = self.q_table.append(
                pd.Series(  # Series 一维
                    [0] * len(self.actions),  # 初始化为列数个0
                    index=self.q_table.columns,  # 一维的索引 = 二维的列
                    name=state  # 表头的值
                )
            )

    def choose_action(self, observation):
        # 检验状态
        self.check_state_exists(observation)
        # 根据epsilon选择动作
        if np.random.uniform() > self.epsilon:  # 随机
            action = np.random.choice(self.actions)
        else:  # 最优
            state_actions = self.q_table.loc[observation, :]  # observation为一个坐标 非数值 不可使用iloc
            # 假设有多个action 并且在Q_table中的值相等 则使用idxmax()取最大值时只会选择第一个action
            # np.random.permutation 随机打乱所有action的索引位置
            # state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            # action = state_actions.idxmax()
            # 选取 与最大值Q值相等的 多个动作actions 并取出其列名
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        # 检验状态
        self.check_state_exists(s_)
        # 估计值
        q_predict = self.q_table.at[s,a]
        # 真实值
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


# on-policy
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        # ***步数衰减λ
        self.lambda_ = trace_decay
        # ***效用表结构与Q_table相同 经历则+1 不经历则衰减
        self.eligibility_trace = self.q_table.copy()

    def check_state_exists(self, state):
        # 状态state不存在 则添加一个状态行到Q_table *** 并且添加该状态到效用表中
        if state not in self.q_table.index:  # .index 查看所有索引行 .columns 查看所有列名称
            to_be_append = pd.Series(  # Series 一维
                    [0] * len(self.actions),  # 初始化为列数个0
                    index=self.q_table.columns,  # 一维的索引 = 二维的列
                    name=state  # 表头的值
                )
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        # 检验状态
        self.check_state_exists(s_)
        # 估计值
        q_predict = self.q_table.at[s, a]
        # 真实值
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_] # 使用确定的后续动作 不用贪婪选择最大值的行动
        else:
            q_target = r
        # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

        # ***误差
        error = q_target - q_predict
        # ***当前状态效用值增加
        # self.eligibility_trace.loc[s, a] += 1 # 不限峰值
        self.eligibility_trace.loc[s, :] *= 0 # 该状态下的所有动作效用值设为0
        self.eligibility_trace.loc[s, a] = 1 # 该状态下的当前动作效用值设为1 峰值即为1

        # 所有步都是有效的 只不过根据效用值决定 离得越近效用值越大 离得越远效用值越小 所以要更新所有状态
        # self.q_table.loc[s, a] += self.lr * error * self.eligibility_trace.loc[s, a] # 不仅仅只更新当前状态 所有状态Q值更新0..
        self.q_table += self.lr * error * self.eligibility_trace

        # ***不仅仅只更新当前状态 所有状态效用值衰减
        self.eligibility_trace *= self.gamma * self.lambda_