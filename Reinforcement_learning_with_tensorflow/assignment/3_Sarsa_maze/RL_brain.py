# @author: Ariel
# @time: 2021/10/11 23:04

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
class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

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
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


