# @author: Ariel
# @time: 2021/10/11 13:52

import numpy as np
import pandas as pd
import time # 探索者的移动速度

np.random.seed(2) # 伪随机数列 每次运行生成的数列相同

# 定义全局变量
N_STATES = 6 # 状态数（一维世界的维度为6）
ACTIONS = ['left', 'right'] # 可行的动作（左、右）
EPSILON = 0.9 # ε-贪婪法 选择动作（90%选择最优动作、10%选择随机动作）
ALPHA = 0.1 # α 学习率（学习误差的效率）
GAMMA = 0.9 # γ 衰减因子（越是后面的状态 奖励权重越小）
MAX_EPISODES = 13 # 最大训练轮数
FRESH_TIME = 0.1 # 每0.3秒移动一次

# 建立Q-table
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # 全0初始化 状态数 x 动作数
        columns = actions, # 列命名为动作名称
    )
    # print(table)
    return table

# build_q_table(N_STATES, ACTIONS)

# 基于状态S 借助Q-table 选择动作A
def choose_actions(state, q_table):
    state_actions = q_table.iloc[state, :] # 位置索引 状态为state的行 所有列
    if (np.random.uniform() > EPSILON or (state_actions.all() == 0)): # 均匀分布中随机采样值>0.9则有10%的概率选择非最优动作 值全为0
        action_name = np.random.choice(ACTIONS) # 随机选择动作
    else:
        action_name = state_actions.idxmax() # 选择值最大的标签（列/动作名称） ：argmax-Series idxmax-DataFrame
    return action_name

# 创建环境及环境对于的反馈 执行动作后 返回新的状态S'和即时奖励R
def get_env_feedback(S,A):
    if A == 'right': # 向右移动
        if S == N_STATES - 2: # 状态标号0-5 S=4时并向右移动 则新的状态会到达终点
            S_ = 'terminal'
            R = 1
        else: # 新的状态标号加一
            S_ = S + 1
            R = 0
    else: # 向左移动
        R = 0
        if S == 0: # 初始状态无法再往左移动
            S_ = S
        else: # 新的状态标号减一
            S_ = S - 1
    return S_, R

# 更新环境 控制输出
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # 初始状态 '---------T' our environment
    if S == 'terminal': # 到达目标状态
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter) # 每一回合走了多少步
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else: # 没有到达目标状态
        env_list[S] = 'o'
        interaction = ''.join(env_list) # 模拟当前所处位置环境
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    # 1. 创建Q_table
    q_table = build_q_table(N_STATES, ACTIONS)
    # 2. 做MAX_EPISODES个回合
    for episode in range(MAX_EPISODES):
        # 3. 初始化
        step_counter = 0
        S = 0 # 最左边
        is_terminated = False # 现在还没到终点 到达终点则这一回合结束
        update_env(S, episode, step_counter) # 每一回合先更新环境

        # 4. 每回合直到终点状态结束
        while not is_terminated:
            # 4.1 基于状态S 借助Q-table 根据ε-贪婪法 选择动作A
            A = choose_actions(S, q_table)
            # 4.2 执行动作A 转化到新状态S' 并返回即时奖励R
            S_, R = get_env_feedback(S, A)
            # 4.3 估计值
            q_predict = q_table.loc[S, A] # A为column名称 所以不能使用iloc iloc适用于位置索引 loc适用于名称索引
            # 4.4 预测值Gt
            if S_ != 'terminal': # 下一个状态非终点
                q_target = R + GAMMA*q_table.iloc[S_, :].max() # （预测值Gt） = 即时奖励R + 衰减因子λ * 贪婪法选择Q最大值
            else: # 下一个状态为终点
                q_target = R # 仅有即时奖励 没有后续状态
                is_terminated = True
            # 4.5 更新Q_table
            q_table.loc[S,A] += ALPHA * (q_target-q_predict)
            # 4.6 更新状态
            S = S_

            update_env(S, episode, step_counter)  # 状态更新后更新环境 做相关输出
            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)