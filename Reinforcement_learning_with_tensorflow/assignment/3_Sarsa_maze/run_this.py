# @author: Ariel
# @time: 2021/10/11 21:28

from maze_env import Maze
from RL_brain import SarsaTable


def update():
    # 100个回合
    for episode in range(100):
        # 初始化状态S
        observation = env.reset()

        # ***基于状态S 选择动作A
        action = RL.choose_action(str(observation))

        while True:
            # 更新环境
            env.render()

            # 执行动作A 转化到新状态S'并返回即时奖励R
            observation_, reward, done = env.step(action)

            # 基于新状态S' 选择新动作A' ***Sarsa ε-贪婪法 探索的动作即为后续使用的动作 | Q Learning 贪婪法 不探索 采取最大Q值对应的动作
            action_ = RL.choose_action(str(observation_))
            # 更新价值函数 ***使用探索到的后续值更新
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # 更新状态
            observation = observation_
            # 更新动作 ***该动作直接作为后续要采取执行的动作
            action = action_

            if done:
                break

if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()