# @author: Ariel
# @time: 2021/10/11 17:05

from maze_env import Maze # 环境
from RL_brain import QLearningTable # 大脑

# ？？？observation str
def update():
    # 100个回合
    for episode in range(100):
        # 初始化状态值为(1,1)
        observation = env.reset()

        while True:
            # 更新环境
            env.render()

            # 基于状态S(observation) 选择动作A(action) ***每次基于状态S 都要重新使用ε-贪婪法 选择动作A 所以写在while循环里面
            action = RL.choose_action(str(observation))

            # 执行动作A(action) 转化到新的状态S'(observation_)并返回即时奖励R(reward) 跳到坑或到达终点(done=True)
            observation_, reward, done = env.step(action)

            # 更新价值函数 ***使用最大值更新
            RL.learn(str(observation), action, reward, str(observation_))
            # 更新状态
            observation = observation_

            if done:
                break

    print('game over')
    env.destroy()


if __name__ == '__main__':
    # 定义环境
    env = Maze()
    # 定义大脑
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # tkinter界面先关处理 无需过于关注
    env.after(100, update())
    env.mainloop()
    # 当前用tkinter实现环境
    # 但若环境变得更复杂 可以使用编好的环境 openai gym（格式类似于当前实现）