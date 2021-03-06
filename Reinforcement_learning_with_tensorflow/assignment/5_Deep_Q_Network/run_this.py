# @author: Ariel
# @time: 2021/10/12 23:16

from maze_env import Maze
from RL_brain import DeepQNetwork

def run_maze():
    # 用于记录当前走到第几步 因为先要存储一些记忆 一开始还不能学习
    step = 0

    for episode in range(100):
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            # 存储记忆
            RL.store_transition(observation, action, reward, observation_)
            # 大于200步 且 每5步学习一次 保证经验充足
            if (step > 200) and (step % 5 == 0):
                # RL.learn(str(observation), action, reward, str(observation_))
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1

    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()