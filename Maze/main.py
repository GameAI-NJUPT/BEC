import gymnasium as gym
from tqdm import trange
import numpy as np
from agent import Agent
from maze_env import MazeEnvironment

# 主函数
if __name__ == "__main__":
    maze_map = [[2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3],
                ]
    env = MazeEnvironment(maze_map)
    # 设置起始位置和目标位置
    env.start = (0, 0)
    env.goal = (12, 12)

    num_states = env.height * env.width
    num_actions = 4

    env.reset()
    # 训练
    episodes = 2000   # -0.889
    iterations = 50
    learningtype = [1,2,3,4]
    alpha = 0.1
    beta = 0.0001
    zeta = 0.001
    for i in learningtype:
        all_steps = []
        for iter in range(iterations):
            agent = Agent(env, features=num_states * num_actions, gamma=0.99, alpha=alpha, beta=beta, zeta=zeta)
            steps = []
            step = 0
            for episode in trange(episodes):
                step = agent.play_TDC(learningtype=i)
                steps.append(step)
                agent.newEpisode()
            all_steps.append(steps)
        numpy_all_steps = np.array(all_steps)

        if i == 1:
            np.save('../results/maze/GQ.npy', numpy_all_steps)
        elif i == 2:
            np.save('../results/maze/CGQ.npy', numpy_all_steps)
        elif i == 3:
            np.save('../results/maze/Q.npy', numpy_all_steps)
        elif i == 4:
            np.save('../results/maze/CQ.npy', numpy_all_steps)
