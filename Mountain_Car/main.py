from Agent import Agent
import gymnasium as gym
from tqdm import trange
import numpy as np

# 主函数
if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    # 设置分割精度
    n_bins = 20
    # env.observation_space.shape[0] : state space: 6
    # bins = tuple([n_bins] * env.observation_space.shape[0])
    bins = (20, 20)
    # print(bins)
    offset_pos = (env.observation_space.high - env.observation_space.low) / (5 * n_bins)
    tiling_specs = [(bins, -2 * offset_pos),
                    (bins, -offset_pos),
                    (bins, tuple([0.0] * env.observation_space.shape[0])),
                    (bins, offset_pos),
                    (bins, 2 * offset_pos)]
    # for tiling_spec in tiling_specs:
    #     print(tiling_spec)
    env.reset()
    # 训练
    episodes = 1500   # -0.889
    iterations = 5
    omga_average = 0
    alpha = 0.1
    beta = 0.0001
    zeta = 0.001
    learningtype = [1,2,3,4]
    for i in learningtype:
        all_steps = []
        for iter in range(iterations):
            agent = Agent(env, tiling_specs, layers=5, features=6000, gamma=0.99, alpha=alpha, beta=beta, zeta=zeta)
            steps = []
            step = 0
            # print(agent.beta)
            for episode in trange(episodes):
                step = agent.play_TDC(learningtype=i)
                steps.append(step)
                agent.newEpisode()
            all_steps.append(steps)
        numpy_all_steps = np.array(all_steps)

        if i == 1:
            np.save('../results/mountain_car/GQ.npy', numpy_all_steps)
        elif i == 2:
            np.save('../results/mountain_car/CGQ.npy', numpy_all_steps)
        elif i == 3:
            np.save('../results/mountain_car/Q.npy', numpy_all_steps)
        elif i == 4:
            np.save('../results/mountain_car/CQ.npy', numpy_all_steps)