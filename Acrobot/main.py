from Agent import Agent
import gymnasium as gym
from tqdm import trange
import numpy as np

# 主函数
if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    # 设置分割精度
    # n_bins = 10
    # env.observation_space.shape[0] : state space: 6
    # bins = tuple([n_bins] * env.observation_space.shape[0])
    bins = (5, 5, 5, 5, 5, 5)
    # print(bins)
    # print(env.observation_space.high - env.observation_space.low)
    offset_pos = ((env.observation_space.high - env.observation_space.low) / 5) / (bins)
    # print(offset_pos)
    tiling_specs = [(bins, -2 * offset_pos),
                    (bins, -offset_pos),
                    (bins, tuple([0.0] * env.observation_space.shape[0])),
                    (bins, offset_pos),
                    (bins, 2 * offset_pos)]
    env.reset()
    # 训练
    episodes = 1500   # -0.889
    iterations = 5
    alpha = 0.1
    beta = 0.0001
    zeta = 0.1
    learningtype = [1,2,3,4]
    for i in learningtype:
        all_steps = []
        for iter in range(iterations):
            agent = Agent(env, tiling_specs, layers=5, features=234375, gamma=0.99, alpha=alpha, beta=beta, zeta=zeta)
            print('现在进行learningtype{}的第{}次迭代'.format(i, iter))
            steps = []
            omgas = []
            step = 0
            for episode in trange(episodes):
                step = agent.play_TDC(learningtype=i)
                steps.append(step)
                # print(step)
                omgas.append(agent.omga)
                agent.newEpisode()
                # print(agent.omga)
            all_steps.append(steps)
        numpy_all_steps = np.array(all_steps)
        if i == 1:
            np.save('../results/acrobot/GQ.npy', numpy_all_steps)
        elif i == 2:
            np.save('../results/acrobot/CGQ.npy', numpy_all_steps)
        elif i == 3:
            np.save('../results/acrobot/Q.npy', numpy_all_steps)
        elif i == 4:
            np.save('../results/acrobot/CQ.npy', numpy_all_steps)