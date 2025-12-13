import gymnasium as gym
from tqdm import trange
import numpy as np
from agent import Agent

# 主函数
if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    # 定义状态和动作空间的大小
    num_states = env.observation_space.n
    # print(num_states)
    num_actions = env.action_space.n
    env.reset()
    # 训练
    episodes = 3000   # -0.889
    iterations = 10
    learningtype = [1,2,3,4]
    alpha = 0.1
    beta = 0.0001
    zeta = 0.005
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
            np.save('../results/cliff_walking/GQ.npy', numpy_all_steps)
        elif i == 2:
            np.save('../results/cliff_walking/CGQ.npy', numpy_all_steps)
            # np.save('data13/PTD_new_2500_3_beta_0.0001_1_omga_-0.6_regularization_L1_10.0.npy', numpy_all_omgas)
        elif i == 3:
            np.save('../results/cliff_walking/Q.npy', numpy_all_steps)
        elif i == 4:
            np.save('../results/cliff_walking/CQ.npy', numpy_all_steps)
                # 0：TD(0)  1:TD(lambda)   2:GTD   3:GTD2   4:TDC   5:VM   6:VM(lambda)