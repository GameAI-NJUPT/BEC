import numpy as np


class Agent:
    def __init__(self, env, features, gamma=0.99, epsilon=0.01, alpha=0.1, zeta=0.01, beta = 0.0001):
        self.env = env
        # self.env = self.env.unwrapped
        self.state_n = env.observation_space.n  # 状态空间
        self.action_n = env.action_space.n  # 动作数
        self.features = features
        self.w = np.zeros(features)  # 权重初始化为0
        self.psi = np.zeros(features)
        self.gamma = gamma  # 折扣
        self.alpha = alpha
        self.etd_f = 1
        self.zeta = zeta
        self.beta = beta
        self.omga = .0
        self.lamda = 0.8
        self.epsilon = epsilon
        self.eligibility = np.zeros(features)

    def init(self):
        self.w = np.zeros(self.features)  # 权重初始化为0
        self.eligibility = np.zeros(self.features)
        self.psi = np.zeros(self.features)

    def newEpisode(self):
        self.epsilon *= 0.9992
        if self.epsilon <= 0.0001:
            self.epsilon = 0.0001

    def eligibilityinit(self):
        self.eligibility = np.zeros(self.features)

    def get_q(self, observation, action):  # 动作价值
        return self.w[action * self.state_n + observation]

    def decide(self, observation):  # 判决  贪心策略
        if np.random.rand() < self.epsilon:
            # print('gaga')
            return np.random.randint(self.action_n), True  # 随机选择一个动作, 是否探索了，True为探索
        else:
            qs = [self.get_q(observation, action) for action in  # 获取该状态下对应所有动作的q值
                  range(self.action_n)]
            return np.argmax(qs), False  # 返回可以获得最大q值的动作

    def argmaxQ(self, observation):
        qs = [self.get_q(observation, action) for action in  # 获取该状态下对应所有动作的q值
              range(self.action_n)]
        return np.argmax(qs)  # 返回可以获得最大q值的动作

    def getRho(self, observation, action, is_epsilon, is_offpolicy):
        muGreedy = 1.0 - self.epsilon + self.epsilon / self.action_n
        na = self.argmaxQ(observation)
        if is_offpolicy:
            if is_epsilon:
                if na == action:
                    return 1.0 / muGreedy
                else:
                    return 0
            else:
                return 1.0 / muGreedy
        else:
            return 1.0

    def Qlearning(self, observation, action, reward, next_observation, terminated):  # 学习
        u = reward + (1. - terminated) * self.gamma * self.get_q(next_observation, self.argmaxQ(next_observation))
        td_error = u - self.get_q(observation, action)
        features = action * self.state_n + observation
        self.w[features] += self.alpha * td_error

    def VMQ(self, observation, action, reward, next_observation, terminated):  # 学习
        u = reward + (1. - terminated) * self.gamma * self.get_q(next_observation, self.argmaxQ(next_observation))
        td_error = u - self.get_q(observation, action)
        features = action * self.state_n + observation
        self.w[features] += self.alpha * (td_error - self.omga)
        self.omga += self.beta * (td_error - self.omga)

    def TDClearning(self, observation, action, reward, next_observation, terminated, isEpsilon, isOffPolicy=False):
        next_A_max = self.argmaxQ(next_observation)
        u = reward + (1. - terminated) * self.gamma * self.get_q(next_observation, next_A_max)
        td_error = u - self.get_q(observation, action)
        rho = self.getRho(observation, action, isEpsilon, isOffPolicy)
        features = action * self.state_n + observation
        next_features = next_A_max * self.state_n + next_observation
        phiPsi = self.psi[features].sum()
        self.w[features] += self.alpha * rho * td_error
        self.w[next_features] -= self.alpha * phiPsi * self.gamma * rho
        self.psi[features] += self.zeta * (rho * td_error - phiPsi)

    def TDClearning_new(self, observation, action, reward, next_observation, terminated, isEpsilon, isOffPolicy=False):
        next_A_max = self.argmaxQ(next_observation)
        u = reward + (1. - terminated) * self.gamma * self.get_q(next_observation, next_A_max)
        td_error = u - self.get_q(observation, action)
        rho = self.getRho(observation, action, isEpsilon, isOffPolicy)
        features = action * self.state_n + observation
        next_features = next_A_max * self.state_n + next_observation
        phiPsi = self.psi[features].sum()
        self.w[features] += self.alpha * rho * (td_error - self.omga)
        self.w[next_features] -= self.alpha * phiPsi * self.gamma * rho
        self.psi[features] += self.zeta * (rho * (td_error - self.omga) - phiPsi)
        self.omga += self.beta * (td_error - self.omga)

    def play_TDC(self, render=False, learningtype=0):
        observation, _ = self.env.reset()  # 初始状态
        # action, _ = self.decide(observation)
        step = 0
        episode_reward = 0
        # print(self.omga)
        while True:
            if render:
                self.env.render()
            action, isEpsilon = self.decide(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            step += 1
            episode_reward += reward
            if learningtype == 1:
                self.TDClearning(observation, action, reward, next_observation, terminated, isEpsilon)
            elif learningtype == 2:
                # print('gggg')
                self.TDClearning_new(observation, action, reward, next_observation, terminated, isEpsilon)
            elif learningtype == 3:
                self.Qlearning(observation, action, reward, next_observation, terminated)
            elif learningtype == 4:
                self.VMQ(observation, action, reward, next_observation, terminated)
            if terminated or truncated or step == 1000:
                return step
            observation = next_observation
        # print(step)