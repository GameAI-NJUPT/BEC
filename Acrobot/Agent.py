from tilecoding_action import TileCoder_action
import numpy as np
from MeanErrors import MeanErrors


class Agent:
    def __init__(self, env, tiling_specs, layers=5, features=6000, gamma=1.
                 , epsilon=0.01, alpha=0.1, zeta=0.0001, beta=0.0004, coherenceLearning=False):
        self.env = env
        # self.env = self.env.unwrapped
        self.action_n = env.action_space.n  # 动作数
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.obs_scale = env.observation_space.high - \
                         env.observation_space.low  # 观测空间范围
        self.tiling_specs = tiling_specs
        self.features = features
        self.encoder = TileCoder_action(layers, features)  # 砖瓦编码器
        self.tilings = self.encoder.create_tilings(self.obs_low, self.obs_high, self.tiling_specs)
        self.w = np.zeros(features)  # 权重初始化为0
        self.etd_f = 1
        self.gamma = gamma  # 折扣
        self.epsilon = epsilon  # 探索
        self.eligibility = np.zeros(features)
        self.psi = np.zeros(features)
        self.error = np.zeros(features)
        self.absoluteError = np.zeros(features)
        self.alpha = alpha
        self.zeta = zeta
        # self.beta = 0.000001
        self.beta = beta
        self.omga = .0
        self.lamda = 0.8
        self.coherenceLearning = coherenceLearning
        self.me = MeanErrors(length=1000)
        self.regularization_lambda1 = 10.0

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

    def encode(self, observation, action):  # 编码
        actions = (action,)
        return self.encoder(self.tilings, observation, actions)

    def get_q(self, observation, action):  # 动作价值
        actions = (action,)
        # features = self.encoder(self.tilings, observation, actions)
        features = self.encode(observation, action)  # [0,1,2,3,4,5,6,7]
        return self.w[features].sum()

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

    def getCoherenceLearningRate(self, index, delta):
        if self.coherenceLearning:
            dynamicAlpha = 1 if self.absoluteError[index] == 0 else abs(self.error[index]) / self.absoluteError[index]
            self.error[index] += delta
            self.absoluteError[index] += abs(delta)
            return dynamicAlpha
        else:
            return self.alpha

    def getRho(self, next_observation, next_action, is_epsilon, is_offpolicy):
        muGreedy = 1.0 - self.epsilon + self.epsilon / self.action_n
        na = self.argmaxQ(next_observation)
        if is_offpolicy:
            if is_epsilon:
                if na == next_action:
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
        features = self.encode(observation, action)
        self.w[features] += self.alpha * td_error

    def VMQ(self, observation, action, reward, next_observation, terminated):
        u = reward + (1. - terminated) * self.gamma * self.get_q(next_observation, self.argmaxQ(next_observation))
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += self.alpha * (td_error - self.omga)
        self.omga += self.beta * (td_error - self.omga)

    def TDClearning(self, observation, action, reward, next_observation, terminated, isEpsilon, isOffPolicy=False):
        next_A_max = self.argmaxQ(next_observation)
        u = reward + (1. - terminated) * self.gamma * self.get_q(next_observation, next_A_max)
        td_error = u - self.get_q(observation, action)
        rho = self.getRho(observation, action, isEpsilon, isOffPolicy)
        features = self.encode(observation, action)
        next_features = self.encode(next_observation, next_A_max)
        phiPsi = self.psi[features].sum()
        self.w[features] += self.alpha * rho * td_error
        self.w[next_features] -= self.alpha * phiPsi * self.gamma * rho
        self.psi[features] += self.zeta * (rho * td_error - phiPsi)

    def TDClearning_new(self, observation, action, reward, next_observation, terminated, isEpsilon, isOffPolicy=False):
        next_A_max = self.argmaxQ(next_observation)
        u = reward + (1. - terminated) * self.gamma * self.get_q(next_observation, next_A_max)
        td_error = u - self.get_q(observation, action)
        rho = self.getRho(observation, action, isEpsilon, isOffPolicy)
        features = self.encode(observation, action)
        next_features = self.encode(next_observation, next_A_max)
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
                self.TDClearning_new(observation, action, reward, next_observation, terminated, isEpsilon)
            elif learningtype == 3:
                self.Qlearning(observation, action, reward, next_observation, terminated)
            elif learningtype == 4:
                self.VMQ(observation, action, reward, next_observation, terminated)
            if terminated or truncated or step == 1000:
                return step
                break
            observation = next_observation
        # print(step)



    def test(self, render=False):
        steps = []
        for episode in range(300):  # 没训练50局的时候，测试200局，返回这200局对应的平局步伐
            observation, _ = self.env.reset()  # 初始状态
            action = self.argmaxQ(observation)
            step = 0
            while True:
                if render:
                    self.env.render()
                step += 1
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.argmaxQ(next_observation)  # 终止状态时此步无意义
                # print(terminated)
                # print(truncated)
                if terminated or truncated:
                    break
                observation, action = next_observation, next_action
            # print(step)
            steps.append(step)
        return np.mean(steps)  # 返回平均步数
