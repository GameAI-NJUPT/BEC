import numpy as np


class MeanErrors:
    def __init__(self, length=1000):
        self.delta = np.zeros(length)
        self.rewards = np.zeros(length)
        self.next_rewards = np.zeros(length)
        self.next_delta = np.zeros(length)
        self.init = False
        self.updated = False
        self.isFull = False
        self.index = 0
        self.averagedDelta = .0
        self.maximalAveragedReward = np.NINF
        self.minimalAbsoluteMeanDelta = np.inf
        # self.zeta = 1.006
        # self.zeta = 1.009
        # self.zeta = 1.0085
        # self.zeta = 1.0055  # 表现不错
        # self.zeta = 1.005  # 不行
        # self.zeta = 1.0035
        # self.zeta = 1.0058
        # self.zeta = 1.0064 # 极
        # self.zeta = 1.020
        # self.zeta = 1.021
        # self.zeta = 1.022
        # self.zeta = 1.023
        # self.zeta = 1.024
        # self.zeta = 1.025
        # self.zeta = 1.0253
        # self.zeta = 1.026
        # self.zeta = 1.027
        # self.zeta = 1.028
        # self.zeta = 1.029
        # self.zeta = 1.1
        # self.zeta = 1.017
        # self.zeta = 1.024
        # self.zeta = 1.022
        # self.zeta = 1.016
        self.zeta = 1.024


    def add(self, delta, reward):
        if self.index >= len(self.delta):
            self.isFull = True
            self.index = self.index % len(self.delta)
        self.delta[self.index] = delta
        self.rewards[self.index] = reward
        self.index += 1

    def newEpisode(self, episode):
        self.updated = False
        # if self.gy <= 1.006:
        #     self.gy *= 1.000002
        # print('gag')
        if self.index > 0 or self.isFull == True:
            count = self.index
            if self.isFull:
                count = len(self.delta)
            absolute = .0
            sum = .0
            rewardsum = .0
            for i in range(count):
                absolute += abs(self.delta[i])
                sum += self.delta[i]
                rewardsum += self.rewards[i]
            absolute = absolute / count
            sum = sum / count
            # rewardsum = rewardsum / count
            rewardsum = rewardsum
            # print(rewardsum, self.maximalAveragedReward)
            # if absolute < self.minimalAbsoluteMeanDelta:
            #     self.maximalAveragedReward = rewardsum
            #     self.minimalAbsoluteMeanDelta = absolute
            #     self.averagedDelta = sum
            #     self.init = True
            #     self.updated = True
            #     # print('new maximal absolute reward:{} \t new averaged error:{}'.format(self.minimalAbsoluteMeanDelta,
            #     #                                                               self.averagedDelta))
            # print(rewardsum,self.maximalAveragedReward)
            if rewardsum > self.maximalAveragedReward:
                self.maximalAveragedReward = rewardsum
                self.minimalAbsoluteMeanDelta = absolute
                self.averagedDelta = sum
                self.init = True
                self.updated = True

            # if rewardsum > self.maximalAveragedReward or absolute < self.minimalAbsoluteMeanDelta:
            #     if rewardsum > self.maximalAveragedReward:
            #         self.maximalAveragedReward = rewardsum
            #     if absolute < self.minimalAbsoluteMeanDelta:
            #         self.minimalAbsoluteMeanDelta = absolute
            #     self.averagedDelta = sum
            #     self.init = True
            #     self.updated = True

            # if rewardsum > self.maximalAveragedReward and absolute < self.minimalAbsoluteMeanDelta:
            #     self.maximalAveragedReward = rewardsum
            #     self.minimalAbsoluteMeanDelta = absolute
            #     self.averagedDelta = sum
            #     self.init = True
            #     self.updated = True

            # print('new maximal reward:{} \t new averaged error:{}'.format(self.maximalAveragedReward, self.averagedDelta))
        self.index = 0

    def getNewDelta(self, delta):
        deltaNew = delta - self.averagedDelta * self.zeta  # 这样写好不灵活
        return deltaNew
