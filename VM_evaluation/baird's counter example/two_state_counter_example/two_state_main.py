import random
import numpy as np
from tqdm import trange
import os


class suttonExample():
    def __init__(self):
        self.state_space = np.array([[1], [2]], dtype=float)
        self.reward = 0
        self.state = 0
        self.states = range(2)
        self.states_num = 2
        self.state_posibility = 1.0 / 2
        self.gammar = 0.9
        self.alpha= 0.1
        self.beta = 0.0001
        self.zeta = 0.001

    def mu(self):
        return 0 if random.random() < 0.5 else 1

    # def rho(self, action):
    #     return 2 * action
    #
    # def x(self):
    #     return 0.5

    def step(self, action):
        self.state = action
        return self.state, self.reward

    def step_pi(self, action_pi):
        return action_pi, self.reward

    def pi(self):
        return 1

    def mu(self):
        return 0 if random.random() < 0.5 else 1

    def fea(self, s):
        return self.state_space[s]

    def reset(self):
        return 0

    def semiGradientOffPolicyTD(self, state, theta, alpha):
        action = self.mu()
        nextState, reward = self.step(action)
        # get the importance ratio
        if action == 1:
            rho = 1 / 0.5  # pi(a=1|s) = 1, mu(a=1|s) = 0.5
        else:
            rho = 0 / 0.5  # pi(a=0|s) = 0, mu(a=0|s) = 0.5
        delta = reward + self.gammar * np.dot(self.fea(nextState), theta) - \
                np.dot(self.fea(state), theta)
        delta *= rho * alpha
        theta += self.fea(state) * delta
        return nextState

    def semiGradientOffPolicyVMTD(self, state, theta, omga, alpha,beta):
        action = self.mu()
        nextState, reward = self.step(action)
        # get the importance ratio
        if action == 1:
            rho = 1 / 0.5  # pi(a=1|s) = 1, mu(a=1|s) = 0.5
        else:
            rho = 0 / 0.5  # pi(a=0|s) = 0, mu(a=0|s) = 0.5
        delta = reward + self.gammar * np.dot(self.fea(nextState), theta) - \
                np.dot(self.fea(state), theta) - omga
        theta += self.fea(state) * (rho * alpha * delta)
        omga += beta * rho * delta
        return nextState

    def TDC(self, state, theta, weight, alpha, zeta):
        action = self.mu()
        nextState, reward = self.step(action)
        # get the importance ratio
        if action == 1:
            rho = 1 / 0.5  # pi(a=1|s) = 1, mu(a=1|s) = 0.5
        else:
            rho = 0 / 0.5  # pi(a=0|s) = 0, mu(a=0|s) = 0.5
        delta = reward + self.gammar * np.dot(self.fea(nextState), theta) - \
                np.dot(self.fea(state), theta)
        theta += alpha * rho * (
                delta * self.fea(state) - self.gammar * self.fea(nextState) * np.dot(self.fea(state), weight))
        weight += zeta * rho * (delta - np.dot(self.fea(state), weight)) * self.fea(state)
        return nextState

    def VMTDC(self, state, theta, weight, omga, alpha, beta, zeta):
        action = self.mu()
        nextState, reward = self.step(action)
        # get the importance ratio
        if action == 1:
            rho = 1 / 0.5  # pi(a=1|s) = 1, mu(a=1|s) = 0.5
        else:
            rho = 0 / 0.5  # pi(a=0|s) = 0, mu(a=0|s) = 0.5
        delta = reward + self.gammar * np.dot(self.fea(nextState), theta) - \
                np.dot(self.fea(state), theta) - omga
        theta += alpha * rho * (
                delta * self.fea(state) - self.gammar * self.fea(nextState) * np.dot(self.fea(state), weight))
        weight += zeta * rho * (delta - np.dot(self.fea(state), weight)) * self.fea(state)
        omga += beta * rho * delta
        return nextState

    def computeRVBE(self, theta):
        stateDistribution = np.ones(self.states_num) / self.states_num
        bellmanError = np.zeros(self.states_num)
        for state in self.states:
            for nextState in self.states:
                if nextState == 1:
                    bellmanError[state] += 0 + self.gammar * np.dot(theta, self.fea(nextState)) - np.dot(theta,
                                                                                                         self.fea(
                                                                                                             state))
        expectedBE = np.mean(bellmanError)
        RVBE = np.sqrt(np.sum(((bellmanError - expectedBE) ** 2) * stateDistribution))
        return RVBE

    def figure_TD(self, alpha):

        iteration = 50

        steps = 5000
        VBE = np.zeros(steps + 1)
        VBEs = np.zeros((iteration, steps + 1))
        for i in trange(iteration):
            theta = np.ones(1)  # feature_size=8
            state = np.random.choice(2)
            VBE[0] = self.computeRVBE(theta)
            for step in range(steps + 1):
                state = self.semiGradientOffPolicyTD(state, theta, alpha)
                VBE[step] = self.computeRVBE(theta)
            VBEs[i, :] = VBE
        return VBEs

    def figure_VMTD(self, alpha, beta):
        # Initialize the theta

        iteration = 50

        steps = 5000

        RVBE = np.zeros(steps + 1)
        RVBEs = np.zeros((iteration, steps + 1))
        for i in trange(iteration):
            theta = np.ones(1)  # feature_size=8
            omga = np.zeros(1)
            RVBE[0] = self.computeRVBE(theta)
            state = np.random.choice(2)
            for sweep in range(1, steps + 1):
                state = self.semiGradientOffPolicyVMTD(state, theta, omga, alpha, beta)
                RVBE[sweep] = self.computeRVBE(theta)
            # print(omga)
            RVBEs[i, :] = RVBE
        return RVBEs

    def figure_TDC(self, alpha, zeta):
        # Initialize the theta

        iteration = 50

        steps = 5000

        RVBE = np.zeros(steps + 1)
        RVBEs = np.zeros((iteration, steps + 1))
        for i in trange(iteration):
            theta = np.ones(1)  # feature_size=8
            weight = np.zeros(1)
            RVBE[0] = self.computeRVBE(theta)
            state = np.random.choice(2)
            for sweep in range(1, steps + 1):
                state = self.TDC(state, theta, weight, alpha, zeta)
                RVBE[sweep] = self.computeRVBE(theta)
            RVBEs[i, :] = RVBE
        return RVBEs

    def figure_VMTDC(self, alpha, beta, zeta):

        iteration = 50

        steps = 5000

        RVBE = np.zeros(steps + 1)
        RVBEs = np.zeros((iteration, steps + 1))
        for i in trange(iteration):
            theta = np.ones(1)  # feature_size=8
            weight = np.zeros(1)
            omga = np.zeros(1)
            RVBE[0] = self.computeRVBE(theta)
            state = np.random.choice(2)
            for sweep in range(1, steps + 1):
                state = self.VMTDC(state, theta, weight, omga, alpha, beta, zeta)
                RVBE[sweep] = self.computeRVBE(theta)
            RVBEs[i, :] = RVBE
        return RVBEs

    def save_TD(self):
        print("TD")
        VBE_TD = self.figure_TD(self.alpha)
        # np.save('TD_RVBE_sweep/TD_VBE_sweep_alpha_0.1', VBE_TD)
        np.save('../../../results/2state/TD.npy', VBE_TD)

    def save_VMTD(self):
        print("VMTD")
        RVBE_VMTD = self.figure_VMTD(self.alpha, self.beta)
        np.save('../../../results/2state/CTD.npy',
                RVBE_VMTD)

    def save_TDC(self):
        print("TDC")
        RVBE_TDC = self.figure_TDC(self.alpha, self.zeta)
        np.save('../../../results/2state/TDC.npy',
                RVBE_TDC)

    def save_VMTDC(self):
        print("VMTDC")
        RVBE_VMTDC = self.figure_VMTDC(self.alpha, self.beta, self.zeta)
        np.save('../../../results/2state/CTDC.npy',RVBE_VMTDC)


twoState = suttonExample()
twoState.save_VMTD()
twoState.save_TD()
twoState.save_TDC()
twoState.save_VMTDC()
