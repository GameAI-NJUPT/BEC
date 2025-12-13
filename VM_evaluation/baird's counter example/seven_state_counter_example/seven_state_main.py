from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
import os

# all states: state 0-5 are upper states
STATES = np.arange(0, 7)
# state 6 is lower state
LOWER_STATE = 6
# discount factor
DISCOUNT = 0.99

# each state is represented by a vector of length 8
FEATURE_SIZE = 8
# FEATURES = np.zeros((len(STATES), FEATURE_SIZE))
# for i in range(LOWER_STATE):
#     FEATURES[i, i] = 2
#     FEATURES[i, 7] = 1
# FEATURES[LOWER_STATE, 6] = 1
# FEATURES[LOWER_STATE, 7] = 2

FEATURES = np.zeros([7, 8])

for i in range(7):
    FEATURES[i][i] = 2
    FEATURES[i][7] = 1
FEATURES[6][6] = 1
FEATURES[6][7] = 2

# all possible actions
DASHED = 0
SOLID = 1
ACTIONS = [DASHED, SOLID]

# reward is always zero
REWARD = 0


# take @action at @state, return the new state
def takeAction(state, action):
    if action == SOLID:
        return LOWER_STATE
    return np.random.choice(STATES[: LOWER_STATE])


# behavior policy
# target policy
def targetPolicy(state):
    return SOLID


# state distribution for the behavior policy
stateDistribution = np.ones(len(STATES)) / 7
stateDistributionMat = np.matrix(np.diag(stateDistribution))
# projection matrix for minimize MSVE
projectionMatrix = np.matrix(FEATURES) * \
                   np.linalg.pinv(np.matrix(FEATURES.T) * stateDistributionMat * np.matrix(FEATURES)) * \
                   np.matrix(FEATURES.T) * \
                   stateDistributionMat

# behavior policy
BEHAVIOR_SOLID_PROBABILITY = 1.0 / 7


def behaviorPolicy(state):
    if np.random.binomial(1, BEHAVIOR_SOLID_PROBABILITY) == 1:
        return SOLID
    return DASHED

def semiGradientOffPolicyTD(state, theta, alpha):
    action = behaviorPolicy(state)
    nextState = takeAction(state, action)
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
            np.dot(FEATURES[state, :], theta)
    delta *= rho * alpha
    # derivatives happen to be the same matrix due to the linearity
    theta += FEATURES[state, :] * delta
    return nextState

def semiGradientOffPolicyVMTD(state, theta, omga, alpha, beta):
    action = behaviorPolicy(state)
    nextState = takeAction(state, action)
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
            np.dot(FEATURES[state, :], theta) - omga
    # delta *= rho * alpha
    # derivatives happen to be the same matrix due to the linearity
    theta += FEATURES[state, :] * (rho * alpha * delta)
    omga += beta * rho *delta
    return nextState

def TDC(state, theta, weight, alpha, zeta):
    action = behaviorPolicy(state)
    nextState = takeAction(state, action)
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
            np.dot(FEATURES[state, :], theta)
    theta += alpha * rho * (
            delta * FEATURES[state, :] - DISCOUNT * FEATURES[nextState, :] * np.dot(FEATURES[state, :], weight))
    weight += zeta * rho * (delta - np.dot(FEATURES[state, :], weight)) * FEATURES[state, :]
    return nextState

def VMTDC(state, theta, weight, omga, alpha, beta, zeta):
    action = behaviorPolicy(state)
    nextState = takeAction(state, action)
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
            np.dot(FEATURES[state, :], theta) - omga
    theta += alpha * rho * (
            delta * FEATURES[state, :] - DISCOUNT * FEATURES[nextState, :] * np.dot(FEATURES[state, :], weight))
    weight += zeta * rho * (delta - np.dot(FEATURES[state, :], weight)) * FEATURES[state, :]
    omga += beta * rho * delta
    return nextState



# compute VBE for a value function parameterized by @theta
# true value function is always 0 in this example


def computeRVBE(theta):
    bellmanError = np.zeros(len(STATES))
    for state in STATES:
        for nextState in STATES:
            if nextState == LOWER_STATE:
                bellmanError[state] += REWARD + DISCOUNT * np.dot(theta, FEATURES[nextState, :]) - np.dot(theta,
                                                                                                          FEATURES[
                                                                                                          state, :])
    expectedBE = np.mean(bellmanError)
    RVBE = np.sqrt(np.sum(((bellmanError - expectedBE) ** 2) * stateDistribution))
    return RVBE

def figure_TD(alpha):
    theta = np.ones(FEATURE_SIZE)
    theta[6] = 10

    # alpha = 0.1

    iteration = 50

    steps = 5000
    thetas = np.zeros((FEATURE_SIZE, steps+1))
    VBE = np.zeros(steps+1)
    VBEs = np.zeros((iteration, steps+1))
    for i in trange(iteration):
        theta = np.ones(FEATURE_SIZE)  # feature_size=8
        theta[6] = 10  # [1,1,1,1,1,1,10,1]
        state = np.random.choice(STATES)
        VBE[0] = computeRVBE(theta)
        for step in range(steps+1):
            state = semiGradientOffPolicyTD(state, theta, alpha)
            thetas[:, step] = theta
            VBE[step] = computeRVBE(theta)
        VBEs[i, :] = VBE
    return VBEs

def figure_VMTD(alpha, beta):
    # Initialize the theta
    theta = np.ones(FEATURE_SIZE)
    theta[6] = 10
    # alpha = alpha
    # zeta = zeta

    iteration = 50

    steps = 5000
    thetas = np.zeros((FEATURE_SIZE, steps + 1))

    RVBE = np.zeros(steps + 1)
    RVBEs = np.zeros((iteration, steps + 1))
    for i in trange(iteration):
        theta = np.ones(FEATURE_SIZE)  # feature_size=8
        theta[6] = 10  # [1,1,1,1,1,1,10,1]
        omga = np.zeros(1)
        RVBE[0] = computeRVBE(theta)
        state = np.random.choice(STATES)
        for sweep in range(1, steps + 1):
            state = semiGradientOffPolicyVMTD(state, theta, omga, alpha, beta)
            thetas[:, sweep] = theta
            RVBE[sweep] = computeRVBE(theta)
        RVBEs[i, :] = RVBE
    return RVBEs

def figure_TDC(alpha, zeta):
    # Initialize the theta
    theta = np.ones(FEATURE_SIZE)
    theta[6] = 10

    iteration = 50

    steps = 5000

    RVBE = np.zeros(steps + 1)
    RVBEs = np.zeros((iteration, steps + 1))
    for i in trange(iteration):
        theta = np.ones(FEATURE_SIZE)  # feature_size=8
        theta[6] = 10  # [1,1,1,1,1,1,10,1]
        weight = np.zeros(FEATURE_SIZE)
        RVBE[0] = computeRVBE(theta)
        state = np.random.choice(STATES)
        for sweep in range(1, steps + 1):
            state = TDC(state, theta, weight, alpha, zeta)
            RVBE[sweep] = computeRVBE(theta)
        RVBEs[i, :] = RVBE
    return RVBEs

def figure_VMTDC(alpha, beta, zeta):
    # Initialize the theta
    theta = np.ones(FEATURE_SIZE)
    theta[6] = 10

    iteration = 50

    steps = 5000

    RVBE = np.zeros(steps + 1)
    RVBEs = np.zeros((iteration, steps + 1))
    for i in trange(iteration):
        theta = np.ones(FEATURE_SIZE)  # feature_size=8
        theta[6] = 10  # [1,1,1,1,1,1,10,1]
        weight = np.zeros(FEATURE_SIZE)
        omga = np.zeros(1)
        RVBE[0] = computeRVBE(theta)
        state = np.random.choice(STATES)
        for sweep in range(1, steps + 1):
            state = VMTDC(state, theta, weight, omga, alpha, beta, zeta)
            RVBE[sweep] = computeRVBE(theta)
        RVBEs[i, :] = RVBE
    return RVBEs



if __name__ == '__main__':
    learningType = [0,1,2,3]
    alpha = 0.01
    beta = 0.01
    zeta = 0.05
    for i in learningType:
        if i == 0:
            print("TD")
            VBE_TD = figure_TD(alpha)
            np.save('../../../results/7state/TD.npy', VBE_TD)

        if i == 1:
            print("CTD")
            RVBE_VMTD = figure_VMTD(alpha, beta)
            np.save('../../../results/7state/CTD.npy',
                        RVBE_VMTD)

        if i == 2:
            print("TDC")
            RVBE_TDC = figure_TDC(alpha, zeta)
            np.save('../../../results/7state/TDC.npy',
                            RVBE_TDC)

        if i == 3:
            print("CTDC")
            RVBE_VMTDC = figure_VMTDC(alpha, beta, zeta)
            np.save('../../../results/7state/CTDC.npy', RVBE_VMTDC)