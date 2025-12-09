#!/usr/bin/env python3
"""
通用强化学习算法实现
包含实际使用的算法实现
根据learningtype = [9,10,4,5]以及two-state和seven-state环境中的算法
"""

import numpy as np
import time


class BaseAlgorithm:
    """基础算法类"""
    
    def __init__(self, env, **kwargs):
        self.env = env
        self.name = "BaseAlgorithm"
        
    def train(self, episodes):
        """训练方法 - 子类需要重写"""
        raise NotImplementedError("子类必须实现train方法")


class TDC(BaseAlgorithm):
    """TDC算法 (对应learningtype=4)"""
    
    def __init__(self, env, alpha=0.1, beta=0.01, gamma=0.99, **kwargs):
        super().__init__(env)
        self.name = "TDC"
        self.w = np.zeros(env.state_space)  # 主权重
        self.v = np.zeros(env.state_space)  # 辅助权重
        self.alpha = alpha
        self.beta = beta  # 原来的zeta参数
        self.gamma = gamma
        
    def train(self, episodes=1000):
        print(f"开始 {self.name} 训练...")
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 10000  # 设置最大步数限制
            
            while not done and steps < max_steps:
                action = self.env.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # TDC更新
                delta = reward + self.gamma * self.w[next_state] - self.w[state]
                self.w[state] += self.alpha * (delta - self.gamma * self.v[state])
                self.v[state] += self.beta * (delta - self.v[state])
                
                state = next_state
                steps += 1
                
            steps_history.append(steps)
            
            if episode % 200 == 0:
                print(f"{self.name} 训练进度: {episode}/{episodes} 步数: {steps}")
                
        return np.array(steps_history)


class ImprovedTDC(BaseAlgorithm):
    """Improved TDC算法 (对应learningtype=5)"""
    
    def __init__(self, env, alpha=0.1, beta=0.01, gamma=0.99, zeta=0.001, **kwargs):
        super().__init__(env)
        self.name = "ImprovedTDC"
        self.w = np.zeros(env.state_space)
        self.v = np.zeros(env.state_n) if hasattr(env, 'state_n') else np.zeros(env.state_space)
        self.alpha = alpha
        self.beta = beta  # 原来的zeta参数
        self.gamma = gamma
        self.omga = 0.0
        self.zeta = zeta  # 原来的beta参数
        
    def train(self, episodes=1000):
        print(f"开始 {self.name} 训练...")
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 10000  # 设置最大步数限制
            
            while not done and steps < max_steps:
                action = self.env.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Improved TDC更新 (类似CTDC)
                next_action = self.env.select_action(next_state)
                u = reward + (1. - done) * self.gamma * self.w[next_state]
                td_error = u - self.w[state]
                
                self.w[state] += self.alpha * (td_error - self.omga)
                self.omga += self.zeta * (td_error - self.omga)
                
                state = next_state
                steps += 1
                
            steps_history.append(steps)
            
            if episode % 200 == 0:
                print(f"{self.name} 训练进度: {episode}/{episodes} 步数: {steps}")
                
        return np.array(steps_history)


class QLearning(BaseAlgorithm):
    """Q-Learning算法 (对应learningtype=9)"""
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.01, **kwargs):
        super().__init__(env)
        self.name = "QLearning"
        self.Q = np.zeros((env.state_space, env.action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def train(self, episodes=1000):
        print(f"开始 {self.name} 训练...")
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 10000  # 设置最大步数限制
            
            while not done and steps < max_steps:
                # epsilon-greedy策略
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.env.action_space)
                else:
                    action = np.argmax(self.Q[state])
                    
                next_state, reward, done = self.env.step(action)
                
                # Q-Learning更新
                self.Q[state, action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
                )
                
                state = next_state
                steps += 1
                
            steps_history.append(steps)
            
            if episode % 200 == 0:
                print(f"{self.name} 训练进度: {episode}/{episodes} 步数: {steps}")
                
        return np.array(steps_history)


class VMQ(BaseAlgorithm):
    """VMQ算法 (对应learningtype=10)"""
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.01, zeta=0.001, **kwargs):
        super().__init__(env)
        self.name = "VMQ"
        self.Q = np.zeros((env.state_space, env.action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.omga = 0.0
        self.zeta = zeta  # 原来的beta参数
        
    def train(self, episodes=1000):
        print(f"开始 {self.name} 训练...")
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 10000  # 设置最大步数限制
            
            while not done and steps < max_steps:
                # epsilon-greedy策略
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.env.action_space)
                else:
                    action = np.argmax(self.Q[state])
                    
                next_state, reward, done = self.env.step(action)
                
                # VMQ更新
                u = reward + (1. - done) * self.gamma * np.max(self.Q[next_state])
                td_error = u - self.Q[state, action]
                
                self.Q[state, action] += self.alpha * (td_error - self.omga)
                self.omga += self.zeta * (td_error - self.omga)
                
                state = next_state
                steps += 1
                
            steps_history.append(steps)
            
            if episode % 200 == 0:
                print(f"{self.name} 训练进度: {episode}/{episodes} 步数: {steps}")
                
        return np.array(steps_history)


class TD(BaseAlgorithm):
    """TD算法 (two-state和seven-state环境使用)"""
    
    def __init__(self, env, alpha=0.1, gamma=0.99, **kwargs):
        super().__init__(env)
        self.name = "TD"
        self.theta = np.ones(env.feature_size) if hasattr(env, 'feature_size') else np.ones(8)
        self.theta[-2] = 10  # 特殊设置
        self.alpha = alpha
        self.gamma = gamma
        
    def train(self, episodes=1000):
        print(f"开始 {self.name} 训练...")
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 10000  # 设置最大步数限制
            
            while not done and steps < max_steps:
                action = self.env.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # TD更新
                if hasattr(self.env, 'fea'):
                    # 对于特征表示的环境
                    delta = reward + self.gamma * np.dot(self.env.fea(next_state), self.theta) - \
                            np.dot(self.env.fea(state), self.theta)
                    self.theta += self.alpha * delta * self.env.fea(state)
                else:
                    # 对于普通环境
                    delta = reward + self.gamma * self.theta[next_state] - self.theta[state]
                    self.theta[state] += self.alpha * delta
                
                state = next_state
                steps += 1
                
            steps_history.append(steps)
            
            if episode % 200 == 0:
                print(f"{self.name} 训练进度: {episode}/{episodes} 步数: {steps}")
                
        return np.array(steps_history)


class VMTD(BaseAlgorithm):
    """VMTD算法 (two-state和seven-state环境使用)"""
    
    def __init__(self, env, alpha=0.1, zeta=0.01, gamma=0.99, **kwargs):
        super().__init__(env)
        self.name = "VMTD"
        self.theta = np.ones(env.feature_size) if hasattr(env, 'feature_size') else np.ones(8)
        self.theta[-2] = 10  # 特殊设置
        self.omga = np.zeros(1)
        self.alpha = alpha
        self.zeta = zeta  # 原来的beta参数
        self.gamma = gamma
        
    def train(self, episodes=1000):
        print(f"开始 {self.name} 训练...")
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 10000  # 设置最大步数限制
            
            while not done and steps < max_steps:
                action = self.env.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # VMTD更新
                if hasattr(self.env, 'fea'):
                    # 对于特征表示的环境
                    delta = reward + self.gamma * np.dot(self.env.fea(next_state), self.theta) - \
                            np.dot(self.env.fea(state), self.theta)
                    self.theta += self.alpha * (delta - self.omga[0]) * self.env.fea(state)
                    self.omga[0] += self.zeta * (delta - self.omga[0])
                else:
                    # 对于普通环境
                    delta = reward + self.gamma * self.theta[next_state] - self.theta[state]
                    self.theta[state] += self.alpha * (delta - self.omga[0])
                    self.omga[0] += self.zeta * (delta - self.omga[0])
                
                state = next_state
                steps += 1
                
            steps_history.append(steps)
            
            if episode % 200 == 0:
                print(f"{self.name} 训练进度: {episode}/{episodes} 步数: {steps}")
                
        return np.array(steps_history)


class VMTDC(BaseAlgorithm):
    """VMTDC算法 (two-state和seven-state环境使用)"""
    
    def __init__(self, env, alpha=0.1, zeta=0.01, gamma=0.99, beta=0.05, **kwargs):
        super().__init__(env)
        self.name = "VMTDC"
        self.theta = np.ones(env.feature_size) if hasattr(env, 'feature_size') else np.ones(8)
        self.theta[-2] = 10  # 特殊设置
        self.weight = np.zeros(env.feature_size) if hasattr(env, 'feature_size') else np.zeros(8)
        self.omga = np.zeros(1)
        self.alpha = alpha
        self.zeta = zeta  # 原来的beta参数
        self.gamma = gamma
        self.beta = beta  # 原来的zeta参数
        
    def train(self, episodes=1000):
        print(f"开始 {self.name} 训练...")
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 10000  # 设置最大步数限制
            
            while not done and steps < max_steps:
                action = self.env.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # VMTDC更新
                if hasattr(self.env, 'fea'):
                    # 对于特征表示的环境
                    delta = reward + self.gamma * np.dot(self.env.fea(next_state), self.theta) - \
                            np.dot(self.env.fea(state), self.theta)
                    delta -= self.omga[0]
                    
                    self.theta += self.alpha * (
                        delta * self.env.fea(state) - self.gamma * self.env.fea(next_state) * 
                        np.dot(self.env.fea(state), self.weight)
                    )
                    self.weight += self.zeta * (
                        delta - np.dot(self.env.fea(state), self.weight)
                    ) * self.env.fea(state)
                    self.omga[0] += self.beta * delta
                else:
                    # 对于普通环境
                    delta = reward + self.gamma * self.theta[next_state] - self.theta[state]
                    delta -= self.omga[0]
                    
                    self.theta[state] += self.alpha * delta
                    self.omga[0] += self.beta * delta
                
                state = next_state
                steps += 1
                
            steps_history.append(steps)
            
            if episode % 200 == 0:
                print(f"{self.name} 训练进度: {episode}/{episodes} 步数: {steps}")
                
        return np.array(steps_history)


# 可用算法列表 (实际使用的算法)
ALGORITHMS = {
    'TDC': TDC,                 # learningtype=4
    'ImprovedTDC': ImprovedTDC, # learningtype=5
    'QLearning': QLearning,     # learningtype=9
    'VMQ': VMQ,                 # learningtype=10
    'TD': TD,                   # two-state和seven-state环境使用
    'VMTD': VMTD,               # two-state和seven-state环境使用
    'VMTDC': VMTDC              # two-state和seven-state环境使用
}