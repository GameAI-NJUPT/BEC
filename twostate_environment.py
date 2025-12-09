#!/usr/bin/env python3
"""
Two State Counter Example环境实现
"""

import numpy as np


class TwoStateEnvironment:
    """Two State Counter Example环境"""
    
    def __init__(self):
        # 定义二状态反例环境参数
        self.state_space = 2    # 只有两个状态
        self.action_space = 2   # 两个动作
        self.current_state = 0
        self.states = ['A', 'B']
        self.feature_size = 8   # 特征大小
        
    def reset(self):
        """重置环境到初始状态"""
        self.current_state = 0  # 状态A
        return self.current_state
        
    def step(self, action):
        """执行动作并返回下一个状态、奖励和是否结束"""
        # Baird反例环境的状态转移（遵循μ策略）
        # μ策略: π(a=1|s) = 0.5, π(a=0|s) = 0.5
        if np.random.rand() < 0.5:
            self.current_state = 0  # 动作0
        else:
            self.current_state = 1  # 动作1
            
        next_state = self.current_state
        
        # 奖励始终为0
        reward = 0
            
        # 这个环境不会结束
        done = False
            
        return next_state, reward, done
    
    def select_action(self, state):
        """为给定状态选择动作（遵循μ策略）
        μ策略: π(a=1|s) = 0.5, π(a=0|s) = 0.5
        """
        # 遵循μ策略，随机选择动作
        return 0 if np.random.rand() < 0.5 else 1
        
    def fea(self, state):
        """特征函数"""
        # 简化的特征表示
        feature = np.zeros(self.feature_size)
        if state < self.feature_size - 1:
            feature[state] = 2
        feature[self.feature_size - 1] = 1
        return feature


# Two State环境的超参数配置
TWOSTATE_ALGORITHM_PARAMS = {
    'TD': {'alpha': 0.1},
    'TDC': {'alpha': 0.1, 'beta': 0.01},
    'VMTD': {'alpha': 0.1, 'zeta': 0.1},
    'VMTDC': {'alpha': 0.1, 'beta': 0.0001, 'zeta': 0.1}
}


def train_twostate_algorithms(algorithms_dict, episodes=1000):
    """训练Two State环境下的算法: TD, TDC, VMTD, VMTDC"""
    print("开始训练Two State环境下的算法...")
    
    # Two State环境训练的算法
    twostate_algorithms = {
        'TD': algorithms_dict.get('TD'),
        'TDC': algorithms_dict.get('TDC'),
        'VMTD': algorithms_dict.get('VMTD'),
        'VMTDC': algorithms_dict.get('VMTDC')
    }
    
    # 移除None值
    twostate_algorithms = {k: v for k, v in twostate_algorithms.items() if v is not None}
    
    # 创建Two State环境
    env = TwoStateEnvironment()
    
    # 训练结果存储
    results = {}
    
    # 训练每种算法
    for name, algorithm_class in twostate_algorithms.items():
        try:
            # 创建算法实例并传入环境特定的超参数
            params = TWOSTATE_ALGORITHM_PARAMS.get(name, {})
            algorithm = algorithm_class(env, **params)
            
            # 训练算法
            steps_history = algorithm.train(episodes)
            
            # 保存结果
            results[name] = steps_history
            
            print(f"{name} 在Two State环境中训练完成")
            
        except Exception as e:
            print(f"{name} 在Two State环境中训练失败: {str(e)}")
            results[name] = None
    
    return results