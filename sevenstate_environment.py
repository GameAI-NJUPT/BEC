#!/usr/bin/env python3
"""
Seven State Counter Example环境实现
"""

import numpy as np


class SevenStateEnvironment:
    """Seven State Counter Example环境"""
    
    def __init__(self):
        # 定义七状态反例环境参数
        self.state_space = 7    # 七个状态
        self.action_space = 2   # 两个动作
        self.current_state = 0
        self.states = ['1', '2', '3', '4', '5', '6', '7']
        self.feature_size = 8   # 特征大小
        
    def reset(self):
        """重置环境到初始状态"""
        self.current_state = 0  # 状态1
        return self.current_state
        
    def step(self, action):
        """执行动作并返回下一个状态、奖励和是否结束"""
        # 简化的状态转移
        if action == 0:
            # 动作0：向前转移
            self.current_state = min(self.state_space - 1, self.current_state + 1)
        else:
            # 动作1：向后转移或保持
            self.current_state = max(0, self.current_state - 1) if np.random.rand() < 0.7 else self.current_state
            
        next_state = self.current_state
        
        # 计算奖励
        if self.current_state == 6:  # 状态7
            reward = 100
        elif self.current_state == 0:  # 状态1
            reward = -10
        else:  # 其他状态
            reward = 0
            
        # 这个环境不会自然结束
        done = False
            
        return next_state, reward, done
    
    def select_action(self, state):
        """为给定状态选择动作（简单随机策略）"""
        return np.random.randint(self.action_space)
        
    def fea(self, state):
        """特征函数"""
        # 简化的特征表示
        feature = np.zeros(self.feature_size)
        if state < self.feature_size - 1:
            feature[state] = 2
        feature[self.feature_size - 1] = 1
        return feature


# Seven State环境的超参数配置
SEVENSTATE_ALGORITHM_PARAMS = {
    'TD': {'alpha': 0.01},
    'TDC': {'alpha': 0.01, 'beta': 0.01},
    'VMTD': {'alpha': 0.01, 'zeta': 0.01},
    'VMTDC': {'alpha': 0.01, 'beta': 0.01, 'zeta': 0.05}
}


def train_sevenstate_algorithms(algorithms_dict, episodes=1000):
    """训练Seven State环境下的算法: TD, TDC, VMTD, VMTDC"""
    print("开始训练Seven State环境下的算法...")
    
    # Seven State环境训练的算法
    sevenstate_algorithms = {
        'TD': algorithms_dict.get('TD'),
        'TDC': algorithms_dict.get('TDC'),
        'VMTD': algorithms_dict.get('VMTD'),
        'VMTDC': algorithms_dict.get('VMTDC')
    }
    
    # 移除None值
    sevenstate_algorithms = {k: v for k, v in sevenstate_algorithms.items() if v is not None}
    
    # 创建Seven State环境
    env = SevenStateEnvironment()
    
    # 训练结果存储
    results = {}
    
    # 训练每种算法
    for name, algorithm_class in sevenstate_algorithms.items():
        try:
            # 创建算法实例并传入环境特定的超参数
            params = SEVENSTATE_ALGORITHM_PARAMS.get(name, {})
            algorithm = algorithm_class(env, **params)
            
            # 训练算法
            steps_history = algorithm.train(episodes)
            
            # 保存结果
            results[name] = steps_history
            
            print(f"{name} 在Seven State环境中训练完成")
            
        except Exception as e:
            print(f"{name} 在Seven State环境中训练失败: {str(e)}")
            results[name] = None
    
    return results