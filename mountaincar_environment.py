#!/usr/bin/env python3
"""
Mountain Car环境实现
"""

import numpy as np


class MountainCarEnvironment:
    """Mountain Car环境"""
    
    def __init__(self):
        # 定义山地车环境参数
        self.state_space = 100  # 简化的状态空间
        self.action_space = 3   # 三个动作：左力、无力、右力
        self.current_state = 0
        self.goal_position = 99
        
    def reset(self):
        """重置环境到初始状态"""
        self.current_state = 0  # 起始位置
        return self.current_state
        
    def step(self, action):
        """执行动作并返回下一个状态、奖励和是否结束"""
        # 简化的状态转移
        if action == 0:  # 左力
            self.current_state = max(0, self.current_state - 1)
        elif action == 2:  # 右力
            self.current_state = min(self.state_space - 1, self.current_state + 1)
        # action == 1 表示无作用力，状态不变
            
        next_state = self.current_state
        
        # 计算奖励和是否结束
        if self.current_state >= self.goal_position:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
            
        return next_state, reward, done
    
    def select_action(self, state):
        """为给定状态选择动作（简单随机策略）"""
        return np.random.randint(self.action_space)


# Mountain Car环境的超参数配置
MOUNTAINCAR_ALGORITHM_PARAMS = {
    'TDC': {'alpha': 0.01, 'beta': 0.01},
    'ImprovedTDC': {'alpha': 0.01, 'beta': 0.01, 'zeta': 0.0001},
    'QLearning': {'alpha': 0.1, 'epsilon': 0.05},
    'VMQ': {'alpha': 0.1, 'zeta': 0.0001, 'epsilon': 0.05}
}


def train_mountaincar_algorithms(algorithms_dict, episodes=1000):
    """训练Mountain Car环境下的算法: TDC, ImprovedTDC, QLearning, VMQ"""
    print("开始训练Mountain Car环境下的算法...")
    
    # Mountain Car环境只训练指定的四种算法
    mountaincar_algorithms = {
        'TDC': algorithms_dict.get('TDC'),
        'ImprovedTDC': algorithms_dict.get('ImprovedTDC'),
        'QLearning': algorithms_dict.get('QLearning'),
        'VMQ': algorithms_dict.get('VMQ')
    }
    
    # 移除None值
    mountaincar_algorithms = {k: v for k, v in mountaincar_algorithms.items() if v is not None}
    
    # 创建Mountain Car环境
    env = MountainCarEnvironment()
    
    # 训练结果存储
    results = {}
    
    # 训练每种算法
    for name, algorithm_class in mountaincar_algorithms.items():
        try:
            # 创建算法实例并传入环境特定的超参数
            params = MOUNTAINCAR_ALGORITHM_PARAMS.get(name, {})
            algorithm = algorithm_class(env, **params)
            
            # 训练算法
            steps_history = algorithm.train(episodes)
            
            # 保存结果
            results[name] = steps_history
            
            print(f"{name} 在Mountain Car环境中训练完成")
            
        except Exception as e:
            print(f"{name} 在Mountain Car环境中训练失败: {str(e)}")
            results[name] = None
    
    return results