#!/usr/bin/env python3
"""
Maze环境实现
"""

import numpy as np


class MazeEnvironment:
    """Maze环境"""
    
    def __init__(self):
        # 定义迷宫环境参数
        self.grid_size = (10, 10)
        self.state_space = 100  # 10x10网格
        self.action_space = 4   # 上下左右四个动作
        self.start_position = (0, 0)
        self.goal_position = (9, 9)
        self.current_position = self.start_position
        self.actions = ['up', 'down', 'left', 'right']
        
    def reset(self):
        """重置环境到初始状态"""
        self.current_position = self.start_position
        return self._position_to_state(self.current_position)
        
    def step(self, action_index):
        """执行动作并返回下一个状态、奖励和是否结束"""
        action = self.actions[action_index]
        x, y = self.current_position
        
        # 根据动作更新位置
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.grid_size[1] - 1, y + 1)
            
        self.current_position = (x, y)
        next_state = self._position_to_state(self.current_position)
        
        # 计算奖励和是否结束
        if self.current_position == self.goal_position:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
            
        return next_state, reward, done
    
    def select_action(self, state):
        """为给定状态选择动作（简单随机策略）"""
        return np.random.randint(self.action_space)
        
    def _position_to_state(self, position):
        """将二维位置转换为一维状态索引"""
        return position[0] * self.grid_size[1] + position[1]


# Maze环境的超参数配置
MAZE_ALGORITHM_PARAMS = {
    'TDC': {'alpha': 0.1, 'beta': 0.01},
    'ImprovedTDC': {'alpha': 0.1, 'beta': 0.01, 'zeta': 0.001},
    'QLearning': {'alpha': 0.1, 'epsilon': 0.01},
    'VMQ': {'alpha': 0.1, 'zeta': 0.001, 'epsilon': 0.01}
}


def train_maze_algorithms(algorithms_dict, episodes=1000):
    """训练Maze环境下的算法: TDC, ImprovedTDC, QLearning, VMQ"""
    print("开始训练Maze环境下的算法...")
    
    # Maze环境只训练指定的四种算法
    maze_algorithms = {
        'TDC': algorithms_dict.get('TDC'),
        'ImprovedTDC': algorithms_dict.get('ImprovedTDC'),
        'QLearning': algorithms_dict.get('QLearning'),
        'VMQ': algorithms_dict.get('VMQ')
    }
    
    # 移除None值
    maze_algorithms = {k: v for k, v in maze_algorithms.items() if v is not None}
    
    # 创建Maze环境
    env = MazeEnvironment()
    
    # 训练结果存储
    results = {}
    
    # 训练每种算法
    for name, algorithm_class in maze_algorithms.items():
        try:
            # 创建算法实例并传入环境特定的超参数
            params = MAZE_ALGORITHM_PARAMS.get(name, {})
            algorithm = algorithm_class(env, **params)
            
            # 训练算法
            steps_history = algorithm.train(episodes)
            
            # 保存结果
            results[name] = steps_history
            
            print(f"{name} 在Maze环境中训练完成")
            
        except Exception as e:
            print(f"{name} 在Maze环境中训练失败: {str(e)}")
            results[name] = None
    
    return results