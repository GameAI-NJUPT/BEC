#!/usr/bin/env python3
"""
完全独立的训练脚本执行器
主训练文件，只负责训练调用和主循环等
"""

import multiprocessing
import sys
import os
import time
import numpy as np

# 导入各个环境的训练函数
from maze_environment import train_maze_algorithms
from acrobot_environment import train_acrobot_algorithms
from cliffwalking_environment import train_cliffwalking_algorithms
from mountaincar_environment import train_mountaincar_algorithms
from twostate_environment import train_twostate_algorithms
from sevenstate_environment import train_sevenstate_algorithms

# 导入算法
from algorithms import ALGORITHMS


def save_training_results(environment_name, results):
    """保存训练结果到文件"""
    try:
        # 创建训练数据目录
        data_dir = f"training_data/{environment_name.lower().replace(' ', '_')}"
        os.makedirs(data_dir, exist_ok=True)
        
        # 保存每种算法的结果
        for algorithm_name, steps_history in results.items():
            if steps_history is not None:
                filename = f"{data_dir}/{algorithm_name.lower().replace(' ', '_')}_results.npy"
                np.save(filename, steps_history)
                print(f"已保存 {algorithm_name} 的训练结果到 {filename}")
                
        return True
    except Exception as e:
        print(f"保存 {environment_name} 训练结果失败: {str(e)}")
        return False


def main():
    """主函数：执行所有训练任务"""
    print("开始执行所有训练任务...")
    print(f"可用算法: {list(ALGORITHMS.keys())}")
    
    # 创建训练数据目录
    if not os.path.exists("training_data"):
        os.makedirs("training_data")
    
    # 定义所有训练任务
    training_tasks = [
        ("Maze", train_maze_algorithms),
        ("Acrobot", train_acrobot_algorithms),
        ("Cliff Walking", train_cliffwalking_algorithms),
        ("Mountain Car", train_mountaincar_algorithms),
        ("Two State Counter Example", train_twostate_algorithms),
        ("Seven State Counter Example", train_sevenstate_algorithms)
    ]
    
    # 顺序执行所有训练任务
    all_results = []
    for task_name, task_func in training_tasks:
        print(f"\n{'='*60}")
        print(f"开始执行 {task_name} 训练任务")
        print(f"{'='*60}")
        
        # 训练该环境下的所有算法
        results = task_func(ALGORITHMS, episodes=1000)
        all_results.append((task_name, results))
        
        # 保存训练结果
        save_training_results(task_name, results)
        
        print(f"\n{task_name} 训练任务完成!")
    
    # 输出结果统计
    print(f"\n{'='*60}")
    print("所有训练任务执行完成!")
    print(f"{'='*60}")
    
    total_algorithms = 0
    successful_algorithms = 0
    
    for environment_name, results in all_results:
        print(f"\n{environment_name} 环境训练结果:")
        for algorithm_name, steps_history in results.items():
            total_algorithms += 1
            if steps_history is not None:
                successful_algorithms += 1
                avg_steps = np.mean(steps_history[-100:]) if len(steps_history) > 0 else 0
                print(f"  ✓ {algorithm_name}: 成功 (平均最后100轮步数: {avg_steps:.2f})")
            else:
                print(f"  ✗ {algorithm_name}: 失败")
    
    print("-"*60)
    print(f"总计: {total_algorithms} 个算法")
    print(f"成功: {successful_algorithms} 个")
    print(f"失败: {total_algorithms - successful_algorithms} 个")
    print(f"\n训练数据已保存到 training_data 目录中")


if __name__ == "__main__":
    main()