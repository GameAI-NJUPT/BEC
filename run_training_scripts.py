#!/usr/bin/env python3
"""
多进程执行训练脚本
"""

import multiprocessing
import subprocess
import sys
import os
import json
from pathlib import Path

def run_script(script_path):
    """
    运行指定的Python脚本
    
    Args:
        script_path (str): 脚本的路径
    
    Returns:
        tuple: (script_name, success, error_message)
    """
    script_name = os.path.basename(script_path)
    try:
        print(f"开始运行: {script_name}")
        # 使用当前Python解释器运行脚本
        result = subprocess.run(
            [sys.executable, script_path], 
            cwd=os.path.dirname(script_path),  # 在脚本所在目录运行
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"完成运行: {script_name}")
        return (script_name, True, "")
    except subprocess.CalledProcessError as e:
        error_msg = f"错误码: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        print(f"运行失败 {script_name}: {error_msg}")
        return (script_name, False, error_msg)
    except Exception as e:
        error_msg = str(e)
        print(f"运行 {script_name} 时发生异常: {error_msg}")
        return (script_name, False, error_msg)

def load_config():
    """加载训练配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_configs.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置运行所有脚本")
        return {
            "Maze": 1,
            "Acrobot": 1,
            "CliffWalking": 1,
            "MountainCar": 1,
            "TwoState": 1,
            "SevenState": 1
        }
    except json.JSONDecodeError as e:
        print(f"配置文件 {config_path} 格式错误: {e}")
        return {}

def get_scripts_to_run(config):
    """根据配置获取需要运行的脚本列表"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 所有可能的脚本
    all_scripts = {
        "Maze": os.path.join(base_dir, "Maze", "main.py"),
        "Acrobot": os.path.join(base_dir, "Acrobot", "main.py"),
        "CliffWalking": os.path.join(base_dir, "Cliff_Walking", "main.py"),
        "MountainCar": os.path.join(base_dir, "Mountain_Car", "main.py"),
        "TwoState": os.path.join(base_dir, "VM_evaluation", "baird's counter example", "two_state_counter_example", "two_state_main.py"),
        "SevenState": os.path.join(base_dir, "VM_evaluation", "baird's counter example", "seven_state_counter_example", "seven_state_main.py"),
    }
    
    # 根据配置筛选需要运行的脚本
    scripts_to_run = []
    for name, enabled in config.items():
        if enabled and name in all_scripts:
            script_path = all_scripts[name]
            if os.path.exists(script_path):
                scripts_to_run.append(script_path)
            else:
                print(f"警告: 找不到 {name} 的脚本文件 {script_path}")
    
    return scripts_to_run

def main():
    # 加载配置
    config = load_config()
    
    # 获取需要运行的脚本
    scripts = get_scripts_to_run(config)
    
    if not scripts:
        print("没有需要运行的脚本")
        return
    
    print(f"准备并行运行 {len(scripts)} 个脚本...")
    
    # 创建进程池，使用CPU核心数作为进程数上限
    max_processes = min(len(scripts), multiprocessing.cpu_count())
    print(f"使用 {max_processes} 个进程并行执行")
    
    # 使用进程池并行执行脚本
    with multiprocessing.Pool(processes=max_processes) as pool:
        results = pool.map(run_script, scripts)
    
    # 输出结果统计
    print("\n" + "="*60)
    print("执行完成!")
    print("="*60)
    
    success_count = 0
    failure_count = 0
    
    for script_name, success, error_msg in results:
        if success:
            success_count += 1
            print(f"✓ {script_name}: 成功")
        else:
            failure_count += 1
            print(f"✗ {script_name}: 失败")
            print(f"  错误信息: {error_msg}")
    
    print("-"*60)
    print(f"总计: {len(results)} 个脚本")
    print(f"成功: {success_count} 个")
    print(f"失败: {failure_count} 个")

if __name__ == "__main__":
    main()