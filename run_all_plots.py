#!/usr/bin/env python3
"""
脚本用于同时运行所有六个绘图脚本
"""

import subprocess
import sys
import os

def run_script(script_path, working_dir=None):
    """
    运行指定的Python脚本
    
    Args:
        script_path (str): 脚本的路径
        working_dir (str): 工作目录，默认为None
    
    Returns:
        bool: 执行成功返回True，否则返回False
    """
    try:
        print(f"正在运行: {script_path}")
        if working_dir:
            # 如果指定了工作目录，则切换到该目录执行脚本
            result = subprocess.run([sys.executable, script_path], cwd=working_dir, check=True)
        else:
            # 否则直接运行脚本
            result = subprocess.run([sys.executable, script_path], check=True)
        print(f"完成运行: {script_path}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_path} 失败: {e}")
        return False
    except FileNotFoundError:
        print(f"找不到脚本文件: {script_path}")
        return False

def main():
    # 获取当前工作目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义六个绘图脚本的路径和工作目录
    scripts = [
        {
            "path": os.path.join(base_dir, "Cliff_Walking", "plot.py"),
            "work_dir": os.path.join(base_dir, "Cliff_Walking")
        },
        {
            "path": os.path.join(base_dir, "Mountain_Car", "plot.py"),
            "work_dir": os.path.join(base_dir, "Mountain_Car")
        },
        {
            "path": os.path.join(base_dir, "Acrobot", "plot.py"),
            "work_dir": os.path.join(base_dir, "Acrobot")
        },
        {
            "path": os.path.join(base_dir, "Maze", "plot.py"),
            "work_dir": os.path.join(base_dir, "Maze")
        },
        {
            "path": os.path.join(base_dir, "VM_evaluation", "baird's counter example", "seven_state_counter_example", "plot.py"),
            "work_dir": os.path.join(base_dir, "VM_evaluation", "baird's counter example", "seven_state_counter_example")
        },
        {
            "path": os.path.join(base_dir, "VM_evaluation", "baird's counter example", "two_state_counter_example", "plot.py"),
            "work_dir": os.path.join(base_dir, "VM_evaluation", "baird's counter example", "two_state_counter_example")
        }
    ]
    
    print("开始运行所有绘图脚本...\n")
    
    # 记录成功和失败的数量
    success_count = 0
    failure_count = 0
    
    # 依次运行每个脚本
    for i, script_info in enumerate(scripts, 1):
        print(f"[{i}/{len(scripts)}] 正在处理...")
        if run_script(script_info["path"], script_info["work_dir"]):
            success_count += 1
        else:
            failure_count += 1
    
    # 输出总结信息
    print("\n" + "="*50)
    print("执行完成!")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")
    print(f"总计: {len(scripts)}")
    print("="*50)

if __name__ == "__main__":
    main()