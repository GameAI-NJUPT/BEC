#!/usr/bin/env python3
"""
完全独立的绘图脚本执行器
包含所有需要的绘图功能，不依赖外部文件
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def sample_data(x_data, y_data, sample_rate):
    """根据采样率对数据进行采样"""
    if sample_rate <= 1:
        return x_data, y_data
    sampled_x = x_data[::sample_rate]
    sampled_y = y_data[::sample_rate]
    return sampled_x, sampled_y


def smoothing_tensorboard2(x, smooth):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):  # 平滑处理循环
        x[i] = (x[i - 1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x


def limit_std_values(std_values, max_std=10):
    """
    限制标准差的最大值以防止异常值影响可视化效果
    """
    limited_std = np.copy(std_values)
    limited_std[limited_std > max_std] = max_std
    return limited_std


def load_training_data(environment_name):
    """加载训练数据"""
    try:
        # 构造数据目录路径
        # 处理特定环境名称映射
        name_mapping = {
            "Two State Counter Example": "two_state_counter_example",
            "Seven State Counter Example": "seven_state_counter_example",
            "Maze": "maze",
            "Acrobot": "acrobot",
            "Cliff Walking": "cliff_walking",
            "Mountain Car": "mountain_car"
        }
        
        if environment_name in name_mapping:
            data_dir = f"training_data/{name_mapping[environment_name]}"
        else:
            data_dir = f"training_data/{environment_name.lower().replace(' ', '_')}"
        
        print(f"尝试加载数据目录: {data_dir}")
        # 检查目录是否存在
        if not os.path.exists(data_dir):
            print(f"警告: 找不到 {environment_name} 的训练数据目录")
            return None
            
        # 查找.npy文件
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('_results.npy')]
        
        if not npy_files:
            print(f"警告: {environment_name} 目录中没有找到训练数据文件")
            return None
            
        # 加载数据
        data = {}
        for file in npy_files:
            algorithm_name = file.replace('_results.npy', '')
            file_path = os.path.join(data_dir, file)
            try:
                data[algorithm_name] = np.load(file_path)
                print(f"已加载 {algorithm_name} 的训练数据，形状: {data[algorithm_name].shape}")
            except Exception as e:
                print(f"加载 {file_path} 失败: {str(e)}")
            
        return data
    except Exception as e:
        print(f"加载 {environment_name} 训练数据失败: {str(e)}")
        return None


def plot_environment_results(environment_name, algorithms_to_plot=None):
    """绘制指定环境的训练结果"""
    print(f"正在生成 {environment_name} 图...")
    
    # 加载训练数据
    data = load_training_data(environment_name)
    if data is None:
        print(f"跳过 {environment_name} 绘图")
        return
    
    # 默认绘制所有算法，除非指定了特定算法
    if algorithms_to_plot is None:
        algorithms_to_plot = list(data.keys())
    
    # 创建图表
    plt.figure(figsize=(10, 9))
    plt.rcParams.update({'font.size': 32, 'font.family': 'Nimbus Roman'})
    plt.ylabel("Steps")
    plt.xlabel("Episodes")
    
    # 算法名称映射和样式定义（参考原始绘图文件）
    # 通用算法名称映射
    general_algorithm_name_mapping = {
        'QLearning': 'Q-learning',
        'VMQ': 'CQ',
        'TDC': 'GQ',
        'ImprovedTDC': 'CGQ',
        'TD': 'Off-policy TD',
        'VMTD': 'Off-policy CTD',
        'VMTDC': 'CTDC'
    }
    
    # 2-state和7-state环境的算法名称映射
    counter_example_algorithm_name_mapping = {
        'QLearning': 'Q-learning',
        'VMQ': 'CQ',
        'TDC': 'TDC',  # 在2-state和7-state环境中TDC仍叫TDC
        'ImprovedTDC': 'CGQ',
        'TD': 'Off-policy TD',
        'VMTD': 'Off-policy CTD',
        'VMTDC': 'CTDC'
    }
    
    # 根据环境选择算法名称映射
    if environment_name in ["Two State Counter Example", "Seven State Counter Example"]:
        algorithm_name_mapping = counter_example_algorithm_name_mapping
    else:
        algorithm_name_mapping = general_algorithm_name_mapping
    
    # 颜色和标记设置（参考原始绘图文件）
    algorithm_styles = {
        'QLearning': {'color': 'blue', 'marker': 's'},
        'VMQ': {'color': 'red', 'marker': 'D'},
        'TDC': {'color': 'green' if environment_name not in ["Two State Counter Example", "Seven State Counter Example"] else 'green', 
                'marker': '^' if environment_name not in ["Two State Counter Example", "Seven State Counter Example"] else '^'},
        'ImprovedTDC': {'color': 'black', 'marker': 'p'},
        'TD': {'color': 'blue', 'marker': 's'},
        'VMTD': {'color': 'red', 'marker': 'D'},
        'VMTDC': {'color': 'black', 'marker': 'p'}
    }
    
    # 绘制每种算法的结果
    plotted_any = False
    lines = []  # 用于图例
    labels = []  # 用于图例标签
    
    for i, algorithm in enumerate(algorithms_to_plot):
        if algorithm not in data:
            print(f"警告: 找不到 {algorithm} 的训练数据")
            continue
            
        # 获取数据
        steps_data = data[algorithm]
        
        # 简化处理：如果数据维度大于1，取第一列
        if len(steps_data.shape) > 1:
            steps_data = steps_data[:, 0] if steps_data.shape[1] > 0 else steps_data.flatten()
        
        episodes = np.arange(len(steps_data))
        
        # 平滑处理
        if len(steps_data) > 0:
            # 根据环境设置不同的平滑参数
            smooth_param = 0.9
            if environment_name in ["Two State Counter Example", "Seven State Counter Example"]:
                smooth_param = 0.1
            elif environment_name in ["Acrobot"]:
                smooth_param = 0.94
            elif environment_name in ["Mountain Car"]:
                smooth_param = 0.89
            
            smoothed_data = smoothing_tensorboard2(steps_data, smooth_param)
            
            # 采样
            sample_rate = max(1, len(episodes) // 100)  # 采样到最多100个点
            episodes_sampled, data_sampled = sample_data(episodes, smoothed_data, sample_rate)
            
            # 获取算法样式
            styles = algorithm_styles.get(algorithm, {
                'color': ['blue', 'green', 'red', 'black', 'orange', 'purple', 'brown', 'pink'][i % 8],
                'marker': ['s', '^', 'D', 'p', '*', 'h', 'o', 'v'][i % 8]
            })
            
            # 获取显示名称
            display_name = algorithm_name_mapping.get(algorithm, algorithm)
            
            # 绘图
            line, = plt.plot(episodes_sampled, data_sampled, 
                           label=display_name,
                           color=styles['color'], 
                           linestyle='-', 
                           marker=styles['marker'], 
                           markersize=10, 
                           linewidth=2)
            lines.append(line)
            labels.append(display_name)
            plotted_any = True
    
    if plotted_any:
        # 根据环境类型设置坐标轴范围
        if environment_name == "Cliff Walking":
            plt.xlim(0, 500)
            plt.ylim(0, 120)
        elif environment_name == "Maze":
            plt.ylim(20, 200)
            plt.xlim(0, 1500)
        elif environment_name == "Acrobot":
            plt.ylim(80, 500)
        elif environment_name == "Mountain Car":
            plt.ylim(50, 700)
        elif environment_name in ["Two State Counter Example", "Seven State Counter Example"]:
            plt.ylabel("RMSCBE")
            
        # 显示图例
        plt.legend()
        
        plt.title(environment_name)
        filename = f"{environment_name.lower().replace(' ', '_')}_plot.png"
        
        # 保存主图（含图例）
        plt.savefig(filename)
        plt.close()
        
        print(f"已保存 {environment_name} 图到 {filename}")
    else:
        plt.close()
        print(f"{environment_name} 没有可绘制的数据")


def plot_all_environments():
    """绘制所有环境的结果"""
    environments = [
        "Maze",
        "Acrobot", 
        "Cliff Walking",
        "Mountain Car",
        "Two State Counter Example",
        "Seven State Counter Example"
    ]
    
    # 创建保存图片的目录
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    # 生成所有图表
    saved_files = []
    for env in environments:
        # 不切换目录，直接在当前位置生成图表，然后移动到plots目录
        plot_environment_results(env)
        filename = f"{env.lower().replace(' ', '_')}_plot.png"
        if os.path.exists(filename):
            # 尝试将文件移动到plots目录
            try:
                os.rename(filename, f"plots/{filename}")
                saved_files.append((filename, env))
            except Exception as e:
                print(f"无法将 {filename} 移动到 plots 目录: {e}")
    
    print("\n所有图表已生成并保存到 plots 目录中:")
    for filename, env in saved_files:
        print(f"- {filename} ({env})")


def main():
    """主函数：执行所有绘图任务"""
    print("开始生成所有图表...")
    plot_all_environments()


if __name__ == "__main__":
    main()