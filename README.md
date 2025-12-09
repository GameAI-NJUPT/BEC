# Bellman Error Centering

## 目录结构

- `algorithms.py`: 包含实际使用的强化学习算法的实现
- `maze_environment.py`: Maze环境及训练函数
- `acrobot_environment.py`: Acrobot环境及训练函数
- `cliffwalking_environment.py`: Cliff Walking环境及训练函数
- `mountaincar_environment.py`: Mountain Car环境及训练函数
- `twostate_environment.py`: Two State Counter Example环境及训练函数
- `sevenstate_environment.py`: Seven State Counter Example环境及训练函数
- `run_training.py`: 主训练脚本执行器
- `run_plots.py`: 绘图脚本执行器

## 环境配置

### Python环境要求
- Python 3.6 或更高版本
- 依赖库:
  - numpy
  - matplotlib

### 安装依赖
```bash
pip install numpy matplotlib
```

或者使用conda:
```bash
conda install numpy matplotlib
```

## 功能说明

### 算法实现 (algorithms.py)

实现了以下实际使用的强化学习算法:
1. TDC (对应曲线GQ和TDC)
2. ImprovedTDC (对应曲线CGQ)
3. QLearning (对应曲线Q)
4. VMQ (对应曲线CQ)
5. TD (对应曲线TD)
6. VMTD (对应曲线CTD)
7. VMTDC (对应曲线CTDC)

### 环境实现

每个环境文件包含:
1. 环境类定义
2. 训练函数，用于训练该环境下的所有算法

### 训练脚本 (run_training.py)

主训练脚本，负责:
1. 调用各环境的训练函数
2. 管理训练主循环
3. 保存训练结果

### 绘图脚本 (run_plots.py)

绘图脚本，负责:
1. 加载训练数据
2. 生成各环境的训练结果图表
3. 保存图表到plots目录

## 使用方法

1. 运行训练:
   ```
   python run_training.py
   ```

2. 运行绘图:
   ```
   python run_plots.py
   ```

## 数据流

1. `run_training.py` 调用各环境的训练函数
2. 各环境文件使用 `algorithms.py` 中的算法进行训练
3. 训练结果保存在 `training_data` 目录中
4. `run_plots.py` 读取训练数据并生成图表
5. 图表保存在 `plots` 目录中