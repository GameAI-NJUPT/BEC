# BEC

这个项目实现了多种强化学习算法，并在不同的 Gym 环境中进行了测试和比较。


## 环境配置

### Python版本要求

推荐使用 Python 3.8 或更高版本。

### 依赖库安装



```bash
pip install -r requirements.txt
```



## 运行方式

### 单独运行某个环境的训练

进入特定环境目录并运行主程序：

```bash
cd Acrobot
python main.py

cd ../Cliff_Walking
python main.py

cd ../Maze
python main.py

cd ../Mountain_Car
python main.py
```

### 并行运行所有环境的训练

在项目根目录下运行：

```bash
python run_training_scripts.py
```

可以通过修改 [training_configs.json](file:///home/coco/workspaces/code_gongyu_rebuild/training_configs.json) 来控制哪些环境参与训练：

```json
{
  "Maze": 1,         // 1表示启用，0表示禁用
  "Acrobot": 1,
  "CliffWalking": 1,
  "MountainCar": 1,
  "TwoState": 1,
  "SevenState": 1
}
```

### 绘制所有环境的结果图表

训练完成后，在项目根目录下运行：

```bash
python run_all_plots.py
```

这将为所有环境生成相应的性能对比图表。图标保存在img目录下。

## 算法实现

项目实现了以下几种强化学习算法：

1. **GQ|TDC**
2. **CGQ|CTDC**
3. **Q|TD**
4. **CQ|CTD**

这些算法在不同环境中进行测试，并比较它们的收敛速度和稳定性。

## 实验环境

- Acrobot-v1: 经典的双摆杆控制问题
- CliffWalking-v0: 悬崖行走问题
- MountainCar-v0: 上山车问题
- 自定义迷宫环境: 简单的路径规划问题
- Baird反例: 包含两状态和七状态的反例环境，用于测试算法的收敛性

## 结果查看

训练结果保存在 [results](file:///home/coco/workspaces/code_gongyu_rebuild/results) 目录下，每个子目录对应一个环境：
- [results/acrobot/](file:///home/coco/workspaces/code_gongyu_rebuild/results/acrobot/)
- [results/cliff_walking/](file:///home/coco/workspaces/code_gongyu_rebuild/results/cliff_walking/)
- [results/maze/](file:///home/coco/workspaces/code_gongyu_rebuild/results/maze/)
- [results/mountain_car/](file:///home/coco/workspaces/code_gongyu_rebuild/results/mountain_car/)

每个目录下包含不同算法的.npy格式结果文件，可用于后续分析和绘图。