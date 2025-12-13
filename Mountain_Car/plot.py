import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv

font2 = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": 20,
}


def csv_to_numpy(path):
    return_list = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            return_list.append(row)
    return return_list


def sample_data(x_data, y_data, sample_rate):
    """根据采样率对数据进行采样"""
    if sample_rate <= 1:
        return x_data, y_data
    sampled_x = x_data[::sample_rate]
    sampled_y = y_data[::sample_rate]
    return sampled_x, sampled_y


def plot_data0(
        path, weight
):  # path:路径； weight:平滑权重
    # result = csv_to_numpy(path)
    # result = np.array(result).astype(float)
    result = np.load(path)
    # print(result.shape)
    # 修改这里，使 plot_steps 长度与实际数据长度一致
    data_length = len(result[0])
    plot_steps = np.arange(1, data_length + 1, 1)  # 读取训练步数
    mean = [0 for _ in range(len(result[0]))]
    std = [0 for _ in range(len(result[0]))]
    for i in range(len(result[0])):
        temp = []
        for j in range(len(result)):
            temp.append(float(result[j][i]))
        mean[i] = np.mean(temp)
        std[i] = np.std(temp)
    # mean = np.mean(result, axis=1)
    smoothed = smoothing_tensorboard2(mean, weight)
    # print(smoothed.shape())
    # smoothed = smooth(mean, weight)
    # std = np.std(result, axis=1)
    print(np.array(mean).shape)
    return plot_steps, smoothed, np.array(mean), np.array(std)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def smoothing_tensorboard2(x, smooth):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):  # 平滑处理循环
        x[i] = (x[i - 1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x


def writecsv(filename, list):
    fileloc = '' + filename
    with open(fileloc, mode='a+', newline='') as reward_file:
        reward_writer = csv.writer(reward_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        reward_writer.writerow(list)


if __name__ == "__main__":
    plt.figure(figsize=(10.0, 9.0))
    plt.rcParams.update({'font.size': 32, 'font.family': 'Nimbus Roman'})
    plt.ylabel("Averaged Steps")  # 设置轴字体及标签
    plt.xlabel("Training episodes")

    path1 = "../results/mountain_car/Q.npy"  # Q
    path2 = "../results/mountain_car/GQ.npy"  # GQ
    path3 = "../results/mountain_car/CQ.npy"  # CQ
    path4 = "../results/mountain_car/CGQ.npy"  # CGQ

    sample_rate = 50  # 控制采样疏密程度的核心参数

    step1, smoothed1, mean1, std1 = plot_data0(path1, 0.9)
    step2, smoothed2, mean2, std2 = plot_data0(path2, 0.9)
    step3, smoothed3, mean3, std3 = plot_data0(path3, 0.9)
    step4, smoothed4, mean4, std4 = plot_data0(path4, 0.9)

    # 对所有相关数据进行采样处理
    step1_sampled, smoothed1_sampled = sample_data(step1, smoothed1, sample_rate)
    _, mean1_sampled = sample_data(step1, mean1, sample_rate)
    _, std1_sampled = sample_data(step1, std1, sample_rate)

    step2_sampled, smoothed2_sampled = sample_data(step2, smoothed2, sample_rate)
    _, mean2_sampled = sample_data(step2, mean2, sample_rate)
    _, std2_sampled = sample_data(step2, std2, sample_rate)

    step3_sampled, smoothed3_sampled = sample_data(step3, smoothed3, sample_rate)
    _, mean3_sampled = sample_data(step3, mean3, sample_rate)
    _, std3_sampled = sample_data(step3, std3, sample_rate)

    step4_sampled, smoothed4_sampled = sample_data(step4, smoothed4, sample_rate)
    _, mean4_sampled = sample_data(step4, mean4, sample_rate)
    _, std4_sampled = sample_data(step4, std4, sample_rate)

    plt.plot(step1_sampled, smoothed1_sampled, label="Q-learning", color="blue", linestyle='-', marker='s',
             markersize=10, linewidth=2)
    plt.fill_between(step1_sampled, mean1_sampled + std1_sampled, mean1_sampled - std1_sampled, color="blue", alpha=0.1)

    plt.plot(step2_sampled, smoothed2_sampled, label="GQ", color="green", linestyle='-', marker='^', markersize=10,
             linewidth=2)
    plt.fill_between(step2_sampled, mean2_sampled + std2_sampled, mean2_sampled - std2_sampled, color="green",
                     alpha=0.1)

    plt.plot(step3_sampled, smoothed3_sampled, label="CQ", color="red", linestyle='-', marker='D', markersize=10,
             linewidth=2)
    plt.fill_between(step3_sampled, mean3_sampled + std3_sampled, mean3_sampled - std3_sampled, color="red", alpha=0.1)

    plt.plot(step4_sampled, smoothed4_sampled, label="CGQ", color="black", linestyle='-', marker='p', markersize=10,
             linewidth=2)
    plt.fill_between(step4_sampled, mean4_sampled + std4_sampled, mean4_sampled - std4_sampled, color="black",
                     alpha=0.1)

    plt.ylim(100, 300)
    plt.xlim(0, 1500)
    plt.legend()
    plt.savefig("../img/mountain_car.pdf")
    plt.show()