import numpy as np


class TileCoder_action:
    def __init__(self, layers, features):
        self.layers = layers  # 3层
        self.features = features  # 140625个特征
        self.codebook = {}

    def init(self):
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:  # 冲突处理
            return hash(codeword) % self.features
        self.codebook[codeword] = count
        return count

    # 创建瓦片化网格
    def create_tiling_grid(self, low, high, bins=(5, 5), offsets=(0.0, 0.0)):
        return [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]

    # 瓦片化
    def create_tilings(self, low, high, tiling_specs):
        return [self.create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs]

    # 根据给定的网格离散样本。
    def discretize(self, sample, grid):
        return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # 返回索引值

    # 使用瓦片编码对给定的样本进行编码
    def tile_encode(self, sample, tilings):
        encoded_sample = [self.discretize(sample, grid) for grid in tilings]  # 返回在相应瓦片上的坐标
        return encoded_sample
        # return np.concatenate(encoded_sample) if flatten else encoded_sample

    def __call__(self, tilings, sample, actions=()):  # 传入了状态和动作
        sample_encode = self.tile_encode(sample, tilings)
        features = []
        for layer, tile in enumerate(sample_encode):
            codeword = (layer,) + tile + actions
            feature = self.get_feature(codeword)
            features.append(feature)
        return features  # 输出的是一个列表
