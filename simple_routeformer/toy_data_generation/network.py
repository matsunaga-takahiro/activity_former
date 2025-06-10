import torch
import pandas as pd
import numpy as np
from utils.logger import logger


class Network:
    def __init__(self, adj_matrix, node_features):
        logger.info('Network init')
        self.adj_matrix = adj_matrix
        self.node_features = node_features
        # self.df_penalty = df_penalty

        self.N = len(self.adj_matrix)
        self.node_id_list = [i for i in range(self.N)]

        #効用関数に入れる変数の行列を作成する
        # 説明変数4つの行列を挿入する
        self.feature_mat_0 = self._node_feature_matrix(feature_num = 0) 
        self.feature_mat_1 = self._node_feature_matrix(feature_num = 1)

    def _node_feature_matrix(self, feature_num):
        part_feature_mat = np.zeros((self.N, self.N))
        feature_vec = self.node_features[:, feature_num]
        for j in range(self.N): # 移動先がアーケードあるか
            part_feature_mat[j, :] = feature_vec 

        feature_mat = np.tile(part_feature_mat, (1, 1)).reshape((self.N, self.N))
        return feature_mat