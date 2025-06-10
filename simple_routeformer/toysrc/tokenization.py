import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
# from data_generation.network import Network
#ルートはタイムステップで作るので，ルートデータはdata_num * Tの配列
#1タイムステップで滞在するか別のリンクに移動することしかできないと仮定する←ここの仮定は実データでは変更するかも

class Tokenization:
    def __init__(self, network):
        self.network = network
        self.num_nodes = network.N
        self.SPECIAL_TOKENS = { # valueはトークンID（nodeid）
            "<p>": self.num_nodes,  # パディングトークン
            "<e>": self.num_nodes + 1,  # 終了トークン
            "<b>": self.num_nodes + 2,  # 開始トークン
            "<m>": self.num_nodes + 3,  # 非隣接ノードトークン
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenization(self, route,  mode, lmax = None):
        self.route = route
        #もとのinputデータにもpaddingは入っている
        num_data = len(self.route)
        adjacency_matrix = self.network.adj_matrix.to(self.device)
        # 特殊トークンの定義
        tokens = self.route.clone().to(self.device) 

        # モードに応じたトークン化処理
        if mode == "simple":
            ##前と後ろにパディングトークンをくっつける
            new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<p>"], device=self.device) 
            tokens = torch.cat((new_column, tokens, new_column), dim=1)
            #前の塊の中で最初に出てくるパディングトークンを開始トークンにおきかえる
            mask = tokens == self.SPECIAL_TOKENS["<p>"]

            # 各行の左端から最初に現れる非パディングトークンの位置を取得
            first_non_padding_indices = (~mask).float().argmax(dim=1)
            # 最初の <p> の塊の最後のインデックスを計算
            last_padding_in_head_indices = first_non_padding_indices - 1
            # 対応するトークンを開始トークンに置き換え
            tokens[torch.arange(tokens.size(0)), last_padding_in_head_indices] = self.SPECIAL_TOKENS["<b>"]

        elif mode == "complete":
            new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<p>"], device=self.device) 
            tokens = torch.cat((new_column, tokens, new_column), dim=1)

            #前の塊の中で最初に出てくるパディングトークンを開始トークンにおきかえる
            mask = tokens == self.SPECIAL_TOKENS["<p>"]
            # 各行の左端から最初に現れる非パディングトークンの位置を取得
            first_non_padding_indices = (~mask).float().argmax(dim=1)
            # 最初の <p> の塊の最後のインデックスを計算
            last_padding_in_head_indices = first_non_padding_indices - 1
            # 対応するトークンを開始トークンに置き換え
            tokens[torch.arange(tokens.size(0)), last_padding_in_head_indices] = self.SPECIAL_TOKENS["<b>"]

            seq_len = tokens.size(1)
            last_non_padding_indices = seq_len - 1 - (~mask).float().flip(dims=[1]).argmax(dim=1)
            first_padding_in_tail_indices = last_non_padding_indices + 1
            tokens[torch.arange(tokens.size(0)), first_padding_in_tail_indices] = self.SPECIAL_TOKENS["<e>"]
            tokens = tokens[:, 1:]
            tokens = torch.cat((tokens, new_column), dim=1)

        elif mode == "next":
            ##前と後ろにパディングトークンをくっつける
            new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<p>"], device=self.device) 
            tokens = torch.cat((new_column, tokens, new_column), dim=1)
            #後ろの塊の中で最初に出てくるパディングトークンを終了トークンにおきかえる
            mask = tokens == self.SPECIAL_TOKENS["<p>"]
            seq_len = tokens.size(1)
            last_non_padding_indices = seq_len - 1 - (~mask).float().flip(dims=[1]).argmax(dim=1)
            first_padding_in_tail_indices = last_non_padding_indices + 1
            tokens[torch.arange(tokens.size(0)), first_padding_in_tail_indices] = self.SPECIAL_TOKENS["<e>"]
            tokens = tokens[:, 1:]
            tokens = torch.cat((tokens, new_column), dim=1)

        elif mode == "discontinuous":
            ##前と後ろにパディングトークンをくっつける
            new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<p>"], device=self.device) 
            tokens = torch.cat((new_column, tokens, new_column), dim=1)

            # 各行の左端から最初に現れる非パディングトークンの位置を取得
            mask = tokens == self.SPECIAL_TOKENS["<p>"]
            first_non_padding_indices = (~mask).float().argmax(dim=1)
            # 最初の <p> の塊の最後のインデックスを計算
            last_padding_in_head_indices = first_non_padding_indices - 1
            # 対応するトークンを開始トークンに置き換え
            batch_indices = torch.arange(tokens.size(0), device=tokens.device)
            tokens[batch_indices, last_padding_in_head_indices] = self.SPECIAL_TOKENS["<b>"]

            #パディングではない最後のトークンを取得
            seq_len = tokens.size(1)  # トークン列の長さ
            last_non_padding_indices = seq_len - 1 - (~mask).float().flip(dims=[1]).argmax(dim=1)
            non_padding_values = tokens[batch_indices, last_non_padding_indices]

            #開始トークンの2つ先を<m>に置き換える
            tokens[batch_indices, last_padding_in_head_indices + 2] = self.SPECIAL_TOKENS["<m>"]
            diff = last_non_padding_indices - last_padding_in_head_indices
            adjust_mask = diff == 2
            last_padding_in_head_indices[adjust_mask] -= 1
            #開始トークンの3つ先を最後のトークンに置き換える
            tokens[batch_indices, last_padding_in_head_indices + 3] = non_padding_values
            #開始トークンの4つ先を最後のトークンに置き換える
            tokens[batch_indices, last_padding_in_head_indices + 4] = self.SPECIAL_TOKENS["<e>"]
            #5つ先以降はパディングトークンに置き換える
            batch_size, seq_len = tokens.size()
            replace_mask = torch.arange(seq_len, device=tokens.device).unsqueeze(0) >= (last_padding_in_head_indices + 5).unsqueeze(1)
            tokens[replace_mask] = self.SPECIAL_TOKENS["<p>"]

        elif mode == "traveled":
            new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<b>"], device=self.device) 
            tokens = torch.cat((new_column, tokens), dim=1)

        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        # リストをPyTorchテンソルに変換
        return tokens.clone().detach().to(torch.long)
    
    def mask(self, mask_rate):
        mask_token_id = self.SPECIAL_TOKENS["<m>"]
        token_sequences = self.route.clone().to(self.device)
        batch_size, seq_len = token_sequences.shape

        # マスクを適用する位置を計算（1列目と最後の列を除外）
        mask_tokens = torch.rand(batch_size, seq_len) < mask_rate
        mask_tokens[:, 0] = False  # 1列目はマスクしない
        mask_tokens[:, -1] = False  # 最後の列はマスクしない

        # マスクトークンを適用
        token_sequences[mask_tokens] = mask_token_id
        num_data = len(self.route)
        new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<b>"], device=self.device) 
        new_column2 = torch.full((num_data, 1), self.SPECIAL_TOKENS["<e>"], device=self.device) 
        token_sequences = torch.cat((token_sequences, new_column2), dim=1)
        token_sequences = torch.cat((new_column, token_sequences), dim=1)
        return token_sequences
    
    def make_feature_mat(self, token_sequences):
        token_sequences = token_sequences.long().to(self.device)
        batch_size, seq_len = token_sequences.shape 
        node_features = self.network.node_features.to(self.device)
        feature_dim =  node_features.shape[1] # 4nohazud
        special_token_features = torch.zeros((4, feature_dim), device=self.device)
        total_node_features = torch.cat((node_features, special_token_features), dim=0)
        feature_mat = torch.zeros((batch_size, seq_len, feature_dim), device=self.device)
        feature_mat = total_node_features[token_sequences]
        return feature_mat
    
'''
    def make_VAE_input(self, token_sequences, time_index, img_dic):
        # 時間ごとの特徴量を格納
        feature_list = [img_dic[idx.item()].to(self.device) for idx in time_index]
        combined_feature_mat = torch.stack(feature_list, dim=0)
        combined_feature_mat = torch.nn.functional.pad(combined_feature_mat, (0, 0, 1, 1, 0, 1, 0, 0), mode='constant', value=0)

        #シーケンスの形状
        batch_size, seq_len = token_sequences.shape
        ble_to_camera = torch.tensor([
        3, 4, 6, 6, 6, 6, 6, 6, 1, 2, 0, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6
        ], device=self.device)
        # トークンシーケンスに基づきカメラインデックスを取得
        camera_indices = ble_to_camera[token_sequences]

        #print(f'camera_indices: {camera_indices.shape}')
        #print(f'combined_feature_mat: {combined_feature_mat.shape}')
        #print("combined_feature_mat.shape:", combined_feature_mat.shape)
        #print("インデックスの範囲: ", torch.min(camera_indices), torch.max(camera_indices))  # 使用しているインデックスを確認

        feature_mat = combined_feature_mat[
        torch.arange(batch_size, device=self.device).unsqueeze(1),  # バッチ次元
        camera_indices,                                            # カメラインデックス
        torch.arange(seq_len, device=self.device).unsqueeze(0)     # 時間次元
        ]

        return feature_mat
    

    def make_VAE_input_sim(self, token_sequences, time_index, img_dic):
        # 時間ごとの特徴量を格納
        feature_list = [img_dic[idx.item()].to(self.device) for idx in time_index]
        combined_feature_mat = torch.stack(feature_list, dim=0)
        combined_feature_mat = torch.nn.functional.pad(combined_feature_mat, (0, 0, 1, 1, 0, 1, 0, 0), mode='constant', value=0)

        #シーケンスの形状
        batch_size, seq_len = token_sequences.shape
        ble_to_camera = torch.tensor([
        3, 4, 6, 6, 6, 0, 0, 6, 1, 2, 0, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6
        ], device=self.device)
        camera_indices = ble_to_camera[token_sequences]
        feature_mat = combined_feature_mat[
        torch.arange(batch_size, device=self.device).unsqueeze(1),  # バッチ次元
        camera_indices,                                            # カメラインデックス
        torch.arange(seq_len, device=self.device).unsqueeze(0)     # 時間次元
        ]

        return feature_mat
    


    def make_VAE_input(self, token_sequences, time_index, img_dic):
        # time_index の各要素に対応する特徴量を格納するリスト
        feature_list = []

        # time_index の各要素についてループ処理
        for idx in time_index:
            # 該当する時間の特徴量が格納されたデータを取得
            idx_value = idx.item()  # tensor -> 整数
            time_feature_mat = img_dic[idx_value].to(self.device)
            #print(time_feature_mat.size())
            if torch.isnan(time_feature_mat).any() or torch.isinf(time_feature_mat).any():
                print(f"NaN or Inf detected in time_feature_mat for {idx_value}")
                break
            #time_feature_mat = time_feature_mat[:, :, :2]

            # サイズを合わせるために，time_feature_mat の前後をゼロ埋め
            feature_dim = time_feature_mat.size(2)
            padding_mat = torch.zeros((time_feature_mat.size(0), 1, feature_dim), device=self.device)
            time_feature_mat = torch.cat((padding_mat, time_feature_mat, padding_mat), dim=1)

            # データ欠損対応ようのパディング
            padding_mat2 = torch.zeros((1, time_feature_mat.size(1), feature_dim), device=self.device)
            time_feature_mat = torch.cat((time_feature_mat, padding_mat2), dim=0)

            # リストに追加
            feature_list.append(time_feature_mat)

        # time_index に対応するすべての特徴量を結合
        # 各 time_feature_mat が同じ形状であると仮定して結合
        combined_feature_mat = torch.stack(feature_list, dim=0)
        #print(combined_feature_mat.size())

        batch_size, seq_len = token_sequences.shape
        feature_mat = torch.zeros(batch_size, seq_len, combined_feature_mat.size(3), device=self.device)
        #print(feature_mat.size())

        # カメラの行と BLE の行の対応
        ble_to_camera_dic = {
            0: 3, 1: 4, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 6, 8: 1, 9: 2, 10: 0,
            11: 6, 12: 6, 13: 6, 14: 6, 15: 6, 16: 5, 17: 6, 18: 6, 19: 6, 20: 6, 21: 6, 22: 6
        }

        # トークンシーケンスに基づいて特徴行列を埋める
        for i in range(batch_size):
            for j in range(seq_len):
                feature_mat[i, j, :] = combined_feature_mat[i, ble_to_camera_dic[token_sequences[i, j].item()], j, :]

        return feature_mat
'''