import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
# from data_generation.network import Network
#ルートはタイムステップで作るので，ルートデータはdata_num * Tの配列
#1タイムステップで滞在するか別のリンクに移動することしかできないと仮定する←ここの仮定は実データでは変更するかも

class Tokenization:
    def __init__(self, network, A, N):
        self.network = network
        self.A = A 
        self.N = N # ノード数
        self.act_SPECIAL_TOKENS = { # valueはトークンID（nodeid）
            "<p>": A,  # パディングトークン # シーケンス全部使う場合は不要 4
            "<e>": A + 1,  # 終了トークン # 5
            "<b>": A + 2,  # 開始トークン 6
            "<m>": A + 3,  # 非隣接ノードトークン # 7 # 現実には生成時にプリズム制約などをかけることができるはず（多分）
        }
        self.loc_SPECIAL_TOKENS = { # valueはトークンID（nodeid）
            "<p>": N,  # パディングトークン # シーケンス全部使う場合は不要
            "<e>": N + 1,  # 終了トークン # 
            "<b>": N + 2,  # 開始トークン
            "<m>": N + 3,  # 非隣接ノードトークン # 19 # 現実には生成時にプリズム制約などをかけることができるはず（多分）
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act_token_sequences = None
        self.loc_token_sequences = None

    def tokenization(self, act_data, loc_data, mode, lmax = None):
        self.act_data = act_data
        self.loc_data = loc_data

        num_data_act = len(self.act_data) # sample数のはず
        num_data_loc = len(self.loc_data)

        # 接続行列 # 不要かなーとりあえず
        # adjacency_matrix = self.network.adj_matrix.to(self.device)

        # tokens = self.route.clone().to(self.device) 
        act_tokens = self.act_data.clone().to(self.device)
        loc_tokens = self.loc_data.clone().to(self.device)

        # モードに応じたトークン化処理 # act, locで分けて用意→encoder, decoderに入れる
        if mode == "simple": # begin only
            ##前と後ろにパディングトークンをくっつける
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) # torch.full((num_data_loc, 1), ...) は、サイズ (num_data_loc, 1) のテンソルを作り、すべての値を <p> トークンのIDで埋めています
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1) # 前の塊の中で最初に出てくるパディングトークンを開始トークンにおきかえる
            
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]

            # 各行の左端から最初に現れる非パディングトークンの位置を取得
            first_non_padding_indices_act = (~mask_act).float().argmax(dim=1)
            first_non_padding_indices_loc = (~mask_loc).float().argmax(dim=1)

            # 対応するトークンを開始トークンに置き換え
            act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act - 1] = self.act_SPECIAL_TOKENS["<b>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc - 1] = self.loc_SPECIAL_TOKENS["<b>"]

        elif mode == "complete": # begin and end
            # new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<p>"], device=self.device) 
            # tokens = torch.cat((new_column, tokens, new_column), dim=1)
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1)            

            #前の塊の中で最初に出てくるパディングトークンを開始トークンにおきかえる
            # mask = tokens == self.SPECIAL_TOKENS["<p>"]
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]

            # 各行の左端から最初に現れる非パディングトークンの位置を取得
            first_non_padding_indices_act = (~mask_act).float().argmax(dim=1)
            first_non_padding_indices_loc = (~mask_loc).float().argmax(dim=1)

            # 対応するトークンを開始トークンに置き換え
            act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act - 1] = self.act_SPECIAL_TOKENS["<b>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc - 1] = self.loc_SPECIAL_TOKENS["<b>"]

            last_non_padding_indices_act = act_tokens.size(1) - 1 - (~mask_act).float().flip(dims=[1]).argmax(dim=1)
            last_non_padding_indices_loc = loc_tokens.size(1) - 1 - (~mask_loc).float().flip(dims=[1]).argmax(dim=1)

            act_tokens[torch.arange(act_tokens.size(0)), last_non_padding_indices_act + 1] = self.act_SPECIAL_TOKENS["<e>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), last_non_padding_indices_loc + 1] = self.loc_SPECIAL_TOKENS["<e>"]

            act_tokens = torch.cat((act_tokens, new_column_act), dim=1)
            loc_tokens = torch.cat((loc_tokens, new_column_loc), dim=1)

        elif mode == "next": # end only
            ##前と後ろにパディングトークンをくっつける
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1)            

            #前の塊の中で最初に出てくるパディングトークンを開始トークンにおきかえる
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]

            last_non_padding_indices_act = act_tokens.size(1) - 1 - (~mask_act).float().flip(dims=[1]).argmax(dim=1)
            last_non_padding_indices_loc = loc_tokens.size(1) - 1 - (~mask_loc).float().flip(dims=[1]).argmax(dim=1)

            act_tokens[torch.arange(act_tokens.size(0)), last_non_padding_indices_act + 1] = self.act_SPECIAL_TOKENS["<e>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), last_non_padding_indices_loc + 1] = self.loc_SPECIAL_TOKENS["<e>"]

        elif mode == "discontinuous": #### ここが変
            ##前と後ろにパディングトークンをくっつける
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1)            

            # 各行の左端から最初に現れる非パディングトークンの位置を取得
            # mask = tokens == self.SPECIAL_TOKENS["<p>"]
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"] # true or false のリスト
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]

            first_non_padding_indices_act = (~mask_act).float().argmax(dim=1)
            first_non_padding_indices_loc = (~mask_loc).float().argmax(dim=1)
            # first_non_padding_indices = (~mask).float().argmax(dim=1)

            # 対応するトークンを開始トークンに置き換え
            # tokens[batch_indices, last_padding_in_head_indices] = self.SPECIAL_TOKENS["<b>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act - 1] = self.act_SPECIAL_TOKENS["<b>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc - 1] = self.loc_SPECIAL_TOKENS["<b>"]

            # end paddingをつける
            last_non_padding_indices_act = act_tokens.size(1) - 1 - (~mask_act).float().flip(dims=[1]).argmax(dim=1)
            last_non_padding_indices_loc = loc_tokens.size(1) - 1 - (~mask_loc).float().flip(dims=[1]).argmax(dim=1)

            act_tokens[torch.arange(act_tokens.size(0)), last_non_padding_indices_act + 1] = self.act_SPECIAL_TOKENS["<e>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), last_non_padding_indices_loc + 1] = self.loc_SPECIAL_TOKENS["<e>"]

            ## 間のトークンを全部<m>に置き換える # <m>部分が予測対象
            # act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act : last_non_padding_indices_act] = self.act_SPECIAL_TOKENS["<e>"]
            # loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc : last_non_padding_indices_loc] = self.loc_SPECIAL_TOKENS["<e>"]
            
            for i in range(loc_tokens.size(0)):
                start = first_non_padding_indices_loc[i]
                end = last_non_padding_indices_loc[i]
                loc_tokens[i, start:end] = self.loc_SPECIAL_TOKENS["<m>"]
                act_tokens[i, start:end] = self.act_SPECIAL_TOKENS["<m>"]

        elif mode == "traveled":
            # new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<b>"], device=self.device) 
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<b>"], device=self.device)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<b>"], device=self.device)
            # tokens = torch.cat((new_column, tokens), dim=1)
            act_tokens = torch.cat((new_column_act, act_tokens), dim=1)
            loc_tokens = torch.cat((new_column_loc, loc_tokens), dim=1)

        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        # リストをPyTorchテンソルに変換
        return act_tokens.clone().detach().to(torch.long), loc_tokens.clone().detach().to(torch.long)    

    def mask(self, mask_rate): # ランダムにトークンを <m> に置換（mask） # つかわない？？？？
        act_mask_token_id = self.act_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ
        loc_mask_token_id = self.loc_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ

        # token_sequences = self.route.clone().to(self.device)
        act_token_sequences = self.act_data.clone().to(self.device)
        loc_token_sequences = self.loc_data.clone().to(self.device)
        batch_size, seq_len = act_token_sequences.shape # どうせ形状は共通

        # マスクを適用する位置を計算（1列目と最後の列を除外）
        mask_tokens = torch.rand(batch_size, seq_len) < mask_rate
        mask_tokens[:, 0] = False  # 1列目はマスクしない
        mask_tokens[:, -1] = False  # 最後の列はマスクしない

        # マスクトークンを適用
        # token_sequences[mask_tokens] = mask_token_id
        act_token_sequences[mask_tokens] = act_mask_token_id
        loc_token_sequences[mask_tokens] = loc_mask_token_id

        # どうせ長さは同じ
        act_new_column = torch.full((len(self.act_data), 1), self.act_SPECIAL_TOKENS["<b>"], device=self.device) 
        act_new_column2 = torch.full((len(self.act_data), 1), self.act_SPECIAL_TOKENS["<e>"], device=self.device) 
        loc_new_column = torch.full((len(self.loc_data), 1), self.loc_SPECIAL_TOKENS["<b>"], device=self.device)
        loc_new_column2 = torch.full((len(self.loc_data), 1), self.loc_SPECIAL_TOKENS["<e>"], device=self.device)

        act_token_sequences = torch.cat((act_token_sequences, act_new_column2), dim=1) # torch.cat dim=0だと行方向＝縦方向に，dim=1だと列方向＝横方向にくっつける
        act_token_sequences = torch.cat((act_new_column, act_token_sequences), dim=1)
        loc_token_sequences = torch.cat((loc_token_sequences, loc_new_column2), dim=1) # torch.cat dim=0だと行方向＝縦方向に，dim=1だと列方向＝横方向にくっつける
        loc_token_sequences = torch.cat((loc_new_column, loc_token_sequences), dim=1)

        self.act_token_sequences = act_token_sequences
        self.loc_token_sequences = loc_token_sequences

        return act_token_sequences, loc_token_sequences
    

    #### 特徴量の埋め込み
    def make_feature_mat(self, token_sequences): # 今の所ノードの特徴量しかないので，token_sequencesはノードのトークン列を想定

        token_sequences = token_sequences.long().to(self.device)
        batch_size, seq_len = token_sequences.shape # B*T
        node_features = self.network.node_features.to(self.device) # N*F
        feature_dim = node_features.shape[1] # F
        special_token_features = torch.zeros((4, feature_dim), device=self.device) # 4*F
        total_node_features = torch.cat((node_features, special_token_features), dim=0) ## 全ノード=loc token ids＋特別トークンids : (N+4)*F
        feature_mat = torch.zeros((batch_size, seq_len, feature_dim), device=self.device)
        feature_mat = total_node_features[token_sequences] # token_sequenceをインデックスとして渡すと対応する特徴量が取得できる
        return feature_mat # B*T*F # 


    ##############################
    ######## below;; not use ########
    ##############################

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
        # トークンシーケンスに基づきカメラインデックスを取得
        camera_indices = ble_to_camera[token_sequences]

        feature_mat = combined_feature_mat[
        torch.arange(batch_size, device=self.device).unsqueeze(1),  # バッチ次元
        camera_indices,                                            # カメラインデックス
        torch.arange(seq_len, device=self.device).unsqueeze(0)     # 時間次元
        ]
        return feature_mat
    