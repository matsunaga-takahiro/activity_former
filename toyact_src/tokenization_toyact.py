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
        # self.SA = SA # 状態数×活動数の同時分布 
        self.A = A 
        self.N = N # ノード数
        # self.num_nodes = network.N # node feature埋め込みの時に必要では
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
        #self.to(self.device)

        self.act_token_sequences = None
        self.loc_token_sequences = None

    def tokenization(self, act_data, loc_data, mode, lmax = None):
        # self.route = route
        self.act_data = act_data
        self.loc_data = loc_data

        # num_data = len(self.route)
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
            # print('after padding tokenization', act_tokens.shape, act_tokens[0])

            # 各行の左端から最初に現れる非パディングトークンの位置を取得
            first_non_padding_indices_act = (~mask_act).float().argmax(dim=1)
            first_non_padding_indices_loc = (~mask_loc).float().argmax(dim=1)

            # 対応するトークンを開始トークンに置き換え
            act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act - 1] = self.act_SPECIAL_TOKENS["<b>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc - 1] = self.loc_SPECIAL_TOKENS["<b>"]
            # print('after begin tokenization', act_tokens.shape, act_tokens[0])


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
            # print('after padding tokenization', act_tokens.shape, act_tokens[0])

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
            # print('after end tokenization', act_tokens.shape, act_tokens[0])

            act_tokens = torch.cat((act_tokens, new_column_act), dim=1)
            loc_tokens = torch.cat((loc_tokens, new_column_loc), dim=1)
            # print('after completre tokenization', act_tokens.shape, act_tokens[0])
            # print('after completre tokenization', loc_tokens.shape, loc_tokens[0])


        elif mode == "next": # end only
            ##前と後ろにパディングトークンをくっつける
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1)            

            #前の塊の中で最初に出てくるパディングトークンを開始トークンにおきかえる
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]
            # print('with padding token ', act_tokens.shape, act_tokens[0])

            last_non_padding_indices_act = act_tokens.size(1) - 1 - (~mask_act).float().flip(dims=[1]).argmax(dim=1)
            last_non_padding_indices_loc = loc_tokens.size(1) - 1 - (~mask_loc).float().flip(dims=[1]).argmax(dim=1)

            act_tokens[torch.arange(act_tokens.size(0)), last_non_padding_indices_act + 1] = self.act_SPECIAL_TOKENS["<e>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), last_non_padding_indices_loc + 1] = self.loc_SPECIAL_TOKENS["<e>"]

            # act_tokens = torch.cat((act_tokens, new_column_act), dim=1)
            # loc_tokens = torch.cat((loc_tokens, new_column_loc), dim=1)
            # print('after next tokenization', act_tokens.shape, act_tokens[0])
            # print('after next tokenization', loc_tokens.shape, loc_tokens[0])

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

            # print('after correction::act_tokens', act_tokens.shape, act_tokens[0])
            # print('after correction::loc_tokens', loc_tokens.shape, loc_tokens[0])

            ## 間のトークンを全部<m>に置き換える # <m>部分が予測対象
            # act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act : last_non_padding_indices_act] = self.act_SPECIAL_TOKENS["<e>"]
            # loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc : last_non_padding_indices_loc] = self.loc_SPECIAL_TOKENS["<e>"]
            
            for i in range(loc_tokens.size(0)):
                start = first_non_padding_indices_loc[i]
                end = last_non_padding_indices_loc[i]
                loc_tokens[i, start:end] = self.loc_SPECIAL_TOKENS["<m>"]
                act_tokens[i, start:end] = self.act_SPECIAL_TOKENS["<m>"]

            '''
            #パディングではない最後のトークンを取得
            # seq_len = tokens.size(1)  # トークン列の長さ
            # last_non_padding_indices = seq_len - 1 - (~mask).float().flip(dims=[1]).argmax(dim=1)
            seq_len_act = act_tokens.size(1)
            seq_len_loc = loc_tokens.size(1)

            last_non_padding_indices_act = seq_len_act - 1 - (~mask_act).float().flip(dims=[1]).argmax(dim=1) # 全部18
            last_non_padding_indices_loc = seq_len_loc - 1 - (~mask_loc).float().flip(dims=[1]).argmax(dim=1)

            #last_non_padding_indices = (~mask).float().flip(dims=[1]).argmax(dim=1)
            # non_padding_values = tokens[batch_indices, last_non_padding_indices]
            non_padding_values_act = act_tokens[batch_indices_act, last_non_padding_indices_act]
            non_padding_values_loc = loc_tokens[batch_indices_loc, last_non_padding_indices_loc]

            # print('non_padding_values_act', non_padding_values_act, non_padding_values_loc)
            # print('shape', non_padding_values_act.shape, non_padding_values_loc.shape) # batchサイズと同じ

            #開始トークンの2つ先を<m>に置き換える
            # tokens[batch_indices, last_padding_in_head_indices + 2] = self.SPECIAL_TOKENS["<m>"]
            act_tokens[batch_indices_act, last_padding_in_head_indices_act + 2] = self.act_SPECIAL_TOKENS["<m>"]
            loc_tokens[batch_indices_loc, last_padding_in_head_indices_loc + 2] = self.loc_SPECIAL_TOKENS["<m>"]
            # print('with mask token ', act_tokens.shape, act_tokens[0])
            #もし，last_padding_in_head_indicesとlast_non_padding_indicesの差が2のとき，last_padding_in_head_indicesの該当する行の値を-1する(もし検出されたのが出発地と到着地の2ノードでかつそれが最後の時エラーになるため)
            # diff = last_non_padding_indices - last_padding_in_head_indices
            diff_act = last_non_padding_indices_act - last_padding_in_head_indices_act
            diff_loc = last_non_padding_indices_loc - last_padding_in_head_indices_loc
            
            # adjust_mask = diff == 2
            adjust_mask_act = diff_act == 2
            adjust_mask_loc = diff_loc == 2

            # last_padding_in_head_indices[adjust_mask] -= 1
            last_padding_in_head_indices_act[adjust_mask_act] -= 1
            last_padding_in_head_indices_loc[adjust_mask_loc] -= 1

            #開始トークンの3つ先を最後のトークンに置き換える
            # tokens[batch_indices, last_padding_in_head_indices + 3] = non_padding_values
            act_tokens[batch_indices_act, last_padding_in_head_indices_act + 3] = non_padding_values_act
            loc_tokens[batch_indices_loc, last_padding_in_head_indices_loc + 3] = non_padding_values_loc

            #開始トークンの4つ先を最後のトークンに置き換える
            # tokens[batch_indices, last_padding_in_head_indices + 4] = self.SPECIAL_TOKENS["<e>"]
            act_tokens[batch_indices_act, last_padding_in_head_indices_act + 4] = self.act_SPECIAL_TOKENS["<e>"]
            loc_tokens[batch_indices_loc, last_padding_in_head_indices_loc + 4] = self.loc_SPECIAL_TOKENS["<e>"]

            #5つ先以降はパディングトークンに置き換える
            # batch_size, seq_len = tokens.size()
            batch_size_act, seq_len_act = act_tokens.size()
            batch_size_loc, seq_len_loc = loc_tokens.size()

            # replace_mask = torch.arange(seq_len, device=tokens.device).unsqueeze(0) >= (last_padding_in_head_indices + 5).unsqueeze(1)
            replace_mask_act = torch.arange(seq_len_act, device=act_tokens.device).unsqueeze(0) >= (last_padding_in_head_indices_act + 5).unsqueeze(1)
            replace_mask_loc = torch.arange(seq_len_loc, device=loc_tokens.device).unsqueeze(0) >= (last_padding_in_head_indices_loc + 5).unsqueeze(1)

            # tokens[replace_mask] = self.SPECIAL_TOKENS["<p>"]
            act_tokens[replace_mask_act] = self.act_SPECIAL_TOKENS["<p>"]
            loc_tokens[replace_mask_loc] = self.loc_SPECIAL_TOKENS["<p>"]
            #print(tokens[0, :])
            #print(last_non_padding_indices[0])

            '''

            '''ここが明らかに変であっる'''
            print('with padding, mask, begin tokens', act_tokens.shape, act_tokens[0], loc_tokens[0])
            

        elif mode == "traveled":
            # new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<b>"], device=self.device) 
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<b>"], device=self.device)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<b>"], device=self.device)
            # tokens = torch.cat((new_column, tokens), dim=1)
            act_tokens = torch.cat((new_column_act, act_tokens), dim=1)
            loc_tokens = torch.cat((new_column_loc, loc_tokens), dim=1)
            print('after traveled tokenization', act_tokens.shape, act_tokens[0])

        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        # リストをPyTorchテンソルに変換
        return act_tokens.clone().detach().to(torch.long), loc_tokens.clone().detach().to(torch.long)
        # return tokens.clone().detach().to(torch.long)
    

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
        # print("token_sequences.shape:", token_sequences.shape) # B*S, 64*20のはず
        # print('unique token_sequences', torch.unique(token_sequences)) # 8-4
        token_sequences = token_sequences.long().to(self.device)
        batch_size, seq_len = token_sequences.shape # B*T
        node_features = self.network.node_features.to(self.device) # N*F
        feature_dim = node_features.shape[1] # F
        # print("node_features.shape:", node_features.shape) # N*F
        special_token_features = torch.zeros((4, feature_dim), device=self.device) # 4*F
        total_node_features = torch.cat((node_features, special_token_features), dim=0) ## 全ノード=loc token ids＋特別トークンids : (N+4)*F
        # print('total_node_features', total_node_features)
        feature_mat = torch.zeros((batch_size, seq_len, feature_dim), device=self.device)
        # print("feature_mat.shape:", feature_mat.shape)
        # print('token_sequences.shape:', token_sequences.shape) # 64-20
        # print('unique token_sequences.shape:', torch.unique(token_sequences)) # 8-4
        # print('total_node_features.shape:', total_node_features.shape) # 8-4
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
    

'''
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