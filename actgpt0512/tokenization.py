import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import sys
# from data_generation.network import Network

class Tokenization:
    def __init__(self, network, TT, Z, A):
        self.network = network
        self.TT = TT # time token num 
        self.Z = Z # zone num
        self.A = A # action num
        # self.num_nodes = network.N # node feature埋め込みの時に必要では
        self.time_SPECIAL_TOKENS = { 
            "<p>": TT,  
            "<e>": TT + 1,  
            "<b>": TT + 2, 
            "<m>": TT + 3,
        }
        self.loc_SPECIAL_TOKENS = { 
            "<p>": Z, 
            "<e>": Z + 1, 
            "<b>": Z + 2, 
            "<m>": Z + 3, 
        }
        self.act_SPECIAL_TOKENS = { # valueはトークンID（nodeid）
            "<p>": A,  # パディングトークン # シーケンス全部使う場合は不要
            "<e>": A + 1,  # 終了トークン 
            "<b>": A + 2,  # 開始トークン
            "<m>": A + 3,  # 非隣接ノードトークン # 現実には生成時にプリズム制約などをかけることができるはず（多分）
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.to(self.device)
        self.time_token_sequences = None   
        self.loc_token_sequences = None
        self.act_token_sequences = None

    def tokenization(self, time_data, loc_data, act_data, mode, lmax = None):
        self.time_data = time_data
        self.loc_data = loc_data
        self.act_data = act_data

        num_data_time = len(self.time_data) # sample数のはず
        num_data_act = len(self.act_data) 
        num_data_loc = len(self.loc_data)
        if num_data_time != num_data_act or num_data_time != num_data_loc:
            raise ValueError("The number of time, act, and loc data must be the same.")
        # 接続行列 
        # adjacency_matrix = self.network.adj_matrix.to(self.device)

        time_tokens = self.time_data.clone().to(self.device) 
        loc_tokens = self.loc_data.clone().to(self.device)
        act_tokens = self.act_data.clone().to(self.device)


        # モードに応じたトークン化処理 # time, loc, actで分けて用意→encoder, decoderに入れる
        if mode == "simple": # begin only
            ##前と後ろにパディングトークンをくっつける
            new_column_time = torch.full((num_data_time, 1), self.time_SPECIAL_TOKENS["<p>"], device=self.device)
            time_tokens = torch.cat((new_column_time, time_tokens, new_column_time), dim=1) 
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1) 
            
            maskmask_time = time_tokens == self.time_SPECIAL_TOKENS["<m>"] 
            maskmask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<m>"]
            maskmask_act = act_tokens == self.act_SPECIAL_TOKENS["<m>"]

            mask_time = time_tokens == self.time_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]

            first_non_padding_indices_time = (~mask_time).float().argmax(dim=1)
            first_non_padding_indices_loc = (~mask_loc).float().argmax(dim=1)
            first_non_padding_indices_act = (~mask_act).float().argmax(dim=1)
            time_tokens[torch.arange(time_tokens.size(0)), first_non_padding_indices_time - 1] = self.time_SPECIAL_TOKENS["<b>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc - 1] = self.loc_SPECIAL_TOKENS["<b>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act - 1] = self.act_SPECIAL_TOKENS["<b>"]

            # end padding 
            non_maskmask_act = ~(maskmask_act)  # 非マスク部分（Trueなら非マスク） # 揃わないとダメ
            valid_lengths = non_maskmask_act.float().sum(dim=1).long() - 2  # 各サンプルごとの非マスクトークン数
            last_non_maskmask_indices_act = valid_lengths + 1  # 最後の非マスク位置
            # loc_tokens[torch.arange(loc_tokens.size(0)), last_non_maskmask_indices_loc] = self.loc_SPECIAL_TOKENS["<e>"]
            # time_tokens[torch.arange(time_tokens.size(0)), last_non_maskmask_indices_loc] = self.time_SPECIAL_TOKENS["<e>"]
            # act_tokens[torch.arange(act_tokens.size(0)), last_non_maskmask_indices_loc] = self.act_SPECIAL_TOKENS["<e>"]

            ## replace all the tokens after end tokens into padding tokens ## checked
            for i in range(loc_tokens.size(0)):
                start = last_non_maskmask_indices_act[i] 
                time_tokens[i, start:] = self.time_SPECIAL_TOKENS["<p>"]
                loc_tokens[i, start:] = self.loc_SPECIAL_TOKENS["<p>"]
                act_tokens[i, start:] = self.act_SPECIAL_TOKENS["<p>"]


        elif mode == "complete": # begin and end
            new_column_time = torch.full((num_data_time, 1), self.time_SPECIAL_TOKENS["<p>"], device=self.device)
            time_tokens = torch.cat((new_column_time, time_tokens, new_column_time), dim=1) 
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1) 
            
            # 欠損部分を最初マスクにしてるので
            maskmask_time = time_tokens == self.time_SPECIAL_TOKENS["<m>"] 
            maskmask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<m>"]
            maskmask_act = act_tokens == self.act_SPECIAL_TOKENS["<m>"]

            #前の塊の中で最初に出てくるパディングトークンを開始トークンにおきかえる
            mask_time = time_tokens == self.time_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]

            first_non_padding_indices_time = (~mask_time).float().argmax(dim=1)
            first_non_padding_indices_loc = (~mask_loc).float().argmax(dim=1)
            first_non_padding_indices_act = (~mask_act).float().argmax(dim=1)
            time_tokens[torch.arange(time_tokens.size(0)), first_non_padding_indices_time - 1] = self.time_SPECIAL_TOKENS["<b>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc - 1] = self.loc_SPECIAL_TOKENS["<b>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act - 1] = self.act_SPECIAL_TOKENS["<b>"]

            # end padding 
            non_maskmask_loc = ~(maskmask_loc)  # 非マスク部分（Trueなら非マスク） # 揃わないとダメ
            valid_lengths = non_maskmask_loc.float().sum(dim=1).long() - 2  # 各サンプルごとの非マスクトークン数
            last_non_maskmask_indices_loc = valid_lengths + 1  # 最後の非マスク位置
            loc_tokens[torch.arange(loc_tokens.size(0)), last_non_maskmask_indices_loc] = self.loc_SPECIAL_TOKENS["<e>"]
            time_tokens[torch.arange(time_tokens.size(0)), last_non_maskmask_indices_loc] = self.time_SPECIAL_TOKENS["<e>"]
            act_tokens[torch.arange(act_tokens.size(0)), last_non_maskmask_indices_loc] = self.act_SPECIAL_TOKENS["<e>"]

            ## replace all the tokens after end tokens into padding tokens ## checked
            for i in range(loc_tokens.size(0)):
                start = last_non_maskmask_indices_loc[i] + 1
                time_tokens[i, start:] = self.time_SPECIAL_TOKENS["<p>"]
                loc_tokens[i, start:] = self.loc_SPECIAL_TOKENS["<p>"]
                act_tokens[i, start:] = self.act_SPECIAL_TOKENS["<p>"]

        elif mode == "next": # end only
            # first: add padding tokens to the head and tail positions ## checked
            new_column_time = torch.full((num_data_time, 1), self.time_SPECIAL_TOKENS["<p>"], device=self.device)
            time_tokens = torch.cat((new_column_time, time_tokens, new_column_time), dim=1) 
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1) 

            maskmask_time = time_tokens == self.time_SPECIAL_TOKENS["<m>"] 
            maskmask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<m>"]
            maskmask_act = act_tokens == self.act_SPECIAL_TOKENS["<m>"]

            # end padding 
            non_maskmask_loc = ~(maskmask_loc)  # 非マスク部分（Trueなら非マスク） # 揃わないとダメ
            valid_lengths = non_maskmask_loc.float().sum(dim=1).long() - 2  # 各サンプルごとの非マスクトークン数
            last_non_maskmask_indices_loc = valid_lengths + 1  # 最後の非マスク位置
            loc_tokens[torch.arange(loc_tokens.size(0)), last_non_maskmask_indices_loc] = self.loc_SPECIAL_TOKENS["<e>"]
            time_tokens[torch.arange(time_tokens.size(0)), last_non_maskmask_indices_loc] = self.time_SPECIAL_TOKENS["<e>"]
            act_tokens[torch.arange(act_tokens.size(0)), last_non_maskmask_indices_loc] = self.act_SPECIAL_TOKENS["<e>"]

            ## replace all the tokens after end tokens into padding tokens ## checked
            for i in range(loc_tokens.size(0)):
                start = last_non_maskmask_indices_loc[i] + 1
                time_tokens[i, start:] = self.time_SPECIAL_TOKENS["<p>"]
                loc_tokens[i, start:] = self.loc_SPECIAL_TOKENS["<p>"]
                act_tokens[i, start:] = self.act_SPECIAL_TOKENS["<p>"]
            
            ## ここで最初のトークンを削除している！！
            time_tokens = time_tokens[:, 1:]
            loc_tokens = loc_tokens[:, 1:]
            act_tokens = act_tokens[:, 1:]

            # 末尾にパディングトークンを追加
            new_column_time = torch.full((num_data_time, 1), self.time_SPECIAL_TOKENS["<p>"], device=self.device)
            time_tokens = torch.cat((time_tokens, new_column_time), dim=1)
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device)
            act_tokens = torch.cat((act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device)
            loc_tokens = torch.cat((loc_tokens, new_column_loc), dim=1)

        elif mode == "discontinuous": 
            # first: add padding tokens to the head and tail positions ## checked
            new_column_time = torch.full((num_data_time, 1), self.time_SPECIAL_TOKENS["<p>"], device=self.device)
            time_tokens = torch.cat((new_column_time, time_tokens, new_column_time), dim=1) 
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device) 
            act_tokens = torch.cat((new_column_act, act_tokens, new_column_act), dim=1)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device) 
            loc_tokens = torch.cat((new_column_loc, loc_tokens, new_column_loc), dim=1) 

            # # 欠損部分を最初マスクにしてるので
            maskmask_time = time_tokens == self.time_SPECIAL_TOKENS["<m>"] 
            maskmask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<m>"]
            maskmask_act = act_tokens == self.act_SPECIAL_TOKENS["<m>"]

            # begin padding ## checked
            mask_time = time_tokens == self.time_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            mask_loc = loc_tokens == self.loc_SPECIAL_TOKENS["<p>"]
            mask_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]

            first_non_padding_indices_time = (~mask_time).float().argmax(dim=1)
            first_non_padding_indices_loc = (~mask_loc).float().argmax(dim=1)
            first_non_padding_indices_act = (~mask_act).float().argmax(dim=1)
            time_tokens[torch.arange(time_tokens.size(0)), first_non_padding_indices_time - 1] = self.time_SPECIAL_TOKENS["<b>"]
            loc_tokens[torch.arange(loc_tokens.size(0)), first_non_padding_indices_loc - 1] = self.loc_SPECIAL_TOKENS["<b>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_non_padding_indices_act - 1] = self.act_SPECIAL_TOKENS["<b>"]
            
            # end padding 
            non_maskmask_loc = ~(maskmask_loc)  # 非マスク部分（Trueなら非マスク） # 揃わないとダメ
            valid_lengths = non_maskmask_loc.float().sum(dim=1).long() - 2  # 各サンプルごとの非マスクトークン数(前後の開始トークンは除く)
            # print('valid_lengths', valid_lengths[0]) 
            last_non_maskmask_indices_loc = valid_lengths + 1  # 最後の非マスク位置 = valid_lengthと同じ（０番目は開始トークンなので）
            loc_tokens[torch.arange(loc_tokens.size(0)), last_non_maskmask_indices_loc] = self.loc_SPECIAL_TOKENS["<e>"]
            time_tokens[torch.arange(time_tokens.size(0)), last_non_maskmask_indices_loc] = self.time_SPECIAL_TOKENS["<e>"]
            act_tokens[torch.arange(act_tokens.size(0)), last_non_maskmask_indices_loc] = self.act_SPECIAL_TOKENS["<e>"]

            ## replace all the tokens after end tokens into padding tokens ## checked
            for i in range(loc_tokens.size(0)):
                start = last_non_maskmask_indices_loc[i] + 1
                time_tokens[i, start:] = self.time_SPECIAL_TOKENS["<p>"]
                loc_tokens[i, start:] = self.loc_SPECIAL_TOKENS["<p>"]
                act_tokens[i, start:] = self.act_SPECIAL_TOKENS["<p>"]
            
            ## adding mask tokens # checked
            for i in range(loc_tokens.size(0)):
                start = first_non_padding_indices_loc[i]
                end = last_non_maskmask_indices_loc[i]
                time_tokens[i, start:end] = self.time_SPECIAL_TOKENS["<m>"]
                loc_tokens[i, start:end] = self.loc_SPECIAL_TOKENS["<m>"]
                act_tokens[i, start:end] = self.act_SPECIAL_TOKENS["<m>"]

        elif mode == "traveled":
            new_column_time = torch.full((num_data_time, 1), self.time_SPECIAL_TOKENS["<p>"], device=self.device)
            new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device)
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device)

            time_tokens = torch.cat((new_column_time, time_tokens), dim=1)
            loc_tokens = torch.cat((new_column_loc, loc_tokens), dim=1)
            act_tokens = torch.cat((new_column_act, act_tokens), dim=1)
            # print('after traveled tokenization', act_tokens.shape, act_tokens[0])

        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        # リストをPyTorchテンソルに変換
        return time_tokens.clone().detach().to(torch.long), loc_tokens.clone().detach().to(torch.long), act_tokens.clone().detach().to(torch.long)
    

    def mask(self, mask_rate): # ランダムにトークンを <m> に置換（mask） # つかわない？？？？
        time_mask_token_id = self.time_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ
        loc_mask_token_id = self.loc_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ
        act_mask_token_id = self.act_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ

        time_token_sequences = self.time_data.clone().to(self.device)
        loc_token_sequences = self.loc_data.clone().to(self.device)
        act_token_sequences = self.act_data.clone().to(self.device)
        batch_size, seq_len = act_token_sequences.shape # どうせ形状は共通

        # マスクを適用する位置を計算（1列目と最後の列を除外）
        mask_tokens = torch.rand(batch_size, seq_len) < mask_rate
        mask_tokens[:, 0] = False  # 1列目はマスクしない
        mask_tokens[:, -1] = False  # 最後の列はマスクしない

        # マスクトークンを適用
        # token_sequences[mask_tokens] = mask_token_id
        time_token_sequences[mask_tokens] = time_mask_token_id
        loc_token_sequences[mask_tokens] = loc_mask_token_id
        act_token_sequences[mask_tokens] = act_mask_token_id

        # どうせ長さは同じ
        time_new_column = torch.full((len(self.time_data), 1), self.time_SPECIAL_TOKENS["<b>"], device=self.device) 
        time_new_column2 = torch.full((len(self.time_data), 1), self.time_SPECIAL_TOKENS["<e>"], device=self.device)
        act_new_column = torch.full((len(self.act_data), 1), self.act_SPECIAL_TOKENS["<b>"], device=self.device) 
        act_new_column2 = torch.full((len(self.act_data), 1), self.act_SPECIAL_TOKENS["<e>"], device=self.device) 
        loc_new_column = torch.full((len(self.loc_data), 1), self.loc_SPECIAL_TOKENS["<b>"], device=self.device)
        loc_new_column2 = torch.full((len(self.loc_data), 1), self.loc_SPECIAL_TOKENS["<e>"], device=self.device)

        time_token_sequences = torch.cat((time_token_sequences, time_new_column2), dim=1) # torch.cat dim=0だと行方向＝縦方向に，dim=1だと列方向＝横方向にくっつける
        time_token_sequences = torch.cat((time_new_column, time_token_sequences), dim=1)
        act_token_sequences = torch.cat((act_token_sequences, act_new_column2), dim=1) 
        act_token_sequences = torch.cat((act_new_column, act_token_sequences), dim=1)
        loc_token_sequences = torch.cat((loc_token_sequences, loc_new_column2), dim=1) 
        loc_token_sequences = torch.cat((loc_new_column, loc_token_sequences), dim=1)

        self.time_token_sequences = time_token_sequences # B*T # context tokens: B * C
        self.loc_token_sequences = loc_token_sequences
        self.act_token_sequences = act_token_sequences

        return time_token_sequences, loc_token_sequences, act_token_sequences
    

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


## 入力データでは＜b＞がないのと<s>が最後に連続している→最後の２つを<e><p>に置換する
class TokenizationGPT:
    def __init__(self, network, TT, A):
        self.network = network
        self.TT = TT # time token num 28
        # self.Z = Z # zone num.
        self.A = A # action num 8
        # self.num_nodes = network.N # node feature埋め込みの時に必要では
        self.time_SPECIAL_TOKENS = { 
            "<p>": TT,  
            "<E>": TT + 1,  
            "<B>": TT + 2, 
            "<m>": TT + 3,
        }
        # self.loc_SPECIAL_TOKENS = { 
        #     "<p>": Z, 
        #     "<e>": Z + 1, 
        #     "<b>": Z + 2, 
        #     "<m>": Z + 3, 
        # }
        self.act_SPECIAL_TOKENS = { # valueはトークンID（nodeid）
            "<p>": A,  # パディングトークン # シーケンス全部使う場合は不要
            "<E>": A + 1,  # 終了トークン 
            "<B>": A + 2,  # 開始トークン
            "<m>": A + 3,  # 非隣接ノードトークン # 現実には生成時にプリズム制約などをかけることができるはず（多分）
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_token_sequences = None   
        self.act_token_sequences = None

    def tokenization(self, time_data, act_data, mode):
        self.time_data = time_data
        # self.loc_data = loc_data
        self.act_data = act_data

        num_data_time = len(self.time_data) # sample数のはず
        num_data_act = len(self.act_data) 
        # num_data_loc = len(self.loc_data) 
        if num_data_time != num_data_act: # or num_data_time != num_data_loc:
            raise ValueError("The number of time, act, and loc data must be the same.")

        time_tokens = self.time_data.clone().to(self.device) 
        act_tokens = self.act_data.clone().to(self.device)

        # モードに応じたトークン化処理 # time, loc, actで分けて用意→encoder, decoderに入れる
        if mode == "simple": # begin only
            pad_time = time_tokens == self.time_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            pad_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]

            # 最後の2つの終了トークンを<p><p>に置換
            first_pad_time_idx = (~pad_time).float().argmin(dim=1)
            first_pad_act_idx = (~pad_act).float().argmin(dim=1)

            time_tokens[torch.arange(time_tokens.size(0)), first_pad_time_idx - 1] = self.time_SPECIAL_TOKENS["<p>"]
            time_tokens[torch.arange(time_tokens.size(0)), first_pad_time_idx - 2] = self.time_SPECIAL_TOKENS["<p>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_pad_act_idx - 1] = self.act_SPECIAL_TOKENS["<p>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_pad_act_idx - 2] = self.act_SPECIAL_TOKENS["<p>"]

            # 冒頭のuseridを<B>に置換
            # print('-----in simple -----')    
            # print('time tokens', time_tokens[0])
            # time_tokens[torch.arange(time_tokens.size(0)), 0] = self.time_SPECIAL_TOKENS["<B>"]
            # act_tokens[torch.arange(act_tokens.size(0)), 0] = self.act_SPECIAL_TOKENS["<B>"]
            # 新しい <B> トークン列を作成（すべて <B> ID）
            bos_time = torch.full((time_tokens.size(0), 1), self.time_SPECIAL_TOKENS["<B>"], dtype=time_tokens.dtype, device=time_tokens.device)
            bos_act  = torch.full((act_tokens.size(0), 1), self.act_SPECIAL_TOKENS["<B>"], dtype=act_tokens.dtype, device=act_tokens.device)
            
            # 先頭に <B> を追加
            time_tokens = torch.cat([bos_time, time_tokens], dim=1)
            act_tokens  = torch.cat([bos_act,  act_tokens],  dim=1)

            # print('-----after simple -----')
            # print('time tokens', time_tokens[0])

            # 最後一個抜く（正解データとして使われるから）
            time_tokens = time_tokens[:, :-1]
            act_tokens = act_tokens[:, :-1]

        elif mode == "complete": # begin and end
            # 冒頭に<B>を追加
            # print('-----in complete-----')    
            # print('time tokens', time_tokens[0])
            # time_tokens[torch.arange(time_tokens.size(0)), 0] = self.time_SPECIAL_TOKENS["<B>"]
            # act_tokens[torch.arange(act_tokens.size(0)), 0] = self.act_SPECIAL_TOKENS["<B>"]
            bos_time = torch.full((time_tokens.size(0), 1), self.time_SPECIAL_TOKENS["<B>"], dtype=time_tokens.dtype, device=time_tokens.device)
            bos_act  = torch.full((act_tokens.size(0), 1), self.act_SPECIAL_TOKENS["<B>"], dtype=act_tokens.dtype, device=act_tokens.device)
            
            # 先頭に <B> を追加
            time_tokens = torch.cat([bos_time, time_tokens], dim=1)
            act_tokens  = torch.cat([bos_act,  act_tokens],  dim=1)

            # print('-----after in complete-----')
            # print('time tokens', time_tokens[0])
            # 非パディングの最後尾をEに置換
            pad_time = time_tokens == self.time_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            pad_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]
            first_pad_time_idx = (pad_time).float().argmin(dim=1)
            first_pad_act_idx = (pad_act).float().argmin(dim=1)

            time_tokens[torch.arange(time_tokens.size(0)), first_pad_time_idx - 1] = self.time_SPECIAL_TOKENS["<p>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_pad_act_idx - 1] = self.act_SPECIAL_TOKENS["<p>"]
            time_tokens[torch.arange(time_tokens.size(0)), first_pad_time_idx - 2] = self.time_SPECIAL_TOKENS["<E>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_pad_act_idx - 2] = self.act_SPECIAL_TOKENS["<E>"]

            # 最後一個抜く（正解データとして使われるから）
            time_tokens = time_tokens[:, :-1]
            act_tokens = act_tokens[:, :-1]

        elif mode == "next": # end only
            # first: add padding tokens to the head and tail positions ## checked
            pad_time = time_tokens == self.time_SPECIAL_TOKENS["<p>"] # dictの値を参照している
            pad_act = act_tokens == self.act_SPECIAL_TOKENS["<p>"]


            # # 最後の終了トークンを<e><p>に置換
            first_pad_time_idx = (~pad_time).float().argmin(dim=1)
            first_pad_act_idx = (~pad_act).float().argmin(dim=1)     

            # print('-------in next------')
            # print('pad time', pad_time[0])

            # print('time tokens', time_tokens[0])
            # print('act tokens', act_tokens[0])

            time_tokens[torch.arange(time_tokens.size(0)), first_pad_time_idx - 1] = self.time_SPECIAL_TOKENS["<p>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_pad_act_idx - 1] = self.act_SPECIAL_TOKENS["<p>"]
            time_tokens[torch.arange(time_tokens.size(0)), first_pad_time_idx - 2] = self.time_SPECIAL_TOKENS["<E>"]
            act_tokens[torch.arange(act_tokens.size(0)), first_pad_act_idx - 2] = self.act_SPECIAL_TOKENS["<E>"]  

            # print('-------after in next------')
            # print('time tokens', time_tokens[0])
            # print('act tokens', act_tokens[0])

            # 冒頭を抜く
            # time_tokens = time_tokens[:, 1:]
            # act_tokens = act_tokens[:, 1:]
            
            # 冒頭の2列を<p>に置換
            # time_tokens[torch.arange(time_tokens.size(0)), 0] = self.time_SPECIAL_TOKENS["<p>"]
            # act_tokens[torch.arange(act_tokens.size(0)), 0] = self.act_SPECIAL_TOKENS["<p>"]
            # time_tokens[torch.arange(time_tokens.size(0)), 1] = self.time_SPECIAL_TOKENS["<p>"]
            # act_tokens[torch.arange(act_tokens.size(0)), 1] = self.act_SPECIAL_TOKENS["<p>"]

        elif mode == "traveled":
            new_column_time = torch.full((num_data_time, 1), self.time_SPECIAL_TOKENS["<p>"], device=self.device)
            # new_column_loc = torch.full((num_data_loc, 1), self.loc_SPECIAL_TOKENS["<p>"], device=self.device)
            new_column_act = torch.full((num_data_act, 1), self.act_SPECIAL_TOKENS["<p>"], device=self.device)

            time_tokens = torch.cat((new_column_time, time_tokens), dim=1)
            # loc_tokens = torch.cat((new_column_loc, loc_tokens), dim=1)
            act_tokens = torch.cat((new_column_act, act_tokens), dim=1)
            # print('after traveled tokenization', act_tokens.shape, act_tokens[0])

        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        # リストをPyTorchテンソルに変換
        return time_tokens.clone().detach().to(torch.long), act_tokens.clone().detach().to(torch.long)
    

    def mask(self, mask_rate): # ランダムにトークンを <m> に置換（mask） # つかわない？？？？
        time_mask_token_id = self.time_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ
        # loc_mask_token_id = self.loc_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ
        act_mask_token_id = self.act_SPECIAL_TOKENS["<m>"] # 辞書の値を参照しているだけ

        time_token_sequences = self.time_data.clone().to(self.device)
        # loc_token_sequences = self.loc_data.clone().to(self.device)
        act_token_sequences = self.act_data.clone().to(self.device)
        batch_size, seq_len = act_token_sequences.shape # どうせ形状は共通

        # マスクを適用する位置を計算（1列目と最後の列を除外）
        mask_tokens = torch.rand(batch_size, seq_len) < mask_rate
        mask_tokens[:, 0] = False  # 1列目はマスクしない
        mask_tokens[:, -1] = False  # 最後の列はマスクしない

        # マスクトークンを適用
        # token_sequences[mask_tokens] = mask_token_id
        time_token_sequences[mask_tokens] = time_mask_token_id
        # loc_token_sequences[mask_tokens] = loc_mask_token_id
        act_token_sequences[mask_tokens] = act_mask_token_id

        # どうせ長さは同じ
        time_new_column = torch.full((len(self.time_data), 1), self.time_SPECIAL_TOKENS["<B>"], device=self.device) 
        time_new_column2 = torch.full((len(self.time_data), 1), self.time_SPECIAL_TOKENS["<E>"], device=self.device)
        act_new_column = torch.full((len(self.act_data), 1), self.act_SPECIAL_TOKENS["<B>"], device=self.device) 
        act_new_column2 = torch.full((len(self.act_data), 1), self.act_SPECIAL_TOKENS["<E>"], device=self.device) 
        # loc_new_column = torch.full((len(self.loc_data), 1), self.loc_SPECIAL_TOKENS["<b>"], device=self.device)
        # loc_new_column2 = torch.full((len(self.loc_data), 1), self.loc_SPECIAL_TOKENS["<e>"], device=self.device)

        time_token_sequences = torch.cat((time_token_sequences, time_new_column2), dim=1) # torch.cat dim=0だと行方向＝縦方向に，dim=1だと列方向＝横方向にくっつける
        time_token_sequences = torch.cat((time_new_column, time_token_sequences), dim=1)
        act_token_sequences = torch.cat((act_token_sequences, act_new_column2), dim=1) 
        act_token_sequences = torch.cat((act_new_column, act_token_sequences), dim=1)
        # loc_token_sequences = torch.cat((loc_token_sequences, loc_new_column2), dim=1) 
        # loc_token_sequences = torch.cat((loc_new_column, loc_token_sequences), dim=1)

        self.time_token_sequences = time_token_sequences # B*T # context tokens: B * C
        # self.loc_token_sequences = loc_token_sequences
        self.act_token_sequences = act_token_sequences

        return time_token_sequences, act_token_sequences
    

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