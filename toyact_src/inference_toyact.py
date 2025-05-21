## 推論
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from datetime import datetime, timedelta

from network_toyact import Network
from tokenization_toyact import Tokenization
# from ActFormer.RoutesFormer.toyact_src.actformer_toyact import Actformer
from actformer_toyact import Actformer
# from utils.logger import logger
import matplotlib.pyplot as plt
# import wandb
from torch.utils.data import Dataset, random_split, DataLoader
# from data_generation.Recursive import TimeFreeRecursiveLogit
import os

# torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
base_path = '/Users/matsunagatakahiro/Desktop/res2025/ActFormer'

########################################
# 1. 補助的な関数群
########################################

def is_subsequence(R, R_i):

    i, j = 0, 0  # R のインデックス, R_i のインデックス
    len_R, len_R_i = len(R), len(R_i)

    # 双方のテンソルをスキャン
    while i < len_R and j < len_R_i:
        if R[i] == R_i[j]:  # 一致する場合、R の次の要素を確認
            i += 1
        j += 1  # R_i の次の要素に進む

    # R のすべての要素が見つかったかを判定
    return i == len_R


def is_subsequence_batch(R, R_i, ignore_value_lis):
    batch_size = R.size(0)
    results = []

    for i in range(batch_size):
        # 各行について無視する値を除外
        r_row = R[i][~torch.isin(R[i], torch.tensor(ignore_value_lis).to(device))]
        r_i_row = R_i[i]
        
        # 部分列判定
        is_subseq = is_subsequence(r_row, r_i_row)
        results.append(is_subseq)
    
    # 結果をテンソルとして返す
    return torch.tensor(results, dtype=torch.bool)


class Neighbor: # 隣接行列で制約
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.N = adj_matrix.size(0)
        self.vocab_size = self.N + 4
    
    def make_neighbor_mask(self, newest_zone, d):
        #inputはbatch_size,最新のノード番号を格納
        results = []
        special_zone_d = torch.tensor([0,1,0,0]) #<e>のみを1に，そのほかを0に
        special_zone_neighbor = torch.tensor([0,0,0,0]) #<e>のみを1に，そのほかを0に
        special_zone_padding = torch.tensor([1,0,0,0]) #<p>のみを1に，そのほかを0に
        for i in range(newest_zone.size(0)):
            if newest_zone[i] == d[i]:
                neighbor = self.adj_matrix[newest_zone[i]]
                neighbor = torch.cat([neighbor, special_zone_d])
                results.append(neighbor)
            elif newest_zone[i] <= (self.N - 1):
                neighbor = self.adj_matrix[newest_zone[i]]
                neighbor = torch.cat([neighbor, special_zone_neighbor])
                results.append(neighbor)
            elif newest_zone[i] == self.N:
                neighbor = torch.cat([torch.zeros(self.N), special_zone_padding])
                results.append(neighbor)
            elif newest_zone[i] == self.N + 2:
                neighbor = torch.cat([torch.ones(self.N), special_zone_neighbor])
                results.append(neighbor)
            else:
                results.append(torch.zeros(self.vocab_size))
            
        results = torch.stack(results)
        results = torch.where(results == 0, float('-inf'), 0.0)
        return results


########################################
# 2. データの前処理・準備
########################################
#input data
# adj_matrix = torch.load(os.path.join(base_path, 'toy_data_generation/grid_adjacency_matrix.pt'), weights_only=True)
df_node = pd.read_csv(os.path.join(base_path, 'toyact_gen/input/node.csv'), index_col= 0)
df_node = df_node.iloc[:, 1:] # nodeidの列を削除
# node_features = torch.load(os.path.join(base_path, 'toy_data_generation/node_features_matrix.pt'), weights_only=True)
node_features_np = df_node.to_numpy()
node_features = torch.tensor(node_features_np, dtype=torch.float32)


df_diary = pd.read_csv(os.path.join(base_path, 'toyact_gen/output/route_traj.csv'), index_col = 0) # 仮想経路データ
df_loc_arr = pd.read_csv(os.path.join(base_path, 'toyact_gen/output/state_traj.csv'), index_col = 0) # 仮想経路データ
df_act_traj = pd.read_csv(os.path.join(base_path, 'toyact_gen/output/act_traj.csv'), index_col = 0) # 仮想経路データ

diary_arr = df_diary.to_numpy()
loc_arr = df_loc_arr.to_numpy()
act_arr = df_act_traj.to_numpy()

# print('diary_arr', diary_arr) # 64*20
# print('loc_arr', loc_arr) # 64*20
# print('act_arr', act_arr) # 64*20
# PyTorch Tensor を NumPy に変換
# adj_matrix_np = adj_matrix.numpy()
node_features_np = node_features.numpy()

#シミュレーション用のデータセット
# trip_arrz = np.load('/mnt/okinawa/9月BLEデータ/route_input/reduced_route_input_0928_all.npz')
# trip_arr = trip_arrz['route_arr']
# time_arr = trip_arrz['time_arr'] # 多分不要

#教師データを保存しておく
loc_teacher_df = pd.DataFrame(loc_arr)
act_teacher_df = pd.DataFrame(act_arr)
loc_teacher_df.to_csv(os.path.join(base_path, 'toyact_gen/output/loc_teacher.csv'))
act_teacher_df.to_csv(os.path.join(base_path, 'toyact_gen/output/act_teacher.csv'))
# 時刻に応じたVAE入力データをロード（例として抜粋） # ここをモバ空に直せば良い
# start_time = datetime(2024, 9, 28, 10, 0, 0)
# end_time = datetime(2024, 9, 28, 15, 0, 0)
# current_time = start_time
# time_lis = []
# while current_time < end_time:
#     time_str = current_time.strftime("%Y%m%d%H")
#     time_lis.append(int(time_str))
#     current_time += timedelta(hours=1)

# start_time = datetime(2024, 9, 28, 18, 0, 0)
# end_time = datetime(2024, 9, 29, 2, 0, 0)
# current_time = start_time
# while current_time < end_time:
#     time_str = current_time.strftime("%Y%m%d%H")
#     time_lis.append(int(time_str))
#     current_time += timedelta(hours=1)
# print(time_lis)

# img_dic = {int(time * 100): torch.load(f"/mnt/okinawa/camera/VAE_input_1to1/{time}.pt") for time in time_lis}

#前処理
timestep = len(act_arr[0])
print('length of timestep: ', timestep)
network = Network(node_features) # Networkクラスから隣接行列は除いている

loc_data = torch.from_numpy(loc_arr)
act_data = torch.from_numpy(act_arr)
A = 4 # activity数
N = 4 # node数
# print('no node', network.N)
SA = A * N # state*activityの同時分布数
# route = torch.from_numpy(trip_arr)
# time_tensor = torch.from_numpy(time_arr)
act_vocab_size = A + 4 # network.N + 4 # node数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
loc_vocab_size = N + 4 # network.N + 4 # node数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
feature_dim = network.node_features.shape[1] # 特徴量の次元数 
batch_size = 64

#trip_arrをdiscontinuous tokenに変換
#tokenizer = Tokenization(network)
#input_discontinuous = tokenizer.tokenization(trip_arr, mode = "discontinuous").long().to(device)

tokenizer = Tokenization(network, A, N)


class MultiModalDataset(Dataset):
    def __init__(self, act_data, loc_data):
        """
        act_data :torch.Tensor or np.ndarray, shape = [N, seq_len]
        loc_data : 同上
        """
        # print('act_data type: ', type(act_data)) # tensor
        # print('loc_data type: ', type(loc_data))
        # 一旦torch.Tensorに変換しておくと後段が楽
        if not isinstance(act_data, torch.Tensor):
            act_data = torch.tensor(act_data, dtype=torch.long)
        if not isinstance(loc_data, torch.Tensor):
            loc_data = torch.tensor(loc_data, dtype=torch.long)
        
        self.act_data = act_data
        self.loc_data = loc_data

        # print('act data type: ', type(self.act_data)) # tensor
        # print('loc data type: ', type(self.loc_data))
        
        # 念のため長さが全部同じかチェック
        assert self.act_data.shape[0] == self.loc_data.shape[0], \
            "act and loc must have the same number of samples"
        # seq_lenは自由にしてOK

        # print('act data type: ', type(self.act_data)) # ここまでちゃんとtensor
        # print('loc data type: ', type(self.loc_data))

    def __len__(self):
        return self.act_data.shape[0]

    def __getitem__(self, idx): # 結局ここは同じ
        return self.act_data[idx], self.loc_data[idx]


# バッチ化
# dataset = MyDataset(route)
dataset = MultiModalDataset(act_data, loc_data) # classのインスタンス化: initしか実行されない

test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

########################################
# 3. モデルの作成・読み込み
########################################

#ハイパーパラメータの取得
# api = wandb.Api()
# run = api.run("tkwnmdr-utokyo/RoutesFormer_test/7sxark22")
# print(f"Run ID: {run.id}, Run Name: {run.name}")

# config = run.config
# #RoutesFormerのハイパーパラメータ
# l_max = config['l_max'] #シークエンスの最大長さ
# B_en = config['B_en'] #エンコーダのブロック数
# B_de = config['B_de'] #デコーダのブロック数
# head_num = config['head_num'] #ヘッド数
# d_ie = config['d_ie'] #トークンの埋め込み次元数
# d_fe = config['d_fe'] #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
# d_ff = config['d_ff'] #フィードフォワード次元数
# z_dim = config['z_dim'] #潜在変数の次元数
# l_max = 62

#RoutesFormerのハイパーパラメータ
l_max = timestep+2 # wandb.config.l_max #シークエンスの最大長さ # 開始・終了＋経路長
B_en = 2 # wandb.config.B_en #エンコーダのブロック数 # 元論文より
B_de = 2 # wandb.config.B_de #デコーダのブロック数 # 元論文より
head_num = 2 # wandb.config.head_num #ヘッド数　＃基本的にヘッド数は変えない，4の倍数にする，マルチヘッドの部分
# d_ie = 22 # wandb.config.d_ie #トークンの埋め込み次元数
d_fe = 4 # wandb.config.d_fe #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
d_ff = 32 # wandb.config.d_ff #フィードフォワード次元数
eos_weight = 3.0 # wandb.config.eos_weight #EOSトークンの重み
savefilename = "toyact.pth" # wandb.config.savefilename #モデルの保存ファイル名
stay_weight = 1 # wandb.config.stay_weight

#モデル
# model = Actformer(enc_vocab_size= vocab_size,
#                             dec_vocab_size = vocab_size,
#                             token_emb_dim = d_ie,
#                             feature_dim = feature_dim, # + z_dim, # 
#                             feature_emb_dim = d_fe, # 2
#                             d_ff = d_ff,
#                             head_num = head_num,
#                             B_en = B_en,
#                             B_de = B_de).to(device)

model = Actformer( # インスタンス生成→以降modelで呼び出すとforward関数が呼ばれる
                    # enc_vocab_size = vocab_size,
                    # dec_vocab_size = vocab_size,
                    # token_emb_dim = d_ie,
                    # time_vocab_size = timestep, # どれくらいの時間数があるか
                    loc_vocab_size = N, # どれくらいの場所数があるか
                    act_vocab_size = A, 
                    
                    # token_dim = 32,
                    # time_emb_dim = 16, 
                    loc_emb_dim = 8, 
                    act_emb_dim = 8,

                    feature_dim = feature_dim,
                    feature_emb_dim = d_fe,
                    d_ff = d_ff,
                    head_num = head_num,
                    B_en = B_en,
                    B_de = B_de).to(device)

# print('modelは回ってる')
model_weights_path = os.path.join(base_path, 'RoutesFormer/output', savefilename) # 学習済みmodelのアウトプット
loadfile = torch.load(model_weights_path)
model.load_state_dict(loadfile['model_state_dict']) 
model.eval()
tokenizer = Tokenization(network, A, N)
# ignore_value_list = [tokenizer.SPECIAL_TOKENS["<p>"], tokenizer.SPECIAL_TOKENS["<m>"]]
# neighbor = Neighbor(adj_matrix)

########################################
# 4. 推論用の関数を分割して定義
########################################

def generate_next_zone_logits(model, 
                              act_batch, loc_batch, disc_feats, 
                              act_traveled_route, loc_traveled_route, traveled_feats):
    """
    モデルから次に出力するトークンのlogitsを取り出す関数。
    """
    act_output, loc_output = model(act_batch, loc_batch, disc_feats,
                    act_traveled_route, loc_traveled_route, traveled_feats)
    # print(f'act_output.shape:{act_output.shape}, loc_output.shape:{loc_output.shape}') # 64*1*8, 64*2*8, と増えてはいる->だんだん後ろにくっつけることはできている　# 基本みんな同じ長さを出力するはず
    # print(f'act_batch.shape:{act_batch.shape}, loc_batch.shape:{loc_batch.shape}')
    return act_output[:, -1, :], loc_output[:, -1, :]   # sequenceの最後のステップの出力のみ返す # 8はemb-dimなので保存

def apply_neighbor_mask(logits, neighbor, newest_zone, d_tensor):
    """
    neighbor マスクを生成して logits に加算する。
    """
    neighbor_mask = neighbor.make_neighbor_mask(newest_zone, d_tensor).to(device)
    # マスクを加算
    masked_logits = logits + neighbor_mask
    return masked_logits

def sample_next_zone(masked_logits):
    """
    softmaxしてトークンをサンプリングする（multinomial）。
    事前にNaN対策などを行う。
    """
    # 数値安定化
    masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
    output_softmax = F.softmax(masked_logits, dim=-1)
    next_zone = torch.multinomial(output_softmax, num_samples=1).squeeze(-1)
    #next_zone = torch.argmax(output_softmax, dim=-1)
    return next_zone

# これまで通った経路＋次のゾーン→結合したい
def update_traveled_route(tokenizer, act_traveled_route, loc_traveled_route, act_next_zone, loc_next_zone): #, time_batch, img_dic, time_is_day):
    """
    traveled_route に next_zone を追加し、特徴行列 (features) も更新する。
    """
    l1  = act_traveled_route.shape
    act_traveled_route = torch.cat([act_traveled_route, act_next_zone.unsqueeze(1)], dim=1)
    loc_traveled_route = torch.cat([loc_traveled_route, loc_next_zone.unsqueeze(1)], dim=1)
    # traveled_route = torch.cat([traveled_route, next_zone.unsqueeze(1)], dim=1)
    traveled_feature_mat = tokenizer.make_feature_mat(loc_traveled_route).to(device)
    l2 = act_traveled_route.shape
    # print('l1, l2', l1, l2) ## 
    # time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(
    #     traveled_feature_mat.shape[0], traveled_feature_mat.shape[1], 1
    # ).to(device)
    # traveled_VAE_mat = tokenizer.make_VAE_input(traveled_route, time_batch, img_dic).to(device)
    # traveled_feature_mat = torch.cat((traveled_feature_mat), dim = 2) #, time_feature_mat, traveled_VAE_mat), dim=2)
    return act_traveled_route, loc_traveled_route, traveled_feature_mat

def check_and_save_completed_routes(
    act_traveled_route, loc_traveled_route, traveled_feats, l_max, tokenizer, idx_lis,
    # time_is_day,
    # d_tensor,
    act_d_tensor, loc_d_tensor,
    act_infer_start_indices, loc_infer_start_indices, act_batch, loc_batch, disc_feats, act_save_route, loc_save_route
    # infer_start_indices, disc_tokens, disc_feats, save_route
    ):
    """
    最新トークンが <e> のものを最終的な出力として保存し、バッチから除去して返す
    """
    # 終了トークンが出たサンプルを取得
    act_true_indices = torch.where(act_traveled_route[:, -1] == tokenizer.act_SPECIAL_TOKENS["<e>"])[0]
    loc_true_indices = torch.where(loc_traveled_route[:, -1] == tokenizer.loc_SPECIAL_TOKENS["<e>"])[0]

    # actとlocの両方が終了しているサンプルのみを対象にする
    common_true_indices = torch.tensor(
        list(set(act_true_indices.tolist()) & set(loc_true_indices.tolist())),
        dtype=torch.long,
        device=act_traveled_route.device
    )

    # print(f'act_true_indices:{act_true_indices}, loc_true_indices:{loc_true_indices}, common: {list(set(act_true_indices.tolist()) & set(loc_true_indices.tolist()))}')
    # print(f'common_true_indices:{common_true_indices}')
    # if len(common_true_indices) <= 0:
    #     print("actの終了トークンが出ていない")

    # if len(act_true_indices) > 0: 
    if len(common_true_indices) > 0:

        # 完了したルートを save_route に書き込み（パディングしてから保存） ### むずい．．．
        # act_padded_routes = torch.nn.functional.pad(
        #     act_traveled_route[act_true_indices],
        #     (0, l_max - act_traveled_route.size(1)),
        #     value=tokenizer.act_SPECIAL_TOKENS["<p>"]
        # )
        # loc_padded_routes = torch.nn.functional.pad(
        #     loc_traveled_route[loc_true_indices],
        #     (0, l_max - loc_traveled_route.size(1)),
        #     value=tokenizer.loc_SPECIAL_TOKENS["<p>"]
        # )

        # 完了したルートを save_route に書き込み（パディングしてから保存）
        print('l_max - act_traveled_route.size(1)', l_max - act_traveled_route.size(1))
        act_padded_routes = torch.nn.functional.pad( # act_traveled_route: size 1 から始まる 19から始まる # (left, right)で左側にパディングする数，右側にパディングする数
            act_traveled_route[common_true_indices], (0, l_max - act_traveled_route.size(1)-1), value=tokenizer.act_SPECIAL_TOKENS["<p>"]
        ) # 長さは19 
        loc_padded_routes = torch.nn.functional.pad(
            loc_traveled_route[common_true_indices], (0, l_max - loc_traveled_route.size(1)-1), value=tokenizer.loc_SPECIAL_TOKENS["<p>"]
        )

        # save_route の該当箇所に書き込む
        # idx = idx_lis[true_indices] 
        actloc_idx = idx_lis[common_true_indices]
        # loc_idx = loc_idx_lis[common_true_indices]
        print('act_padded_routes', act_padded_routes.shape) # 64*20
        act_save_route[actloc_idx] = act_padded_routes
        loc_save_route[actloc_idx] = loc_padded_routes
        # save_route[idx] = padded_routes

        # # 完了したサンプルをバッチから除外
        # act_mask_del = torch.ones(act_traveled_route.size(0), dtype=torch.bool, device=device)
        # loc_mask_del = torch.ones(loc_traveled_route.size(0), dtype=torch.bool, device=device)
        # # mask_del = torch.ones(traveled_route.size(0), dtype=torch.bool, device=device)
        # act_mask_del[act_true_indices] = False
        # loc_mask_del[loc_true_indices] = False
        # # mask_del[true_indices] = False


        # 各種インデックスで同期して除去 # 代わりにこっち
        mask_del = torch.ones(act_traveled_route.size(0), dtype=torch.bool, device=device)
        mask_del[common_true_indices] = False # 終了トークンが出ているものがfalse
        # 全部 mask_del で除去
        act_traveled_route = act_traveled_route[mask_del]
        loc_traveled_route = loc_traveled_route[mask_del]

        # idx_lis = idx_lis[mask_del]
        idx_lis = idx_lis[mask_del]
        # loc_idx_lis = loc_idx_lis[mask_del]
        # time_is_day = time_is_day[mask_del]
        # d_tensor = d_tensor[mask_del]
        act_d_tensor = act_d_tensor[mask_del]
        loc_d_tensor = loc_d_tensor[mask_del]

        # infer_start_indices = infer_start_indices[mask_del]
        act_infer_start_indices = act_infer_start_indices[mask_del]
        loc_infer_start_indices = loc_infer_start_indices[mask_del]
        
        # disc_tokens = disc_tokens[mask_del]
        act_batch = act_batch[mask_del]
        loc_batch = loc_batch[mask_del]
        disc_feats = disc_feats[mask_del]

        # traveled_route = traveled_route[mask_del]
        # act_traveled_route = act_traveled_route[act_mask_del]
        # loc_traveled_route = loc_traveled_route[loc_mask_del]
        traveled_feats = traveled_feats[mask_del]
    
    return (
        act_traveled_route,
        loc_traveled_route,
        traveled_feats,
        # act_idx_lis,
        idx_lis,
        # idx_lis,
        # 
        # time_is_day,
        # d_tensor,
        act_d_tensor,
        loc_d_tensor,
        act_infer_start_indices,
        loc_infer_start_indices,
        # infer_start_indices,
        act_batch,
        loc_batch,
        # disc_tokens,
        disc_feats,
        # save_route
        act_save_route,
        loc_save_route
    )

########################################
# 5. 推論の実行（メイン部分）
########################################

def run_inference(test_loader, model, tokenizer, 
                # neighbor, #### これ使わない
                #  img_dic, 
                # l_max=62 # なぜ62???
                l_max
                ):
    """
    実際にtest_loaderからバッチを読み出し、推論を行うメイン関数。
    """
    # ignore_value_list = [tokenizer.SPECIAL_TOKENS["<p>"], tokenizer.SPECIAL_TOKENS["<m>"]]
    act_all_results = []
    loc_all_results = []

    # for batch_idx, (disc_tokens) in enumerate(test_loader):
    for act_batch, loc_batch in test_loader:

        # disc_tokens = disc_tokens.to(device)
        # time_batch = time_batch.to(device)
        act_batch = act_batch.to(device)
        loc_batch = loc_batch.to(device)

        # print('len of batch: ', act_batch.shape, loc_batch.shape) # 64*8->減ってく（batch sizeが減っていく）

        # 昼/夜フラグ (例: ある時刻より小さければ昼)
        # time_is_day = (time_batch < 202409281500).to(device)
        batch_size = act_batch.shape[0]
        # batch_size = disc_tokens.shape[0]

        # 終了トークン <e> の直前のトークン（=目的地）を d_tensor として取得
        # is_end_tokens = (disc_tokens == tokenizer.SPECIAL_TOKENS["<e>"])
        act_is_end_tokens = (act_batch == tokenizer.act_SPECIAL_TOKENS["<e>"])
        loc_is_end_tokens = (loc_batch == tokenizer.loc_SPECIAL_TOKENS["<e>"])

        # indices = is_end_tokens.float().argmax(dim=1)
        act_indices = act_is_end_tokens.float().argmax(dim=1)
        loc_indices = loc_is_end_tokens.float().argmax(dim=1)

        # d_tensor = disc_tokens[torch.arange(disc_tokens.size(0)), indices - 1]
        # 目的地＝家のはず # activity path においては特に重要ではない
        act_d_tensor = act_batch[torch.arange(act_batch.size(0)), act_indices - 1]
        loc_d_tensor = loc_batch[torch.arange(loc_batch.size(0)), loc_indices - 1]

        # 開始トークン <b> の位置を見て、推論開始インデックスを求める
        # is_begin_tokens = (disc_tokens == tokenizer.SPECIAL_TOKENS["<b>"])
        act_is_begin_tokens = (act_batch == tokenizer.act_SPECIAL_TOKENS["<b>"])
        loc_is_begin_tokens = (loc_batch == tokenizer.loc_SPECIAL_TOKENS["<b>"])

        # begin_indices = is_begin_tokens.float().argmax(dim=1)
        act_begin_indices = act_is_begin_tokens.float().argmax(dim=1)
        loc_begin_indices = loc_is_begin_tokens.float().argmax(dim=1)

        # infer_start_indices = begin_indices + 1
        act_infer_start_indices = act_begin_indices + 1
        loc_infer_start_indices = loc_begin_indices + 1

        # discontinuous_feature_matを作る # loc dataから
        disc_feature_mat = tokenizer.make_feature_mat(loc_batch).to(device)
        # time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(
        #     disc_feature_mat.shape[0], disc_feature_mat.shape[1], 1
        # ).to(device)
        # disc_VAE_mat = tokenizer.make_VAE_input(disc_tokens, time_batch, img_dic).to(device)
        # disc_feature_mat = torch.cat((disc_feature_mat, time_feature_mat, disc_VAE_mat), dim=2)

        # 推論中のルートを <p> だけ入った状態で初期化 ### for what purpose???
        act_traveled_route = torch.full( # 64*1
            (batch_size, 1),
            tokenizer.act_SPECIAL_TOKENS["<p>"],
            dtype=torch.long
        ).to(device)
        loc_traveled_route = torch.full(
            (batch_size, 1),
            tokenizer.loc_SPECIAL_TOKENS["<p>"],
            dtype=torch.long
        ).to(device)
        print(f'jyoban::: act_traveled_route shape:{act_traveled_route.shape},')#  loc_traveled_route:{loc_traveled_route}')
        traveled_feats = tokenizer.make_feature_mat(loc_traveled_route).to(device) # paddingに対応した特徴量ベクトルになる
        # time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(
        #     traveled_feats.shape[0], traveled_feats.shape[1], 1
        # ).to(device)
        # traveled_VAE_mat = tokenizer.make_VAE_input(traveled_route, time_batch, img_dic).to(device)
        # traveled_feats = torch.cat((traveled_feats, time_feature_mat, traveled_VAE_mat), dim=2)

        # 出力を保存する変数 (l_maxに合わせたサイズに最終的に揃える) # 64*18なので元のバッチの形状と同じ
        act_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device) # 64*20になる
        loc_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device)

        # バッチ内サンプルのインデックス管理 ####### 揃ってないといけない ########
        idx_lis = torch.arange(batch_size, device=device)
        # act_idx_lis = torch.arange(batch_size, device=device)
        # loc_idx_lis = torch.arange(batch_size, device=device)

        i = 0
        # print('l_max', l_max)ddd
        while i <= l_max - 3: # timestep+2(前後トークンのはず)→変？
            # モデル推論 # act_traveled_routeは1始まり
            act_next_zone_logits, loc_next_zone_logits = generate_next_zone_logits(
                                                                                    model, 
                                                                                    act_batch, loc_batch, disc_feature_mat, 
                                                                                    act_traveled_route, loc_traveled_route, traveled_feats
                                                                                )
            ## 残存バッチ数（64から単調減少） * 埋め込み次元数８
            # print('act_next_zone_logi', act_next_zone_logits.shape, 'loc_next_zone_logi', loc_next_zone_logits.shape)
            
            # neighborマスクを作成し、logitsに加算
            # act_newest_zone = act_traveled_route[:, -1]
            # loc_newest_zone = loc_traveled_route[:, -1]
            # masked_logits = apply_neighbor_mask(next_zone_logits, neighbor, newest_zone, d_tensor)
            
            #### 終了トークンの生成確率を0にする ####
            # ★ ここで <e> トークンのスコアを下げる
            e_act = tokenizer.act_SPECIAL_TOKENS["<e>"]
            e_loc = tokenizer.loc_SPECIAL_TOKENS["<e>"]
            b_act = tokenizer.act_SPECIAL_TOKENS["<b>"]
            b_loc = tokenizer.loc_SPECIAL_TOKENS["<b>"]
            p_act = tokenizer.act_SPECIAL_TOKENS["<p>"]
            p_loc = tokenizer.loc_SPECIAL_TOKENS["<p>"]
            m_act = tokenizer.act_SPECIAL_TOKENS["<m>"]
            m_loc = tokenizer.loc_SPECIAL_TOKENS["<m>"]

            act_next_zone_logits[:, e_act] = float('-inf')
            loc_next_zone_logits[:, e_loc] = float('-inf')
            act_next_zone_logits[:, b_act] = float('-inf')
            loc_next_zone_logits[:, b_loc] = float('-inf')
            act_next_zone_logits[:, p_act] = float('-inf')
            loc_next_zone_logits[:, p_loc] = float('-inf')
            act_next_zone_logits[:, m_act] = float('-inf')
            loc_next_zone_logits[:, m_loc] = float('-inf')
            

            # softmax + サンプリング
            # next_zone = sample_next_zone(masked_logits)
            act_next_zone = sample_next_zone(act_next_zone_logits) # softmaxする
            loc_next_zone = sample_next_zone(loc_next_zone_logits)


            # print('output of softmax', act_next_zone.shape, loc_next_zone.shape) # 1batchにつき1個ずつ

            # まだルートが開始していない（つまり推論のステップより小さい）場合は既知のトークンを代入
            # not_start = infer_start_indices >= i
            act_not_start = act_infer_start_indices >= i
            loc_not_start = loc_infer_start_indices >= i
            
            ## 次のトークンを決定（推論）
            # print(f'act_not_start:{act_not_start}, loc_not_start:{loc_not_start}')
            # print('imano i ha', i)
            act_next_zone[act_not_start] = act_batch[act_not_start, i].long()
            loc_next_zone[loc_not_start] = loc_batch[loc_not_start, i].long()
            # next_zone[not_start] = disc_tokens[not_start, i]
            # print('act_next_batch: ', act_next_zone, 'loc_next_zone:', loc_next_zone)
            # print('act_batch[act_not_start, i].long()', act_batch[act_not_start, i].long())
            # print('loc_batch[loc_not_start, i].long()', loc_batch[loc_not_start, i].long())

            # traveled_route の更新
            # 次のsoftmax最大のトークンを用いてupdate
            ##### ここでact_traveled routeが1長くなる #####
            l1 = act_traveled_route.shape
            act_traveled_route, loc_traveled_route, traveled_feats = update_traveled_route(
                tokenizer, act_traveled_route, loc_traveled_route, act_next_zone, loc_next_zone
                # , time_batch, img_dic, time_is_day
            )
            l2 = act_traveled_route.shape

            # print('l1, l2', l1, l2) # 64*18, 64*19
            # 終了トークン <e> を出したサンプルを確認して保存＆削除
            (
             act_traveled_route,
             loc_traveled_route,
             traveled_feats,
             idx_lis,
             # loc_idx_lis,
             #time_is_day,
             # d_tensor,
             act_d_tensor,
             loc_d_tensor,
             # infer_start_indices,
             act_infer_start_indices,
             loc_infer_start_indices,
            #  disc_tokens,
             act_batch,
             loc_batch, 
             disc_feature_mat,
             act_save_route,
             loc_save_route
             ) = check_and_save_completed_routes(
                act_traveled_route, loc_traveled_route, traveled_feats, l_max, tokenizer, idx_lis, # loc_idx_lis,
                # time_is_day, 
                act_d_tensor, loc_d_tensor,
                act_infer_start_indices, loc_infer_start_indices, act_batch, loc_batch, disc_feature_mat, act_save_route, loc_save_route
            )
            # print(f"traveled_route:{len(act_traveled_route)}, {len(loc_traveled_route)}, traveled_feats:{len(traveled_feats)}") # 最初はバッチ数（バッチサイズ64のミニバッチ学習するので）→減っていく
            # print(i)

            # バッチ内サンプルがなくなったら終了
            if act_traveled_route.size(0) == 0:
                print("All samples in this batch have finished.", loc_traveled_route.size(0), 'これも0のはず')
                break

            i += 1

            # save routeになる？？？

            #### inferのループももっとちゃんと理解する必要がある．．．．

        # ループを抜けた後、まだ完了していないものはそのまま保存に使う
        print('whlie loop out')
        print(f'act_traveled_route.shape:{act_traveled_route.shape}, loc_traveled_route.shape:{loc_traveled_route.shape}')
        print(f'act_save_route.shape:{act_save_route.shape}, loc_save_route.shape:{loc_save_route.shape}')
        
        # act, locで場所を揃えて完成ルートを入れる
        act_save_route[idx_lis] = act_traveled_route 
        loc_save_route[idx_lis] = loc_traveled_route # loc_save_route: 64*l_max=20

        act_all_results.append(act_save_route)
        loc_all_results.append(loc_save_route)

        

    # 結果をまとめて返す
    act_final_result = torch.cat(act_all_results, dim=0)
    loc_final_result = torch.cat(loc_all_results, dim=0)

    return act_final_result, loc_final_result


########################################
# 6. 実際に推論を実行して結果を保存
########################################

# 推論実行
print('run temae') # kokomade ha kita
act_result, loc_result = run_inference(test_loader, model, tokenizer,
                        # neighbor,
                        # img_dic, 
                        l_max=l_max)

# CSVへ保存
act_result_df = pd.DataFrame(act_result.cpu().numpy())
loc_result_df = pd.DataFrame(loc_result.cpu().numpy())
act_result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/act_inference_result.csv')
loc_result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/loc_inference_result.csv')
# result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/inference_result.csv')
print("推論結果を保存しました！")