import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F

from ActFormer.RoutesFormer.simple_routeformer.toysrc.network import Network
from ActFormer.RoutesFormer.simple_routeformer.toysrc.tokenization import Tokenization
from ActFormer.RoutesFormer.simple_routeformer.toysrc.routesformer import Routesformer
from utils.logger import logger
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, random_split, DataLoader
from data_generation.Recursive import TimeFreeRecursiveLogit

torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

class Neighbor:
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
adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)
node_features = torch.load("/mnt/okinawa/9月BLEデータ/route_input/network/node_features_matrix.pt", weights_only=True)

#シミュレーション用のデータセット
trip_arrz = np.load('/mnt/okinawa/9月BLEデータ/route_input/reduced_route_input_0928_all.npz')
trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']

#教師データを保存しておく
teacher_df = pd.DataFrame(trip_arr)
teacher_df.to_csv("/mnt/okinawa/9月BLEデータ/route_output/teacher.csv")


#前処理
timestep = len(trip_arr[0])
print(timestep)
network = Network(adj_matrix, node_features)
route = torch.from_numpy(trip_arr)
time_tensor = torch.from_numpy(time_arr)
vocab_size = network.N + 4
feature_dim = network.node_features.shape[1] + 1

tokenizer = Tokenization(network)
#バッチの作成
class MyDataset(Dataset):
    def __init__(self, data, time_data):
        self.data = data
        self.time_data = time_data
        self.discontinuous = tokenizer.tokenization(data, mode = "discontinuous").long().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            return self.discontinuous[idx], self.time_data[idx]
        except Exception as e:
            print(f"Error in __getitem__: {e}, idx: {idx}")
            raise

dataset =  MyDataset(route, time_tensor)
test_loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, drop_last=False)

########################################
# 3. モデルの作成・読み込み
########################################

#ハイパーパラメータの取得
api = wandb.Api()
run = api.run("tkwnmdr-utokyo/RoutesFormer_test/yngqjjsg")
print(f"Run ID: {run.id}, Run Name: {run.name}")

config = run.config
#RoutesFormerのハイパーパラメータ
l_max = config['l_max'] #シークエンスの最大長さ
B_en = config['B_en'] #エンコーダのブロック数
B_de = config['B_de'] #デコーダのブロック数
head_num = config['head_num'] #ヘッド数
d_ie = config['d_ie'] #トークンの埋め込み次元数
d_fe = config['d_fe'] #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
d_ff = config['d_ff'] #フィードフォワード次元数
l_max = 62


#モデル
model = Routesformer(enc_vocab_size= vocab_size,
                            dec_vocab_size = vocab_size,
                            token_emb_dim = d_ie,
                            feature_dim = feature_dim,
                            feature_emb_dim = d_fe,
                            d_ff = d_ff,
                            head_num = head_num,
                            B_en = B_en,
                            B_de = B_de).to(device)

model_weights_path = "model_weights_all_2.pth"
loadfile = torch.load(model_weights_path)
model.load_state_dict(loadfile['model_state_dict'])
model.eval()

tokenizer = Tokenization(network)
ignore_value_list = [tokenizer.SPECIAL_TOKENS["<p>"], tokenizer.SPECIAL_TOKENS["<m>"]]
neighbor = Neighbor(adj_matrix)

########################################
# 4. 推論用の関数を分割して定義
########################################

def generate_next_zone_logits(model, disc_tokens, disc_feats, traveled_route, traveled_feats):
    """
    モデルから次に出力するトークンのlogitsを取り出す関数。
    """
    outputs = model(disc_tokens, disc_feats, traveled_route, traveled_feats)
    return outputs[:, -1, :]  # sequenceの最後のステップの出力のみ返す

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

def update_traveled_route(tokenizer, traveled_route, next_zone, time_is_day):
    """
    traveled_route に next_zone を追加し、特徴行列 (features) も更新する。
    """
    traveled_route = torch.cat([traveled_route, next_zone.unsqueeze(1)], dim=1)
    traveled_feature_mat = tokenizer.make_feature_mat(traveled_route).to(device)
    time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(
        traveled_feature_mat.shape[0], traveled_feature_mat.shape[1], 1
    ).to(device)
    traveled_feature_mat = torch.cat((traveled_feature_mat, time_feature_mat), dim=2)
    return traveled_route, traveled_feature_mat

def check_and_save_completed_routes(
    traveled_route, traveled_feats, l_max, tokenizer, idx_lis, time_is_day, d_tensor,
    infer_start_indices, disc_tokens, disc_feats, save_route
):
    """
    最新トークンが <e> のものを最終的な出力として保存し、バッチから除去して返す。
    """
    # 終了トークンが出たサンプルを取得
    true_indices = torch.where(traveled_route[:, -1] == tokenizer.SPECIAL_TOKENS["<e>"])[0]
    print(f'true_indices:{len(true_indices)}')
    if len(true_indices) > 0:
        # 完了したルートを save_route に書き込み（パディングしてから保存）
        padded_routes = torch.nn.functional.pad(
            traveled_route[true_indices],
            (0, l_max - traveled_route.size(1)),
            value=tokenizer.SPECIAL_TOKENS["<p>"]
        )
        # save_route の該当箇所に書き込む
        idx = idx_lis[true_indices]
        save_route[idx] = padded_routes

        # 完了したサンプルをバッチから除外
        mask_del = torch.ones(traveled_route.size(0), dtype=torch.bool, device=device)
        mask_del[true_indices] = False
        
        idx_lis = idx_lis[mask_del]
        time_is_day = time_is_day[mask_del]
        d_tensor = d_tensor[mask_del]
        infer_start_indices = infer_start_indices[mask_del]
        disc_tokens = disc_tokens[mask_del]
        disc_feats = disc_feats[mask_del]
        traveled_route = traveled_route[mask_del]
        traveled_feats = traveled_feats[mask_del]
    return (
        traveled_route,
        traveled_feats,
        idx_lis,
        time_is_day,
        d_tensor,
        infer_start_indices,
        disc_tokens,
        disc_feats,
        save_route
    )

########################################
# 5. 推論の実行（メイン部分）
########################################

def run_inference(test_loader, model, tokenizer, neighbor, l_max=62):
    """
    実際にtest_loaderからバッチを読み出し、推論を行うメイン関数。
    """
    ignore_value_list = [tokenizer.SPECIAL_TOKENS["<p>"], tokenizer.SPECIAL_TOKENS["<m>"]]
    all_results = []

    for batch_idx, (disc_tokens, time_batch) in enumerate(test_loader):
        print(f"Batch {batch_idx + 1} / {len(test_loader)}")
        disc_tokens = disc_tokens.to(device)
        time_batch = time_batch.to(device)

        # 昼/夜フラグ (例: ある時刻より小さければ昼)
        time_is_day = (time_batch < 202409281500).to(device)

        batch_size = disc_tokens.shape[0]

        # 終了トークン <e> の直前のトークン（=目的地）を d_tensor として取得
        is_end_tokens = (disc_tokens == tokenizer.SPECIAL_TOKENS["<e>"])
        indices = is_end_tokens.float().argmax(dim=1)
        d_tensor = disc_tokens[torch.arange(disc_tokens.size(0)), indices - 1]

        # 開始トークン <b> の位置を見て、推論開始インデックスを求める
        is_begin_tokens = (disc_tokens == tokenizer.SPECIAL_TOKENS["<b>"])
        begin_indices = is_begin_tokens.float().argmax(dim=1)
        infer_start_indices = begin_indices + 1

        # discontinuous_feature_matを作る
        disc_feature_mat = tokenizer.make_feature_mat(disc_tokens).to(device)
        time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(
            disc_feature_mat.shape[0], disc_feature_mat.shape[1], 1
        ).to(device)
        disc_feature_mat = torch.cat((disc_feature_mat, time_feature_mat), dim=2)

        # 推論中のルートを <p> だけ入った状態で初期化
        traveled_route = torch.full(
            (batch_size, 1),
            tokenizer.SPECIAL_TOKENS["<p>"],
            dtype=torch.long
        ).to(device)
        traveled_feats = tokenizer.make_feature_mat(traveled_route).to(device)
        time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(
            traveled_feats.shape[0], traveled_feats.shape[1], 1
        ).to(device)
        traveled_feats = torch.cat((traveled_feats, time_feature_mat), dim=2)

        # 出力を保存する変数 (l_maxに合わせたサイズに最終的に揃える)
        save_route = torch.zeros((batch_size, l_max), dtype=torch.long).to(device)

        # バッチ内サンプルのインデックス管理
        idx_lis = torch.arange(batch_size, device=device)

        i = 1
        while i <= l_max - 1:
            # モデル推論
            next_zone_logits = generate_next_zone_logits(
                model, disc_tokens, disc_feature_mat, traveled_route, traveled_feats
            )

            # neighborマスクを作成し、logitsに加算
            newest_zone = traveled_route[:, -1]
            masked_logits = apply_neighbor_mask(next_zone_logits, neighbor, newest_zone, d_tensor)

            # softmax + サンプリング
            next_zone = sample_next_zone(masked_logits)

            # まだルートが開始していない（つまり推論のステップより小さい）場合は既知のトークンを代入
            not_start = infer_start_indices >= i
            next_zone[not_start] = disc_tokens[not_start, i]

            # traveled_route の更新
            traveled_route, traveled_feats = update_traveled_route(
                tokenizer, traveled_route, next_zone, time_is_day
            )

            # 終了トークン <e> を出したサンプルを確認して保存＆削除
            (traveled_route,
             traveled_feats,
             idx_lis,
             time_is_day,
             d_tensor,
             infer_start_indices,
             disc_tokens,
             disc_feature_mat,
             save_route) = check_and_save_completed_routes(
                traveled_route, traveled_feats, l_max, tokenizer, idx_lis, time_is_day, d_tensor,
                infer_start_indices, disc_tokens, disc_feature_mat, save_route
            )
            print(f"traveled_route:{len(traveled_route)}")
            print(i)

            # バッチ内サンプルがなくなったら終了
            if traveled_route.size(0) == 0:
                print("All samples in this batch have finished.")
                break

            i += 1

        # ループを抜けた後、まだ完了していないものはそのまま保存に使う
        save_route[idx_lis] = traveled_route

        all_results.append(save_route)

    # 結果をまとめて返す
    final_result = torch.cat(all_results, dim=0)
    return final_result


########################################
# 6. 実際に推論を実行して結果を保存
########################################

# 推論実行
result = run_inference(test_loader, model, tokenizer, neighbor, l_max=l_max)

# CSVへ保存
result_df = pd.DataFrame(result.cpu().numpy())
result_df.to_csv("/mnt/okinawa/9月BLEデータ/route_output/result_ordinary_1.csv")
print("推論結果を保存しました！")