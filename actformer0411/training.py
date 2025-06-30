# RoutesFormer (Vanilla Transformer * BERT) for Tokyo PT2019
# Orijinal: Kurasawa Master Thesis
# Rewritten: Takahiro Matsunaga for 2025 cpij

import torch
import pandas as pd
import numpy as np
from torch import nn
from network import Network
from tokenization import Tokenization
from actformer import Actformer
# from utils.logger import logger
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, random_split, DataLoader, Subset
import torch.nn.functional as F
import os
os.environ["WANDB_MODE"] = "offline"
import wandb
import json
import pickle
import sys
import networkx as nx
import datetime

nrow = 15000 # nrow行だけ読み込む
wandb.init(
    project="ActFormer0411",
    config={
    "nrow": nrow,
    "learning_rate": 0.001,
    "architecture": "Normal",
    "dataset": "0928_day_night",
    "epochs": 200,
    "batch_size": 256,
    "l_max" : 21,
    "B_en" : 6,
    "B_de" : 6,
    "head_num" : 4,
    "d_ie_time" : 32,
    "d_ie_loc" : 32, 
    "d_ie_act" : 32, 
    "d_fe" : 4, 
    "d_ff" : 32,
    "eos_weight" : 3.0,
    "stay_weight" : 1,
    "mask_rate" : 0.1,
    "savefilename": None,
    "alignment_loss_weight": 0.01,
    "identical_penalty_weight": 1,
    }
)

run_id = wandb.run.id  # or use wandb.run.name
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_nrow" + str(nrow))
savefilename = f"ACT_{timestamp}_{run_id}.pth"
wandb.config.update({"savefilename": savefilename}, allow_val_change=True)

base_path = '/Users/matsunagatakahiro/Desktop/res2025/ActFormer'

k = 1 # 10分割してるので
df_time_arr = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/time_cleaned_{k}.csv'), index_col = 0, nrows=nrow)
df_loc_arr = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/loc_cleaned_{k}.csv'), index_col = 0, nrows=nrow) 
df_act_traj = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/act_cleaned_{k}.csv'), index_col = 0, nrows=nrow) 
df_indivi = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/indivi_cleaned_{k}.csv'), index_col = False, nrows=nrow)
df_indivi.drop(columns=['household_size', 'age', 'work_start_am_pm', 'work_start_hour', 'work_start_minute', 'household_income', 'survey_month', 'survey_day'], inplace=True)
print(df_indivi.columns)
print('nrows', df_indivi.shape[0])

loc_columns = df_loc_arr.columns.tolist() 
time_columns = df_time_arr.columns.tolist() 
act_columns = df_act_traj.columns.tolist() 
indivi_columns = df_indivi.columns.tolist() 

time_sequence = pd.unique(df_time_arr[time_columns].values.ravel())
loc_sequence = pd.unique(df_loc_arr[loc_columns].values.ravel())
act_sequence = pd.unique(df_act_traj[act_columns].values.ravel())

context_sequences_dict = {
    col : pd.unique(df_indivi[col].values.ravel()) 
    for col in df_indivi.columns[1:]
}

time_fromindivi_sequence = pd.unique(df_indivi['work_start_time'].values.ravel())
loc_fromindivi_sequence = pd.unique(df_indivi[['home_city', 'workplace_city']].values.ravel())

time_sequence = [int(float(time)) for time in time_sequence if pd.notna(time) and str(time) != '99.0']
loc_sequence = [int(float(loc)) for loc in loc_sequence if pd.notna(loc) and str(loc) != '99.0']
act_sequence = [int(float(act)) for act in act_sequence if pd.notna(act) and str(act) != '99.0']
context_sequences_dict = {
    col : [int(float(val)) for val in context_sequences_dict[col] if pd.notna(val) and str(val) != '99.0']
    for col in df_indivi.columns[1:]
}
time_fromindivi_sequence = [int(float(time)) for time in time_fromindivi_sequence if pd.notna(time) and str(time) != '99.0']
loc_fromindivi_sequence = [int(float(loc)) for loc in loc_fromindivi_sequence if pd.notna(loc) and str(loc) != '99.0']
total_time_sequence = set(time_sequence + time_fromindivi_sequence)
total_loc_sequence = set(loc_sequence + loc_fromindivi_sequence)

unique_time = sorted(set(total_time_sequence))
unique_locs = sorted(set(total_loc_sequence))
unique_acts = sorted(set(act_sequence))
num_indivi = len(df_indivi.columns) - 1 # 1列目は個人IDなので除外
unique_contexts_dict = {
    col: sorted(set(context_sequences_dict[col]))
    for col in df_indivi.columns[1:]
}

context_categori_nums = [len(df_indivi[col].unique()) for col in df_indivi.columns[1:]] # 1列目は個人IDなので除外
print('<<<<<context_categori_nums>>>>>', context_categori_nums)
context_categori_nums_dict = {df_indivi.columns[i+1]: context_categori_nums[i] + 1 for i in range(len(context_categori_nums))}
context_categori_nums_dict['work_start_time'] = len(unique_time)
context_categori_nums_dict['home_city'] = len(unique_locs)
context_categori_nums_dict['workplace_city'] = len(unique_locs)


#### 各トークンのvocab_size ####
context_vocab_sizes = context_categori_nums_dict.values()
print('ccontext_vocab_sizes', context_vocab_sizes)

# map(tokenid - index)
time2id = {time: i for i, time in enumerate(unique_time)}  # mapping辞書作成

print('time2id', time2id)
loc2id = {loc: i for i, loc in enumerate(unique_locs)}  # mapping辞書作成
print('loc2id', loc2id)
act2id = {1: 0, 2: 0, 
          3: 1, 4: 1, 5: 1, 6: 1, 
          7: 2, 8: 2, 
          9: 3, 10: 3, 
          11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 
          16: 5, 17: 5, 18: 5, 
          19: 6, 
          20: 7, 
          21: 8}
context2id_dict = {
    col: {id: i for i, id in enumerate(unique_contexts_dict[col])}
    for col in df_indivi.columns[1:]
}

# 独立なvalueの数
TT = len(set(time2id.values())) # 時間数 44
Z = len(set(loc2id.values())) #  274
A = len(set(act2id.values())) # nan以外は通常トークンにした
print('TT', TT, 'Z', Z, 'A', A)

# DataFrame → IDに変換(上書き)
df_time_arr = df_time_arr.applymap(lambda x: time2id.get(int(float(x)), TT + 3) if pd.notna(x) and str(x) != '99.0' else TT + 3) ## 未知なら<m>にする
df_loc_arr = df_loc_arr.applymap(lambda x: loc2id.get(int(float(x)), Z + 3) if pd.notna(x) and str(x) != '99.0' else Z + 3)
df_act_traj = df_act_traj.applymap(lambda x: act2id.get(int(float(x)), A + 3) if pd.notna(x) and str(x) != '99.0' else A + 3)

# 辞書使って一気に変換する
for col in df_indivi.columns[1:]:
    if col == 'work_start_time':
        df_indivi[col] = df_indivi[col].apply(lambda x: time2id.get(int(float(x)), TT + 3) if pd.notna(x) and str(x) != '99.0' else TT + 3)
    elif col in ['home_city', 'workplace_city']:
        df_indivi[col] = df_indivi[col].apply(lambda x: loc2id.get(int(float(x)), Z + 3) if pd.notna(x) and str(x) != '99.0' else Z + 3)
    else:
        context2id = context2id_dict[col]
        df_indivi[col] = df_indivi[col].apply(lambda x: context2id.get(int(float(x)), len(set(context2id.values()))) if pd.notna(x) and str(x) != '99.0' else len(set(context2id.values())) + 1) # if pd.notna(x) and str(x) != '99.0' else len(set(context2id.values())) + 3)

time_arr = df_time_arr.to_numpy()
loc_arr = df_loc_arr.to_numpy()
act_arr = df_act_traj.to_numpy()
context_arr = df_indivi.to_numpy()[:, 1:] # 個人特徴量の部分だけ抜き出す

# adj_matrix_np = adj_matrix.numpy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 前処理
time_data = torch.from_numpy(time_arr)
loc_data = torch.from_numpy(loc_arr)
act_data = torch.from_numpy(act_arr)
context_data = torch.from_numpy(context_arr)
# contextが読めているかの確認のために全部０にしてみる．
# context_data = torch.zeros_like(torch.from_numpy(context_arr))

# adj_matrix = torch.load(os.path.join(base_path, 'toy_data_generation/grid_adjacency_matrix.pt'), weights_only=True)
df_node = pd.read_csv(os.path.join(base_path, 'tokyoPT2019/actformer_input/node.csv'))
df_node['node_id'] = df_node['node_id'].apply(lambda x: loc2id.get(int(float(x)), Z + 3) if pd.notna(x) and str(x) != '99.0' else Z + 3)
df_node = df_node.sort_values(by='node_id')
df_node = df_node.drop(columns=['node_id'])  # もし使わないなら削除（あるいは保持してもOK）

node_features = torch.tensor(df_node.to_numpy(), dtype=torch.float32)
network = Network(node_features)
node_features_np = node_features.numpy()

time_vocab_size = TT + 4 # 時間数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
loc_vocab_size = Z + 4
act_vocab_size = A + 4
context_dim = context_data.shape[1] - 1 # 個人特徴量の次元数（個人IDは除外）
feature_dim = network.node_features.shape[1] # 特徴量の次元数 

# 学習のハイパーパラメータ
num_epoch = wandb.config.epochs #エポック数
eta = wandb.config.learning_rate #学習率
batch_size = wandb.config.batch_size

#RoutesFormerのハイパーパラメータ
l_max = wandb.config.l_max #シークエンスの最大長さ # 開始・終了＋経路長
B_en = wandb.config.B_en #エンコーダのブロック数 # 元論文より
B_de = wandb.config.B_de #デコーダのブロック数 # 元論文より
head_num = wandb.config.head_num #ヘッド数　＃基本的にヘッド数は変えない，4の倍数にする，マルチヘッドの部分
d_ie_time = wandb.config.d_ie_time #トークンの埋め込み次元数
d_ie_loc = wandb.config.d_ie_loc 
d_ie_act = wandb.config.d_ie_act
d_fe = wandb.config.d_fe #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
d_ff = wandb.config.d_ff #フィードフォワード次元数
eos_weight = wandb.config.eos_weight #EOSトークンの重み
savefilename = wandb.config.savefilename #モデルの保存ファイル名
stay_weight = wandb.config.stay_weight
mask_rate = wandb.config.mask_rate #マスク率
alignment_loss_weight = wandb.config.alignment_loss_weight #
identical_penalty_weight = wandb.config.identical_penalty_weight # 同じトークンが続くときのペナルティ

class MultiModalDataset(Dataset):
    def __init__(self, time_data, loc_data, act_data,
                 context_data
                 ):
        """
        act_data :torch.Tensor or np.ndarray, shape = [N, seq_len]
        loc_data : 同上
        """
        # 一旦torch.Tensorに変換しておくと後段が楽
        if not isinstance(time_data, torch.Tensor):
            time_data = torch.tensor(time_data, dtype=torch.long)
        if not isinstance(act_data, torch.Tensor):
            act_data = torch.tensor(act_data, dtype=torch.long)
        if not isinstance(loc_data, torch.Tensor):
            loc_data = torch.tensor(loc_data, dtype=torch.long)
        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float32)
        
        self.time_data = time_data
        self.loc_data = loc_data
        self.act_data = act_data
        self.context_data = context_data

        # 念のため長さが全部同じかチェック
        assert self.time_data.shape[0] == self.loc_data.shape[0], \
            "act and loc must have the same number of samples"
        # seq_lenは自由にしてOK

    def __len__(self):
        return self.act_data.shape[0]

    def __getitem__(self, idx): # 結局ここは同じ
        return self.time_data[idx], self.loc_data[idx], self.act_data[idx], self.context_data[idx]


dataset = MultiModalDataset(time_data, loc_data, act_data, 
                            context_data
                            ) # classのインスタンス化: initしか実行されない

num_samples = len(dataset) # = len(loc_dataset)
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

num_batches = num_samples // batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

model = Actformer( # インスタンス生成→以降modelで呼び出すとforward関数が呼ばれる
                    context_vocab_sizes= context_vocab_sizes, # contextの数
                    time_vocab_size = time_vocab_size, # どれくらいの時間数があるか
                    loc_vocab_size = loc_vocab_size, 
                    act_vocab_size = act_vocab_size, 

                    time_emb_dim = d_ie_time,
                    loc_emb_dim = d_ie_loc,
                    act_emb_dim = d_ie_act,

                    feature_dim = feature_dim,
                    feature_emb_dim = d_fe,
                    d_ff = d_ff,
                    head_num = head_num,
                    B_en = B_en,
                    B_de = B_de,
                    )
model = model.to(device)

def find_eos_positions(logits, eos_token_id):
    predicted_ids = logits.argmax(dim = -1) # (batch_size, seq_len)
    eos_position = []

    for seq in predicted_ids: # seq: (seq_len,)
        eos_idx = (seq == eos_token_id).nonzero(as_tuple = True)
        if len(eos_idx[0]) > 0:
            eos_position.append(eos_idx[0][0].item())
        else:
            eos_position.append(len(seq)-1)
    
    return torch.tensor(eos_position, device=logits.device)

def eos_alignment_loss(time_logits, loc_logits, act_logits,
                       time_eos_token_id, loc_eos_token_id, act_eos_token_id, mode = "l1"):
    time_eos_pos = find_eos_positions(time_logits, time_eos_token_id)
    loc_eos_pos = find_eos_positions(loc_logits, loc_eos_token_id)
    act_eos_pos = find_eos_positions(act_logits, act_eos_token_id)

    diff_tl = (time_eos_pos - loc_eos_pos).abs()
    diff_la = (loc_eos_pos - act_eos_pos).abs()
    diff_at = (act_eos_pos - time_eos_pos).abs()

    if mode == 'l1':
        alignment_loss = (diff_tl + diff_la + diff_at).float().mean()
    elif mode == 'l2':
        alignment_loss = (diff_tl**2 + diff_la**2 + diff_at**2).float().mean()
    else:
        raise NotImplementedError
    return alignment_loss


def repetition_penalty_loss(time_logits, loc_logits, act_logits):
    # 各系列の予測IDを取得
    time_preds = time_logits.argmax(dim=-1)  # shape: (B, L)
    loc_preds = loc_logits.argmax(dim=-1)
    act_preds = act_logits.argmax(dim=-1)

    # 前ステップと同じ三連組かどうかチェック
    repeated_mask = (
        (time_preds[:, 1:] == time_preds[:, :-1]) &
        (loc_preds[:, 1:] == loc_preds[:, :-1]) &
        (act_preds[:, 1:] == act_preds[:, :-1])
    ).float()  # shape: (B, L-1), 値は0か1

    # バッチ内平均 or 合計を取る
    identical_penalty = repeated_mask.mean() 
    return identical_penalty


# padding部分は無視する # 最終的に<b><m>以外は<p>になってるはず
criterion_time = nn.CrossEntropyLoss(ignore_index=TT)#, reduction='none') 
criterion_loc = nn.CrossEntropyLoss(ignore_index=Z)#, reduction='none')
criterion_act = nn.CrossEntropyLoss(ignore_index=A)#, reduction='none')

# lossが改善しなければ学習を打ち切る
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 10  # 何エポック改善しなければ止めるか（お好みで）
best_model_state = None

optimizer = torch.optim.Adam(model.parameters(), lr = eta) 
history = {"train_loss": [], "val_loss": []} 
for epoch in range(num_epoch): # 各エポックで学習と評価を繰り返す
    print(f'------- {epoch} th epoch -------')
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    #### 各個人ごとに実行される！！
    batch_count = 0
    for time_batch, loc_batch, act_batch, context_batch in train_loader:
        batch_count += 1
        tokenizer = Tokenization(network, TT, Z, A)
        context_tokens = context_batch 

        # encoderに入れる
        time_tokens, loc_tokens, act_tokens = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "discontinuous")
        time_discontinuous_route_tokens = time_tokens.long().to(device)
        loc_discontinuous_route_tokens = loc_tokens.long().to(device)
        act_discontinuous_route_tokens = act_tokens.long().to(device)
        discontinuous_feature_mat = tokenizer.make_feature_mat(loc_discontinuous_route_tokens).to(device) # 特徴量はノードデータのみなのでloc_dataに対応させて読み込む
        
        # with begin：decoderに入れる
        time_tokens2, loc_tokens2, act_tokens2 = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "simple")
        time_complete_route_tokens = time_tokens2.long().to(device) # long: 変数形式の変換
        act_complete_route_tokens = act_tokens2.long().to(device) 
        loc_complete_route_tokens = loc_tokens2.long().to(device) 
        complete_feature_mat = tokenizer.make_feature_mat(loc_complete_route_tokens).to(device)

        # 教師データ（クロスエントロピー計算用）
        time_tokens3, loc_tokens3, act_tokens3 = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "next")
        time_next_route_tokens = time_tokens3.long().to(device)
        act_next_route_tokens = act_tokens3.long().to(device)
        loc_next_route_tokens = loc_tokens3.long().to(device)

        time_output, loc_output, act_output = model(context_tokens, # この中で自動でcontextと3つのtokensequenceが結合される→
                                                    time_discontinuous_route_tokens, loc_discontinuous_route_tokens, act_discontinuous_route_tokens, discontinuous_feature_mat,  # for encoder
                                                    time_complete_route_tokens, loc_complete_route_tokens, act_complete_route_tokens, complete_feature_mat) # for decoder
        # output of the model: logit
        time_output_copy = time_output.clone()
        act_output_copy = act_output.clone()
        loc_output_copy = loc_output.clone()

        # predicted_indices = time_output[0].argmax(dim=-1)  # dim=-1 は vocab_size 次元で最大値を取る
        # print('argmax', predicted_indices)
        time_softmax_output = F.softmax(time_output[0], dim=-1)
        loc_softmax_output = F.softmax(loc_output[0], dim=-1)
        act_softmax_output = F.softmax(act_output[0], dim=-1)
        # print('softmax output', softmax_output)

        # 最大値とそのインデックスを取得
        _, time_predicted_indices_2 = torch.max(time_softmax_output, dim=-1)  # dim=-1 → 語彙次元で最大を取る
        _, loc_predicted_indices_2 = torch.max(loc_softmax_output, dim=-1)
        _, act_predicted_indices_2 = torch.max(act_softmax_output, dim=-1)

        print('time softmax', time_softmax_output[0])
        print('max softmax', loc_predicted_indices_2)
        print('time_next', time_next_route_tokens[0])
        print('time simple', time_complete_route_tokens[0])
        print('time discontinuous', time_discontinuous_route_tokens[0])

        sys.exit()

        loss_time = criterion_time(time_output.reshape(-1, time_vocab_size), time_next_route_tokens.reshape(-1)) # viewの部分は，テンソル loc_output の形状を変換（reshape）
        loss_act = criterion_act(act_output.reshape(-1, act_vocab_size), act_next_route_tokens.reshape(-1))
        loss_loc = criterion_loc(loc_output.reshape(-1, loc_vocab_size), loc_next_route_tokens.reshape(-1))

        alignment_loss = eos_alignment_loss(time_output, loc_output, act_output, 
                                            time_eos_token_id = tokenizer.time_SPECIAL_TOKENS["<e>"],
                                            loc_eos_token_id = tokenizer.loc_SPECIAL_TOKENS["<e>"],
                                            act_eos_token_id = tokenizer.act_SPECIAL_TOKENS["<e>"],
                                            mode = "l1"
                                            )
        
        identical_loss = repetition_penalty_loss(time_output, loc_output, act_output)
        # 合成損失関数
        loss = loss_time + loss_loc + loss_act + alignment_loss_weight * alignment_loss + identical_penalty_weight * identical_loss
        print('loss_time', loss_time, 'loss_act', loss_act, 'loss_loc', loss_loc, 'alignment_loss', alignment_loss, 'identical_loss', identical_loss)

        if loss < 1:
            print('time teacher', time_next_route_tokens[0], 'shape', time_next_route_tokens[0].shape)
            print('time_predict', time_predicted_indices_2) 
            print('loc teacher', loc_next_route_tokens[0], 'shape', loc_next_route_tokens[0].shape)
            print('loc_predict', loc_predicted_indices_2)
            print('act teacher', act_next_route_tokens[0], 'shape', act_next_route_tokens[0].shape)
            print('act_predict', act_predicted_indices_2)
            print('context teacher', context_tokens[0])
            print('loss_time', loss_time, 'loss_act', loss_act, 'loss_loc', loss_loc, 'alignment_loss', alignment_loss, 'identical_loss', identical_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 損失を累積
        epoch_loss += loss.item()
        num_batches += 1

        # if (i + 1) % 10 == 0:  # 10バッチごとにログ
        #     logger.info(f"Epoch [{epoch+1}/{num_epoch}]")
        #     logger.info(f"  Loss: {loss.item():.4f}")
        #     logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        #     logger.info(f"  Sample Prediction: {outputs_copy[0].argmax(dim=-1).tolist()}")
        #     logger.info(f"  Sample Target: {next_route_tokens[0].tolist()}")

    # 平均損失を計算
    train_loss = epoch_loss / num_batches
    history["train_loss"].append(train_loss)
    # logger.info(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        num_batches = 0
        # for i, batch in enumerate(val_loader): # バリデーションデータが読まれる
        for time_batch, loc_batch, act_batch, context_batch in val_loader:
            time_batch = time_batch.to(device)
            act_batch = act_batch.to(device)
            loc_batch = loc_batch.to(device)
            context_tokens = context_batch.to(device) # そのまま渡すのでいいはず

            tokenizer = Tokenization(network, TT, Z, A)
            '''           
            act_discontinuous_route_tokens, loc_discontinuous_route_tokens = tokenizer.tokenization(act_batch, loc_batch, mode = "discontinuous").long().to(device)
            discontinuous_feature_mat = tokenizer.make_feature_mat(loc_discontinuous_route_tokens).to(device) # 特徴量はノードデータのみなのでloc_dataに対応させて読み込む
            
            # complete_route_tokens = tokenizer.tokenization(route_batch, mode = "simple").long().to(device) # long: 変数形式の変換
            # decoderに入れる
            act_complete_route_tokens, loc_complete_route_tokens = tokenizer.tokenization(act_batch, loc_arr, mode = "simple").long().to(device) # long: 変数形式の変換
            complete_feature_mat = tokenizer.make_feature_mat(loc_complete_route_tokens).to(device)
            
            # next_route_tokens = tokenizer.tokenization(route_batch, mode = "next").long().to(device)
            # 正解データ（クロスエントロピー計算用）
            act_next_route_tokens, loc_next_route_tokens = tokenizer.tokenization(act_batch, loc_batch, mode = "next").long().to(device)
            '''

            # batch が tokenに変換される
            time_tokens, loc_tokens, act_tokens = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "discontinuous")
            time_discontinuous_route_tokens = time_tokens.long().to(device)
            act_discontinuous_route_tokens = act_tokens.long().to(device)
            loc_discontinuous_route_tokens = loc_tokens.long().to(device)
            discontinuous_feature_mat = tokenizer.make_feature_mat(loc_discontinuous_route_tokens).to(device) # 特徴量はノードデータのみなのでloc_dataに対応させて読み込む

            # decoderに入れる
            time_tokens2, loc_tokens2, act_tokens2 = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "simple")
            time_complete_route_tokens = time_tokens2.long().to(device) # long: 変数形式の変換
            act_complete_route_tokens = act_tokens2.long().to(device) # long: 変数形式の変換
            loc_complete_route_tokens = loc_tokens2.long().to(device) 
            complete_feature_mat = tokenizer.make_feature_mat(loc_complete_route_tokens).to(device)

            # 正解データ（クロスエントロピー計算用） 
            time_tokens3, loc_tokens3, act_tokens3 = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "next")
            time_next_route_tokens = time_tokens3.long().to(device)
            act_next_route_tokens = act_tokens3.long().to(device)
            loc_next_route_tokens = loc_tokens3.long().to(device)
            
            ### それかここでくっつける？→ここのはず　tokenizationでの処置は共通なのでここでくっつけるべき，
            time_output, loc_output, act_output = model(
                                        context_tokens, # B * context_dim
                                        time_discontinuous_route_tokens, loc_discontinuous_route_tokens, act_discontinuous_route_tokens, discontinuous_feature_mat,  # for encoder
                                        time_complete_route_tokens, loc_complete_route_tokens, act_complete_route_tokens, complete_feature_mat) # for decoder

            loss_time = criterion_time(time_output.reshape(-1, time_vocab_size), time_next_route_tokens.reshape(-1)) # viewの部分は，テンソル loc_output の形状を変換（reshape）
            loss_act = criterion_act(act_output.reshape(-1, act_vocab_size), act_next_route_tokens.reshape(-1))
            loss_loc = criterion_loc(loc_output.reshape(-1, loc_vocab_size), loc_next_route_tokens.reshape(-1))

            alignment_loss = eos_alignment_loss(time_output, loc_output, act_output, 
                                                time_eos_token_id = tokenizer.time_SPECIAL_TOKENS["<e>"],
                                                loc_eos_token_id = tokenizer.loc_SPECIAL_TOKENS["<e>"],
                                                act_eos_token_id = tokenizer.act_SPECIAL_TOKENS["<e>"],
                                                mode = "l1"
                                                )
            identical_loss = repetition_penalty_loss(time_output, loc_output, act_output)

            # 合成損失関数
            loss = loss_time + loss_loc + loss_act + alignment_loss_weight * alignment_loss + identical_penalty_weight * identical_loss

            print('val-loss: ', loss) # 0.0001
            epoch_loss += loss.item()
            num_batches += 1

    val_loss = epoch_loss / num_batches
    history["val_loss"].append(val_loss)

        # Early Stopping チェック
    if val_loss < best_val_loss - 1e-4:  # 十分な改善があったら
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # モデルの重みを保存
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        print(f"EarlyStopping: {early_stopping_counter}/{early_stopping_patience}")

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break


    # logger.info(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {val_loss:.4f}")
    wandb.log({"total_loss": train_loss, "val_loss": val_loss})
wandb.finish()

# モデルのパラメータを保存 (多くの行列やテンソルを含んでいるため)
save_data = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
}
# torch.save(save_data, os.path.join(base_path, 'RoutesFormer/output', savefilename))
# print("Model weights saved successfully") # /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/training_toydata.py
# 最良モデルを保存（early stoppingがある場合も含む）
torch.save({
    "model_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
}, os.path.join(base_path, 'RoutesFormer/output', savefilename))
print("Best model weights saved successfully")