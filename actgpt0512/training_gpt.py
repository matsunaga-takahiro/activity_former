# Decoder only Transformer for Syutoken PP
# Orijinal: Kurasawa Master Thesis
# Rewritten: Takahiro Matsunaga for 2025 ieee

import torch
import pandas as pd
import numpy as np
from torch import nn
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
from network import *
from tokenization import *
from decoderonly import GPT

wandb.init(
    project="ActGPT0429",
    config={
    "learning_rate": 0.0001,
    "architecture": "Normal",
    "dataset": "0928_day_night",
    "epochs": 500,
    "batch_size": 64,
    "l_max" : 91,
    "B_en" : 1,
    "B_de" : 2,
    "head_num" : 4,
    "d_ie_time" : 16, 
    # "d_ie_loc" : 16, 
    "d_ie_act" : 16, 
    "d_fe" : 4, 
    "d_ff" : 32,
    "eos_weight" : 2.5,
    "separate_weight" : 2.5,
    "mask_rate" : 0.1,
    "savefilename": None,
    "alignment_loss_weight": 0.05,
    "identical_penalty_weight": 1,
    }
)

run_id = wandb.run.id  # or use wandb.run.name
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_GPT")
savefilename = f"ACTGPT_{timestamp}_{run_id}.pth"
wandb.config.update({"savefilename": savefilename}, allow_val_change=True)
base_path = '/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# df_act = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog_finalinput_cleaned.csv', index_col=0) #, header=None)
# df_time = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog_finalinput_cleaned.csv', index_col=0) #, header=None)
df_act = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog_finalinput_structured.csv', index_col=0) #, header=None)
df_time = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog_finalinput_structured.csv', index_col=0) #, header=None)
df_indivi_ori = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/attri_all.csv', index_col=None) #, index_col=0)#, header=None)

# print('df_indivi_ori')
# print(df_indivi_ori.head(5))
# print('df_act[:, 1]')
# print(df_act.iloc[:, 0].head(5))
user_order = df_act.iloc[:, 0].rename('userid')  # Seriesに名前をつける
df_indivi = df_indivi_ori.set_index('userid').reindex(user_order).reset_index()

# print('df_indivi')
# print(df_indivi.head(5))
df_time.columns = range(df_time.shape[1])
df_act.columns = range(df_act.shape[1])

# print('------after-----')
# print(df_indivi, df_indivi.columns)
# df_indivi = df_indivi.iloc[:, 1:] # useridを除く
df_indivi['age'] = df_indivi['age'].apply(lambda x: x // 10)
context_vocab_sizes_dict = {}

# print('======df_indivi_afrer=====') # OK
# print(df_indivi)


for col in df_indivi.columns:
    # print(len(df_indivi[col].unique())) # ユニークな値を取得
    # print(df_indivi[col].unique()) # ユニークな値を取得
    context_vocab_sizes_dict[col] = len(df_indivi[col].unique()) # ユニークな値の数を辞書に格納

context_vocab_sizes = list(context_vocab_sizes_dict.values())

# 独立なvalueの数
TT = 33 # 0-24, 25, 26, 27-> 28 # len(set(time2id.values())) # 時間数 24
# Z = len(set(loc2id.values())) #  
A = 13 # Aは１始まりなので # len(set(act2id.values())) # nan以外は通常トークンにした
# time2id = {'<s>':26} # TT = 27
time2id = {'<s>': 26, '<s1>': 27, '<s2>': 28, '<s3>': 29, '<s4>': 30, '<s5>': 31, '<s6>': 32} #, '<s7>': 33, '<s8>': 34, '<s9>': 35, 'nan': 36} # nanは<955>に変換
# act2id = {'955': 6, '<s>': 7} # A = 7
act2id = {'955': 6, '<s>': 7, '<s1>': 8, '<s2>': 9, '<s3>': 10, '<s4>': 11, '<s5>': 12, '<s6>': 13}  # 955はnanの代わりに入れている
print('TT', TT, 'A', A)

df_time = df_time.applymap(lambda x: time2id.get(str(x), x) if pd.notna(x) else x)
df_act  = df_act.applymap(lambda x: act2id.get(str(x), x)  if pd.notna(x) else x)

df_time = df_time.iloc[:, 1:] # index, useridを除く
df_act = df_act.iloc[:, 1:] # index, useridを除く

# print('after fileter')
# print('df_time')
# print(df_time.iloc[:5, :25])
# print('df_act')
# print(df_act.iloc[:5, :25])

# floatとかをintに変換＆<p>
def safe_float_to_int_act(x): # actを0始まりに変える
    try:
        if isinstance(x, str) and x.strip() in ['<p>']:
            # return x.strip()
            return int(float(A))
        return int(float(x)) - 1 
    except:
        return 0  # or np.nan or any special ID like <m>

def safe_float_to_int_time(x): # timeは0始まりのままなのでOK
    try:
        if isinstance(x, str) and x.strip() in ['<p>']:
            # return x.strip()
            return int(float(TT))
        # それ以外は float → int
        return int(float(x))
    except:
        return 0  # or np.nan or any special ID like <m>

context_arr = df_indivi.fillna(-1).astype('int32').to_numpy()

df_act_clean = df_act.applymap(safe_float_to_int_act)
df_time_clean = df_time.applymap(safe_float_to_int_time)

# print('df_act_clean')
# print(df_act_clean.iloc[:5, :25])
# print('df_time_clean')
# print(df_time_clean.iloc[:5, :25])

gender_tensor = df_indivi['gender'].apply(safe_float_to_int_time).astype('int32').to_numpy()
age_tensor = df_indivi['age'].apply(safe_float_to_int_time).astype('int32').to_numpy()

# 最終的にテンソルを結合
context_arr = np.stack([gender_tensor, age_tensor], axis=1)  # shape = [N, 2]
time_arr = df_time_clean.astype('int32').to_numpy()
time_tensor = torch.from_numpy(time_arr)
act_arr = df_act_clean.astype('int32').to_numpy()
act_tensor = torch.from_numpy(act_arr)

time_data = torch.from_numpy(time_arr)
# loc_data = torch.from_numpy(loc_arr)
act_data = torch.from_numpy(act_arr)
# context_data = torch.from_numpy(context_arr)
context_data = torch.from_numpy(context_arr)
'''contextが読めているかの確認のために全部0にしてみる'''
# context_data = torch.zeros_like(torch.from_numpy(context_arr))

time_vocab_size = TT + 4 # 時間数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
# loc_vocab_size = Z + 4
act_vocab_size = A + 4
# context_dim = context_data.shape[1] - 1 # 個人特徴量の次元数（個人IDは除外）
# feature_dim = network.node_features.shape[1] # 特徴量の次元数 

# hyper parameters
num_epoch = wandb.config.epochs
eta = wandb.config.learning_rate 
batch_size = wandb.config.batch_size
l_max = wandb.config.l_max #シークエンスの最大長さ # 開始・終了＋経路長
B_en = wandb.config.B_en 
B_de = wandb.config.B_de 
head_num = wandb.config.head_num 
d_ie_time = wandb.config.d_ie_time 
# d_ie_loc = wandb.config.d_ie_loc 
d_ie_act = wandb.config.d_ie_act
d_fe = wandb.config.d_fe 
d_ff = wandb.config.d_ff 
eos_weight = wandb.config.eos_weight 
savefilename = wandb.config.savefilename 
separate_weight = wandb.config.separate_weight
mask_rate = wandb.config.mask_rate
alignment_loss_weight = wandb.config.alignment_loss_weight 
identical_penalty_weight = wandb.config.identical_penalty_weight 

class MultiModalDataset(Dataset):
    def __init__(self, time_data, 
                 # loc_data, 
                 act_data,
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
        # if not isinstance(loc_data, torch.Tensor):
        #     loc_data = torch.tensor(loc_data, dtype=torch.long)
        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float32)
        
        self.time_data = time_data
        # self.loc_data = loc_data
        self.act_data = act_data
        self.context_data = context_data

        # 念のため長さが全部同じかチェック
        assert self.time_data.shape[0] == self.act_data.shape[0], \
            "act and loc must have the same number of samples" # seq_lenは自由にしてOK

    def __len__(self):
        return self.act_data.shape[0]

    def __getitem__(self, idx): 
        # return self.time_data[idx], self.loc_data[idx], self.act_data[idx], self.context_data[idx]
        return self.time_data[idx], self.act_data[idx], self.context_data[idx]

# バッチ化
dataset = MultiModalDataset(time_data, act_data, context_data)

num_samples = len(dataset)
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
num_batches = num_samples // batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

model = GPT(
            context_vocab_sizes= context_vocab_sizes, # None, #context_vocab_sizes, 
            time_vocab_size = time_vocab_size, 
            # loc_vocab_size = None, #loc_vocab_size, 
            act_vocab_size = act_vocab_size, 
            time_emb_dim = d_ie_time,
            # loc_emb_dim = None, #  d_ie_loc,
            act_emb_dim = d_ie_act,
            #feature_dim = None, # feature_dim,
            # feature_emb_dim = None, # d_fe,
            d_ff = d_ff,
            head_num = head_num,
            B_de = B_de,
            ).to(device)


class WeightedCrossEntropyWithIgnoreIndex(nn.Module):
    def __init__(self, eos_token_id, eos_weight, ignore_index,
                 separate_ids, separate_weight
                 ): #  stay_weight=0.5 # <-- 新たに追加by kurasawa
        """
        CrossEntropyLoss with weighted EOS token loss, ignore_index, 
        and an optional stay_weight to handle 'stay' transitions.
        """
        super().__init__()
        self.eos_token_id = eos_token_id
        self.eos_weight = eos_weight
        self.separate_ids = separate_ids # list
        self.separate_weight = separate_weight # scaler
        self.ignore_index = ignore_index
        # self.stay_weight = stay_weight  # stayの重み(小さめにする)

        # ベースのCEは reduction='none' で呼び出す
        self.base_loss_fn = nn.CrossEntropyLoss(
            reduction='none', 
            ignore_index=ignore_index
        )
        
    def forward(self, logits, targets): #, former_targets):
        """
        :param logits: [batch_size, seq_len, vocab_size]
        :param targets: [batch_size, seq_len]
        :param former_targets: [batch_size, seq_len]
        """
        # 形状変換
        #print(logits.shape)
        #print(targets.shape)
        #print(former_targets.shape)
        batch_size_seq_len, vocab_size = logits.size()
        #logits = logits.view(-1, vocab_size)           # [batch_size * seq_len, vocab_size]
        #targets = targets.view(-1)                     # [batch_size * seq_len]
        #former_targets = former_targets.view(-1)       # [batch_size * seq_len]

        # 基本のCE lossを計算 (要素ごとの損失: shape [batch_size*seq_len])
        loss_per_token = self.base_loss_fn(logits, targets)

        # デフォルトは weight=1.0
        weights = torch.ones_like(loss_per_token, device=loss_per_token.device)

        # EOSトークンに対して重み付け
        eos_mask = (targets == self.eos_token_id)
        weights[eos_mask] = self.eos_weight
        sep_mask_list = []
        for i in range(len(self.separate_ids)):
            sep_mask = (targets == self.separate_ids[i])
            sep_mask_list.append(sep_mask)
            weights[sep_mask] = self.separate_weight
        # sep_mask = [targets == id for id in self.separate_ids]
        # print('sep_mask', sep_mask, len(sep_mask))
        # print('weights', weights, len(weights))
        # weights[sep_mask] = self.separate_weight
        
        # stay (とどまる) 判定: (target == former_target) かつ ignore_index でない
        # stay_mask = (targets == former_targets) & (targets != self.ignore_index)
        # stay の重み (例: 0.5)
        # weights[stay_mask] = self.stay_weight

        # 最終的な重み付き損失
        weighted_loss = (loss_per_token * weights).sum() / (weights[targets != self.ignore_index].sum())

        return weighted_loss


# preparation for alignment loss
def find_eos_positions(logits, eos_token_id):
    predicted_ids = logits.argmax(dim = -1) 
    eos_position = []

    for seq in predicted_ids: 
        eos_idx = (seq == eos_token_id).nonzero(as_tuple = True)
        if len(eos_idx[0]) > 0:
            eos_position.append(eos_idx[0][0].item())
        else:
            eos_position.append(len(seq)-1)
    
    return torch.tensor(eos_position, device=logits.device)


def eos_alignment_loss(time_logits, 
                    #    loc_logits, 
                       act_logits,
                       time_eos_token_id, 
                    #    loc_eos_token_id, 
                       act_eos_token_id, mode = "l1"):
    time_eos_pos = find_eos_positions(time_logits, time_eos_token_id)
    # loc_eos_pos = find_eos_positions(loc_logits, loc_eos_token_id)
    act_eos_pos = find_eos_positions(act_logits, act_eos_token_id)

    # diff_tl = (time_eos_pos - loc_eos_pos).abs()
    # diff_la = (loc_eos_pos - act_eos_pos).abs()
    diff_at = (act_eos_pos - time_eos_pos).abs()

    if mode == 'l1':
        alignment_loss = (diff_at).float().mean()
    elif mode == 'l2':
        alignment_loss = (diff_at**2).float().mean()
    else:
        raise NotImplementedError
    return alignment_loss


def repetition_penalty_loss(time_logits, 
                            # loc_logits, 
                            act_logits):
    # 各系列の予測IDを取得
    time_preds = time_logits.argmax(dim=-1)  # shape: (B, L)
    # loc_preds = loc_logits.argmax(dim=-1)
    act_preds = act_logits.argmax(dim=-1)

    # 前ステップと同じ三連組かどうかチェック
    repeated_mask = (
        (time_preds[:, 1:] == time_preds[:, :-1]) &
        # (loc_preds[:, 1:] == loc_preds[:, :-1]) &
        (act_preds[:, 1:] == act_preds[:, :-1])
    ).float()  # shape: (B, L-1), 値は0か1

    # バッチ内平均 or 合計を取る
    identical_penalty = repeated_mask.mean() 
    return identical_penalty

# padding部分は無視する # 最終的に<b><m>以外は<p>になってるはず
# time2id = {'<s>': 26, '<s1>': 27, '<s2>': 28, '<s3>': 29, '<s4>': 30, '<s5>': 31, '<s6>': 32} #, '<s7>': 33, '<s8>': 34, '<s9>': 35, 'nan': 36} # nanは<955>に変換
# act2id = {'955': 6, '<s>': 7, '<s1>': 8, '<s2>': 9, '<s3>': 10, '<s4>': 11, '<s5>': 12, '<s6>': 13}  # 955はnanの代わりに入れている

criterion_act = WeightedCrossEntropyWithIgnoreIndex(
    eos_token_id = TT + 1,
    eos_weight = eos_weight,
    separate_ids = [i for i in range(7, 14)], separate_weight = separate_weight,
    ignore_index= TT, # padding
)

criterion_time = WeightedCrossEntropyWithIgnoreIndex(
    eos_token_id = A + 1,
    eos_weight = eos_weight,
    separate_ids = [i for i in range(27, 33)], separate_weight = separate_weight,
    ignore_index= A # padding
)

# criterion_time = nn.CrossEntropyLoss(ignore_index = TT) # loss = CrossEntropyLoss(ignore_index=TT)(input=logits, target=labels)
# criterion_act = nn.CrossEntropyLoss(ignore_index = A) 

##### 学習開始 #####
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 2000  # 何エポック改善しなければ止めるか（お好みで）
best_model_state = None

optimizer = torch.optim.Adam(model.parameters(), lr = eta) 
history = {"train_loss": [], "val_loss": []} 
for epoch in range(num_epoch): # 各エポックで学習と評価を繰り返す
    print(f'------- {epoch} th epoch -------')
    model.train()
    epoch_loss = 0
    num_batches = 0
    batch_count = 0
    for time_batch, act_batch, context_batch in train_loader:
        # print(' ************** in loop **************')
        # print('time tokens', time_batch[0], 'act tokens', act_batch[0], 'context', context_batch[0])
        tokenizer = TokenizationGPT(network = None, TT = TT, A = A)
        context_tokens = context_batch 
        batch_count += 1

        # decoder input 
        time_tokens2, act_tokens2 = tokenizer.tokenization(time_batch, act_batch, mode = "simple")
        time_complete_route_tokens = time_tokens2.long().to(device) # long: 変数形式の変換
        act_complete_route_tokens = act_tokens2.long().to(device) 
        # loc_complete_route_tokens = loc_tokens2.long().to(device) 
        complete_feature_mat = None # tokenizer.make_feature_mat(loc_complete_route_tokens).to(device)

        # 教師データ（クロスエントロピー計算用）
        time_tokens3, act_tokens3 = tokenizer.tokenization(time_batch, act_batch, mode = "next")
        time_next_route_tokens = time_tokens3.long().to(device)
        act_next_route_tokens = act_tokens3.long().to(device)
        
        # logit分布→softmaxで確率分布に変換
        # print('time_complete_route_tokens', time_complete_route_tokens.shape)
        # print('time complete_route_tokens', time_complete_route_tokens.shape, 'act complete_route_tokens', act_complete_route_tokens.shape)
        time_output, act_output = model(
                                        context_tokens, # この中で自動でcontextと3つのtokensequenceが結合される→
                                        # time_discontinuous_route_tokens, loc_discontinuous_route_tokens, act_discontinuous_route_tokens, discontinuous_feature_mat,  # for encoder
                                        time_complete_route_tokens, 
                                        act_complete_route_tokens #, complete_feature_mat
                                        ) # for decoder
 
        # 出力をソフトマックスで確率に変換
        act_probs = F.softmax(act_output, dim=-1)  # shape: [B, L, V]
        time_probs = F.softmax(time_output, dim=-1)  # shape: [B, L, V]

        # <e> トークンのインデックス（例：7）
        act_e_token_id = tokenizer.act_SPECIAL_TOKENS["<E>"]
        time_e_token_id = tokenizer.time_SPECIAL_TOKENS["<E>"]

        # <e> の確率だけ取り出す
        act_e_probs = act_probs[:, :, act_e_token_id]  # shape: [B, L]
        time_e_probs = time_probs[:, :, time_e_token_id]  # shape: [B, L]
        time_softmax_output = F.softmax(time_output[0], dim=-1)
        # loc_softmax_output = F.softmax(loc_output[0], dim=-1)
        act_softmax_output = F.softmax(act_output[0], dim=-1)

        # 最大値とそのインデックスを取得
        _, time_predicted_indices_2 = torch.max(time_softmax_output, dim=-1)  # dim=-1 → 語彙次元で最大を取る
        # _, loc_predicted_indices_2 = torch.max(loc_softmax_output, dim=-1)
        _, act_predicted_indices_2 = torch.max(act_softmax_output, dim=-1)

        loss_time = criterion_time(time_output.reshape(-1, time_vocab_size), time_next_route_tokens.reshape(-1)) # viewの部分は，テンソル loc_output の形状を変換（reshape）
        loss_act = criterion_act(act_output.reshape(-1, act_vocab_size), act_next_route_tokens.reshape(-1))
        # loss_loc = criterion_loc(loc_output.reshape(-1, loc_vocab_size), loc_next_route_tokens.reshape(-1))

        alignment_loss = eos_alignment_loss(time_output, act_output, 
                                            time_eos_token_id = tokenizer.time_SPECIAL_TOKENS["<E>"],
                                            # loc_eos_token_id = tokenizer.loc_SPECIAL_TOKENS["<e>"],
                                            act_eos_token_id = tokenizer.act_SPECIAL_TOKENS["<E>"],
                                            mode = "l1"
                                            )

        identical_loss = repetition_penalty_loss(time_output, act_output)
        
        # 合成損失関数
        loss = loss_time + loss_act + alignment_loss_weight * alignment_loss + identical_penalty_weight * identical_loss
        # print('loss_time', loss_time, 'loss_act', loss_act, 'alignment_loss', alignment_loss, 'identical_loss', identical_loss)

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
        for time_batch, act_batch, context_batch in val_loader:
            time_batch = time_batch.to(device)
            act_batch = act_batch.to(device)
            # loc_batch = loc_batch.to(device)
            context_tokens = context_batch.to(device) # そのまま渡すのでいいはず
            tokenizer = TokenizationGPT(network = None, TT = TT, A = A)

            # encoder input 
            # time_tokens, loc_tokens, act_tokens = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "discontinuous")
            # time_discontinuous_route_tokens = time_tokens.long().to(device)
            # act_discontinuous_route_tokens = act_tokens.long().to(device)
            # loc_discontinuous_route_tokens = loc_tokens.long().to(device)
            # discontinuous_feature_mat = tokenizer.make_feature_mat(loc_discontinuous_route_tokens).to(device) # 特徴量はノードデータのみなのでloc_dataに対応させて読み込む

            # decoder input 
            time_tokens2, act_tokens2 = tokenizer.tokenization(time_batch, act_batch, mode = "simple")
            time_complete_route_tokens = time_tokens2.long().to(device) # long: 変数形式の変換
            act_complete_route_tokens = act_tokens2.long().to(device) # long: 変数形式の変換
            # loc_complete_route_tokens = loc_tokens2.long().to(device) 
            # complete_feature_mat = tokenizer.make_feature_mat(loc_complete_route_tokens).to(device)

            # 正解データ（クロスエントロピー計算用） 
            time_tokens3, act_tokens3 = tokenizer.tokenization(time_batch, act_batch, mode = "next")
            time_next_route_tokens = time_tokens3.long().to(device)
            act_next_route_tokens = act_tokens3.long().to(device)

            # print('context tokens', context_tokens[0], context_tokens.shape)
            # sys.exit()
            # loc_next_route_tokens = loc_tokens3.long().to(device)
            time_output, act_output = model(
                                        context_tokens, # B * context_dim
                                        # time_discontinuous_route_tokens, loc_discontinuous_route_tokens, act_discontinuous_route_tokens, discontinuous_feature_mat,  # for encoder
                                        time_complete_route_tokens, 
                                        act_complete_route_tokens, 
                                        # complete_feature_mat
                                        ) # for decoder
            loss_time = criterion_time(time_output.reshape(-1, time_vocab_size), time_next_route_tokens.reshape(-1)) # viewの部分は，テンソル loc_output の形状を変換（reshape）
            loss_act = criterion_act(act_output.reshape(-1, act_vocab_size), act_next_route_tokens.reshape(-1))
            # loss_loc = criterion_loc(loc_output.reshape(-1, loc_vocab_size), loc_next_route_tokens.reshape(-1))

            alignment_loss = eos_alignment_loss(time_output, act_output, 
                                                time_eos_token_id = tokenizer.time_SPECIAL_TOKENS["<E>"],
                                               # loc_eos_token_id = tokenizer.loc_SPECIAL_TOKENS["<>"],
                                                act_eos_token_id = tokenizer.act_SPECIAL_TOKENS["<E>"],
                                                mode = "l1"
                                                )
            identical_loss = repetition_penalty_loss(time_output, act_output)

            # 合成損失関数
            loss = loss_time + loss_act + alignment_loss_weight * alignment_loss + identical_penalty_weight * identical_loss


            # if loss < 1.5:
            #     print('time teacher', time_next_route_tokens[0], 'shape', time_next_route_tokens[0].shape)
            #     print('time_predict', time_predicted_indices_2) 
            #     # print('loc teacher', loc_next_route_tokens[0], 'shape', loc_next_route_tokens[0].shape)
            #     # print('loc_predict', loc_predicted_indices_2)
            #     print('act teacher', act_next_route_tokens[0], 'shape', act_next_route_tokens[0].shape)
            #     print('act_predict', act_predicted_indices_2)
            #     print('context teacher', context_tokens[0])
            #     print('loss_time', loss_time, 'loss_act', loss_act, 'alignment_loss', alignment_loss, 'identical_loss', identical_loss)

            #     sys.exit()

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

print('********** end of epoch **********')
print('time teacher', time_next_route_tokens[0], 'shape', time_next_route_tokens[0].shape)
print('time_predict', time_predicted_indices_2) 
# print('loc teacher', loc_next_route_tokens[0], 'shape', loc_next_route_tokens[0].shape)
# print('loc_predict', loc_predicted_indices_2)
print('act teacher', act_next_route_tokens[0], 'shape', act_next_route_tokens[0].shape)
print('act_predict', act_predicted_indices_2)
print('context teacher', context_tokens[0])
print('loss_time', loss_time, 'loss_act', loss_act, 'alignment_loss', alignment_loss, 'identical_loss', identical_loss)

# モデルのパラメータを保存 (多くの行列やテンソルを含んでいるため)
save_data = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
}

# 最良モデルを保存（early stoppingがある場合も含む）
torch.save({
    "model_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
}, os.path.join('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output', savefilename))
print("Best model weights saved successfully")