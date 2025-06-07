## 推論 
# decoder only transformer

import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from network import Network
from tokenization import *
from decoderonly import *
# from utils.logger import logger
import os
os.environ["WANDB_MODE"] = "dryrun"

import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, random_split, DataLoader
# from data_generation.Recursive import TimeFreeRecursiveLogit
import os
import sys
import pandas as pd
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)# ACTGPT_20250512_151256_GPT_krre4zb6ACTGPT_20250512_151256_GPT_krre4zb6
base_path = '/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog'
base_path0 = '/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer'
initial_len = 5 # 5 or 10 or 20
# /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/ACTGPT_20250523_234006_GPT_6omx3bos.pth
print('checking////', os.path.join(base_path0, 'output/ACTGPT_20250523_234006_GPT_6omx3bos.pth'))
# loadfile = torch.load(os.path.join(base_path0, '/output/ACTGPT_20250523_234006_GPT_6omx3bos.pth')) # offline-run-20250512_151255-krre4zb6 ACTGPT_20250523_234006_GPT_6omx3bos
loadfile = torch.load('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/ACTGPT_20250523_234006_GPT_6omx3bos.pth')
config_path = os.path.join(base_path0, 'actgpt0512/wandb/offline-run-20250523_234006-6omx3bos/files', "config.yaml") # /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/actgpt0512/wandb/offline-run-20250523_234006-6omx3bos/files/config.yaml
# /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/ACTGPT_20250523_234006_GPT_6omx3bos.pth
with open(config_path, 'r') as f: # offline-run-20250523_234006-6omx3bos 
    config = yaml.safe_load(f) 
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
print(config)

df_act = pd.read_csv(os.path.join(base_path0, 'actgpt0512/input/act_weekly_filled_structured.csv'))#, index_col=0) #, header=None)
df_time = pd.read_csv(os.path.join(base_path0, 'actgpt0512/input/time_weekly_filled_structured.csv'))#, index_col=0) #, header=None)
df_loc = pd.read_csv(os.path.join(base_path0, 'actgpt0512/input/mesh_weekly_filled_structured.csv'))#, index_col=0) #, header=None)
df_indivi_ori = pd.read_csv(os.path.join(base_path0, 'actgpt0512/input/attri_all.csv'), index_col=None) #, index_col=0)#, header=None)

user_order = df_act.iloc[:, 0].rename('userid')  # Seriesに名前をつける
df_indivi = df_indivi_ori.set_index('userid').reindex(user_order).reset_index()

df_time.columns = range(df_time.shape[1])
df_act.columns = range(df_act.shape[1])
df_act.iloc[:, 1] = df_act.iloc[:, 1].astype(str).str.split('.').str[0] # 最初の行動を整数に

def to_int_if_possible(x):
    try:
        f = float(x)
        i = int(f)
        return i if f == i else x  # 2.0 → 2, 2.5 はそのまま（必要なら切り捨てにしても可）
    except:
        return x  # 変換できない文字列（例：'<s1>'など）はそのまま返す

df_act_val = df_act.iloc[1:, 1:].values.astype(str)
nunique_act = np.unique(df_act_val)
print('act num:', len(nunique_act)) 
print(nunique_act)

df_indivi['age'] = df_indivi['age'].apply(lambda x: x // 10)
context_vocab_sizes_dict = {}

for col in df_indivi.columns:
    context_vocab_sizes_dict[col] = len(df_indivi[col].unique()) # ユニークな値の数を辞書に格納
context_vocab_sizes = list(context_vocab_sizes_dict.values())

# 独立なvalueの数
TT = 33 # 0-24, 25, 26, 27-> 28 # len(set(time2id.values())) # 時間数 24
Z = 145 # ユニークなメッシュが137個　このほかに特殊トークン
A = 13 # Aは１始まりなので # len(set(act2id.values())) # nan以外は通常トークンにした
# time2id = {'<s>':26} # TT = 27
time2id = {'<s>': 26, '<s1>': 27, '<s2>': 28, '<s3>': 29, '<s4>': 30, '<s5>': 31, '<s6>': 32} #, '<s7>': 33, '<s8>': 34, '<s9>': 35, 'nan': 36} # nanは<955>に変換
# act2id = {'955': 6, '<s>': 7} # A = 7
act2id = {'955': 6, '<s>': 7, '<s1>': 8, '<s2>': 9, '<s3>': 10, '<s4>': 11, '<s5>': 12, '<s6>': 13}  # 955はnanの代わりに入れている

uniqueloc_list = np.unique(df_loc.iloc[:, 1:].values.flatten())
# uniqueloc_list.remove('<p>')
uniqueloc_list = uniqueloc_list[uniqueloc_list != '<p>']

uniqueloc_list = sorted(uniqueloc_list)
loc2id = {str(loc): i for i, loc in enumerate(uniqueloc_list)}
loc2id_add = {'999999': Z-8, '<s>': Z-7, '<s1>': Z-6, '<s2>': Z-5, '<s3>': Z-4, '<s4>': Z-3, '<s5>': Z-2, '<s6>': Z-1}
loc2id.update(loc2id_add)
print('TT', TT, 'A', A, 'Z', Z)

df_time = df_time.applymap(lambda x: time2id.get(str(x), x) if pd.notna(x) else x)
df_act  = df_act.applymap(lambda x: act2id.get(str(x), x)  if pd.notna(x) else x)
df_loc = df_loc.applymap(lambda x: loc2id.get(str(x), x)  if pd.notna(x) else x)
df_time = df_time.iloc[:, 1:] # index, useridを除く
df_act = df_act.iloc[:, 1:] # index, useridを除く
df_loc = df_loc.iloc[:, 1:] # index, useridを除く

# floatとかをintに変換＆<p>
def safe_float_to_int_act(x): # actを0始まりに変える
    try:
        if isinstance(x, str) and x.strip() in ['<p>']:
            return int(float(A)) # <p>だったらtokenizationに対応してA,Z,TTを返す
        return int(float(x)) - 1 
    except:
        return 0  

def safe_float_to_int_time(x): # timeは0始まりのままなのでOK
    try:
        if isinstance(x, str) and x.strip() in ['<p>']:
            return int(float(TT))
        return int(float(x))
    except:
        return 0  
    
def safe_float_to_int_loc(x): # locは0始まりのままなのでOK
    try:
        if isinstance(x, str) and x.strip() in ['<p>']:
            return int(float(Z))
        return int(float(x))
    except:
        return 0

context_arr = df_indivi.fillna(-1).astype('int32').to_numpy()

df_act_clean = df_act.applymap(safe_float_to_int_act)
df_time_clean = df_time.applymap(safe_float_to_int_time)
df_loc_clean = df_loc.applymap(safe_float_to_int_loc)

gender_tensor = df_indivi['gender'].apply(safe_float_to_int_time).astype('int32').to_numpy()
age_tensor = df_indivi['age'].apply(safe_float_to_int_time).astype('int32').to_numpy()

# 最終的にテンソルを結合
context_arr = np.stack([gender_tensor, age_tensor], axis=1)  # shape = [N, 2]
time_arr = df_time_clean.astype('int32').to_numpy()
act_arr = df_act_clean.astype('int32').to_numpy()
loc_arr = df_loc_clean.astype('int32').to_numpy()

time_tensor = torch.from_numpy(time_arr)
act_tensor = torch.from_numpy(act_arr)
loc_tensor = torch.from_numpy(loc_arr)

time_data = torch.from_numpy(time_arr)
loc_data = torch.from_numpy(loc_arr)
act_data = torch.from_numpy(act_arr)
context_data = torch.from_numpy(context_arr)
'''contextが読めているかの確認のために全部0にしてみる'''
# context_data = torch.zeros_like(torch.from_numpy(context_arr))

time_vocab_size = TT + 4 # 時間数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
loc_vocab_size = Z + 4
act_vocab_size = A + 4

df_act_clean.to_csv(os.path.join(base_path0, f'output/0506act_teacher_{initial_len}.csv'))
df_time_clean.to_csv(os.path.join(base_path0, f'output/0506time_teacher_{initial_len}.csv'))
df_indivi.to_csv(os.path.join(base_path0, f'output/0506context_teacher_{initial_len}.csv'))

tokenizer = TokenizationGPT(network = None, TT = TT, Z = Z, A = A)


class MultiModalDataset(Dataset):
    def __init__(self, time_data, 
                 loc_data, 
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
        if not isinstance(loc_data, torch.Tensor):
            loc_data = torch.tensor(loc_data, dtype=torch.long)
        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float32)
        
        self.time_data = time_data
        self.loc_data = loc_data
        self.act_data = act_data
        self.context_data = context_data

        # 念のため長さが全部同じかチェック
        assert self.time_data.shape[0] == self.act_data.shape[0], \
            "act and loc must have the same number of samples" # seq_lenは自由にしてOK
        assert self.time_data.shape[0] == self.loc_data.shape[0], \
            "act and loc must have the same number of samples"

    def __len__(self):
        return self.act_data.shape[0]

    def __getitem__(self, idx): 
        return self.time_data[idx], self.loc_data[idx], self.act_data[idx], self.context_data[idx]


########################################
# 3. モデルの作成・読み込み
########################################

B_de = config['B_de'] #デコーダのブロック数
head_num = config['head_num'] #ヘッド数
d_ie_time = config['d_ie_time'] #トークンの埋め込み次元数
d_ie_loc = config['d_ie_loc']
d_ie_act = config['d_ie_act']
d_fe = config['d_fe'] #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
d_ff = config['d_ff'] #フィードフォワード次元数
batch_size = config['batch_size'] #バッチサイズ
# l_max = config['l_max'] #シークエンスの最大長さ
l_max = 83
eos_weight = config['eos_weight'] #終了トークンの重み
separate_weight = config['separate_weight'] #分離トークンの重み

print('************* Hyperparameters *************')
print('l_max', l_max)
# print('B_en', B_en)
print('B_de', B_de)
print('head_num', head_num)
print('d_ie_time', d_ie_time)
print('d_ie_loc', d_ie_loc)
print('d_ie_act', d_ie_act)
print('d_fe', d_fe)
print('d_ff', d_ff)
print('batch_size', batch_size)
print('********************************************')

# バッチ化
dataset = MultiModalDataset(time_data, loc_data, act_data, context_data)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

model = GPT( # インスタンス生成→以降modelで呼び出すとforward関数が呼ばれる
            context_vocab_sizes = context_vocab_sizes,
            time_vocab_size = time_vocab_size, # どれくらいの時間数があるか
            loc_vocab_size = loc_vocab_size, 
            act_vocab_size = act_vocab_size, 
            time_emb_dim = d_ie_time, 
            loc_emb_dim = d_ie_loc, 
            act_emb_dim = d_ie_act,

            #feature_dim = feature_dim,
            # feature_emb_dim = d_fe,
            d_ff = d_ff,
            head_num = head_num,
            # B_en = B_en,
            B_de = B_de).to(device)

model.load_state_dict(loadfile['model_state_dict'])
model.eval()

########################################
# 4. 推論用の関数を分割して定義
########################################

# def generate_next_zone_logits(
#                             context_tokens, 
#                             model, 
#                             time_batch, # loc_batch, 
#                             act_batch, #disc_feats, 
#                             time_traveled_route, # loc_traveled_route, 
#                             act_traveled_route): #, traveled_feats):
        
#     time_output, act_output = model( # decoderの出力
#                                     context_tokens,
#                                     # time_batch, # loc_batch, act_batch, disc_feats, # -> encoder 
#                                     time_traveled_route, # loc_traveled_route, 
#                                     act_traveled_route) #, traveled_feats) # -> decoder

#     last_time_logits = time_output[:, -1, :]  # time_output から最後のステップのlogitsのみ取得 # softmaxに入れる値が入ってる# shape: (batch_size, time_vocab_size)    
#     if time_output.shape[1] != 1:
        
#         # time_range = torch.arange(time_vocab_size, device=device).unsqueeze(0)
#         # mask = time_range <= cur_time.unsqueeze(1)
#         # last_time_logits = last_time_logits.masked_fill(mask, float('-inf'))
        
#         # fallback: 全部 -inf の場合はゼロベクトルにしてsoftmaxエラー回避
#         all_inf_mask = torch.isinf(last_time_logits).all(dim=-1)
#         last_time_logits[all_inf_mask] = torch.zeros_like(last_time_logits[all_inf_mask])
#         time_output[:, -1, :] = last_time_logits
    
#     return time_output[:, -1, :], act_output[:, -1, :] 


# def apply_neighbor_mask(logits, neighbor, newest_zone, d_tensor):
#     """
#     neighbor マスクを生成して logits に加算する。
#     """
#     neighbor_mask = neighbor.make_neighbor_mask(newest_zone, d_tensor).to(device)
#     # マスクを加算
#     masked_logits = logits + neighbor_mask
#     return masked_logits

def sample_next_zone(masked_logits): 
    """
    softmaxしてトークンをサンプリングする（multinomial）。
    """
    masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
    output_softmax = F.softmax(masked_logits, dim=-1)
    next_zone = torch.multinomial(output_softmax, num_samples=1).squeeze(-1) # 確率分布に従ってトークンをランダムにサンプリング(確定的ではない！！)
    return next_zone

# これまで通った経路＋次のゾーン→結合したい
def update_traveled_route(time_traveled_route, loc_traveled_route, act_traveled_route, 
                          time_next_zone, loc_next_zone, act_next_zone): #, time_batch, img_dic, time_is_day):
    """
    traveled_route に next_zone を追加し、特徴行列 (features) も更新する。
    """
    l1  = act_traveled_route.shape
    time_traveled_route = torch.cat([time_traveled_route, time_next_zone.unsqueeze(1)], dim=1)
    act_traveled_route = torch.cat([act_traveled_route, act_next_zone.unsqueeze(1)], dim=1)
    loc_traveled_route = torch.cat([loc_traveled_route, loc_next_zone.unsqueeze(1)], dim=1)
    # traveled_feature_mat = tokenizer.make_feature_mat(loc_traveled_route).to(device)
    l2 = act_traveled_route.shape

    return time_traveled_route, loc_traveled_route, act_traveled_route # , traveled_feature_mat


####### 重要 ########
def check_and_save_completed_routes( 
    context_tokens,# 未終了のバッチのためのコンテキスト
    context_save_route, # 終了済みのバッチのためのコンテキスト
    time_traveled_route, loc_traveled_route, act_traveled_route,# traveled_feats,
    l_max, tokenizer, idx_lis,
    time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
    time_batch, loc_batch, act_batch,# disc_feats, 
    time_save_route, loc_save_route, act_save_route,
    time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_route,
    time_clone, loc_clone, act_clone, context_clone,
    global_indices
    ):

    context_tokens_original = context_tokens.clone()
    time_tokens_original = time_batch[:, :-1].clone()
    loc_tokens_original = loc_batch[:, :-1].clone()
    act_tokens_original = act_batch[:, :-1].clone()

    # ここ？？？
    time_tokens_original = time_batch[:, :-5].clone()
    loc_tokens_original = loc_batch[:, :-5].clone()
    act_tokens_original = act_batch[:, :-5].clone()

    """
    最新トークンが <e> のものを最終的な出力として保存し、バッチから除去して返す
    """
    # 1. 各系列の末尾が <e> かどうか
    time_is_end = time_traveled_route[:, -1] == tokenizer.time_SPECIAL_TOKENS["<E>"]
    act_is_end = act_traveled_route[:, -1] == tokenizer.act_SPECIAL_TOKENS["<E>"]
    loc_is_end = loc_traveled_route[:, -1] == tokenizer.loc_SPECIAL_TOKENS["<E>"]

    # 2. どれか1つでも <e> が出ていれば True
    any_is_end = time_is_end | act_is_end #| loc_is_end

    # 3. <e> が出ているサンプルのインデックス
    end_indices = torch.where(any_is_end)[0]

    if len(end_indices) > 0:
        # 4. 強制的に他の系列も <e> にする 
        for i in end_indices:
            if not time_is_end[i]:
                time_traveled_route[i, -1] = tokenizer.time_SPECIAL_TOKENS["<E>"]
            if not act_is_end[i]:
                act_traveled_route[i, -1] = tokenizer.act_SPECIAL_TOKENS["<E>"]
            if not loc_is_end[i]:
                loc_traveled_route[i, -1] = tokenizer.loc_SPECIAL_TOKENS["<E>"]

        # 5. パディングして保存
        act_padded_routes = torch.nn.functional.pad(
            act_traveled_route[end_indices], (0, l_max - act_traveled_route.size(1) - 1), # +1?????
            value=tokenizer.act_SPECIAL_TOKENS["<p>"]
        )
        loc_padded_routes = torch.nn.functional.pad(
            loc_traveled_route[end_indices], (0, l_max - loc_traveled_route.size(1) - 1),
            value=tokenizer.loc_SPECIAL_TOKENS["<p>"]
        )
        time_padded_routes = torch.nn.functional.pad(
            time_traveled_route[end_indices], (0, l_max - time_traveled_route.size(1) - 1), # -1? 
            value=tokenizer.time_SPECIAL_TOKENS["<p>"]
        )
        # globalでのインデックス（変化しない）
        true_global_indices = global_indices[end_indices] 
        true_batch_indices = end_indices

        # # 2. 教師データの保存部分を修正
        # time_teacher_route[true_global_indices] = time_batch[true_batch_indices, 1:]  # cloneでなくても良い
        # loc_teacher_route[true_global_indices] = loc_batch[true_batch_indices, 1:]
        # act_teacher_route[true_global_indices] = act_batch[true_batch_indices, 1:]
        # context_teacher_route[true_global_indices] = context_tokens[true_batch_indices]

        print('time traveled', time_traveled_route.shape, 'loc traveled', loc_traveled_route.shape, 'act traveled', act_traveled_route.shape)
        print('time teacher', time_teacher_route.shape, 'loc teacher', loc_teacher_route.shape, 'act teacher', act_teacher_route.shape)
        print('original', time_tokens_original.shape, loc_tokens_original.shape, act_tokens_original.shape)
        time_save_route[true_global_indices] = time_padded_routes # 推論結果
        act_save_route[true_global_indices] = act_padded_routes
        loc_save_route[true_global_indices] = loc_padded_routes
        # context_save_route[true_global_indices] = context_tokens_original[end_indices]

        context_save_route[true_global_indices] = context_tokens_original[end_indices].long()
        time_teacher_route[true_global_indices] = time_tokens_original[end_indices] # orijinalの方がシーケンス長が長い，，
        loc_teacher_route[true_global_indices] = loc_tokens_original[end_indices]
        act_teacher_route[true_global_indices] = act_tokens_original[end_indices]
        
        # context_teacher_route[true_global_indices] = context_clone[end_indices]
        context_teacher_route[true_global_indices] = context_clone[end_indices].long()
        # save ファイルは256分ある，だんだん完成版で埋まっていく感じ

        # 6. マスクで除去
        '''
        mask_del = torch.ones(time_traveled_route.size(0), dtype=torch.bool, device=device)
        mask_del[end_indices] = False
        # mask_delは終了してないところだけTrue→終了してないところが残る

        # 7. バッチ更新（以下元のコードと同様）
        time_traveled_route = time_traveled_route[mask_del]
        act_traveled_route = act_traveled_route[mask_del]
        # loc_traveled_route = loc_traveled_route[mask_del]
        # traveled_feats = traveled_feats[mask_del]
        idx_lis = idx_lis[mask_del]

        time_infer_start_indices = time_infer_start_indices[mask_del]
        act_infer_start_indices = act_infer_start_indices[mask_del]
        # loc_infer_start_indices = loc_infer_start_indices[mask_del]
        time_batch = time_batch[mask_del]
        act_batch = act_batch[mask_del]
        # loc_batch = loc_batch[mask_del]
        context_tokens = context_tokens[mask_del]
        # disc_feats = disc_feats[mask_del]
        '''
    
    return (
            context_tokens,
            context_save_route,
            time_traveled_route, loc_traveled_route, act_traveled_route, #traveled_feats,
            idx_lis,
            # time_d_tensor, loc_d_tensor, act_d_tensor,
            time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
            time_batch, loc_batch, act_batch,# disc_feats, 
            time_save_route, loc_save_route, act_save_route,
            time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_route,
            global_indices
            ) 


########################################
# 5. 推論の実行（メイン部分）
########################################

def run_inference(test_loader, model, tokenizer, l_max, initial_len):
    """
    実際にtest_loaderからバッチを読み出し、推論を行うメイン関数。
    """
    time_all_results = []
    loc_all_results = []
    act_all_results = []
    context_all_results = []
    time_all_teacher = []
    loc_all_teacher = []
    act_all_teacher = []
    context_all_teacher = []
    global_indices = torch.arange(batch_size, device = device)
    # initial_len = 5
    print('-----initial_len----', initial_len)

    for time_batch, loc_batch, act_batch, context_batch in test_loader:
        # print('time batch', time_batch.shape, 'act batch', act_batch.shape) # time batch torch.Size([64, 21]) act batch torch.Size([64, 21])
        time_clone = time_batch[:, :-1].clone()
        loc_clone = loc_batch[:, :-1].clone()
        act_clone = act_batch[:, :-1].clone()
        context_clone = context_batch.clone()

        time_batch = time_batch.to(device)
        act_batch = act_batch.to(device)
        loc_batch = loc_batch.to(device)
        context_tokens = context_batch.to(device)

        time_tokens, loc_tokens, act_tokens = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "complete")

        time_end_indices = (time_tokens == tokenizer.time_SPECIAL_TOKENS["<E>"]).float().argmax(dim=1)
        act_end_indices = (act_tokens == tokenizer.act_SPECIAL_TOKENS["<E>"]).float().argmax(dim=1)
        loc_end_indices = (loc_tokens == tokenizer.loc_SPECIAL_TOKENS["<E>"]).float().argmax(dim=1)
        
        time_begin_indices = (time_tokens == tokenizer.time_SPECIAL_TOKENS["<B>"]).float().argmax(dim=1)
        act_begin_indices = (act_tokens == tokenizer.act_SPECIAL_TOKENS["<B>"]).float().argmax(dim=1)
        loc_begin_indices = (loc_tokens == tokenizer.loc_SPECIAL_TOKENS["<B>"]).float().argmax(dim=1)        

        time_infer_start_indices = time_begin_indices + 1 # 全部1
        act_infer_start_indices = act_begin_indices + 1
        loc_infer_start_indices = loc_begin_indices + 1
        # disc_feature_mat = tokenizer.make_feature_mat(loc_tokens).to(device) ## ここ直したけど

        # 推論中のルートを初期化 # 最初は開始トークンになるのでbを入れる
        # i = 0
        # time_traveled_route = torch.full((batch_size, 1), tokenizer.time_SPECIAL_TOKENS["<B>"], dtype=torch.long).to(device)
        # act_traveled_route = torch.full((batch_size, 1), tokenizer.act_SPECIAL_TOKENS["<B>"], dtype=torch.long).to(device)
        # loc_traveled_route = torch.full((batch_size, 1), tokenizer.loc_SPECIAL_TOKENS["<b>"], dtype=torch.long).to(device)
        # traveled_feats = tokenizer.make_feature_mat(loc_traveled_route).to(device) # paddingに対応した特徴量ベクトルになる
        
        # 例: time_batch の最初の 5 トークンを使う
        i = initial_len - 1
        # print('&&&&&&initial_len', initial_len)
        time_traveled_route = time_tokens[:, :initial_len].clone()
        act_traveled_route = act_tokens[:, :initial_len].clone()
        loc_traveled_route = loc_tokens[:, :initial_len].clone()
        # print('inittime_traveled_route', time_traveled_route.shape, 'act_traveled_route', act_traveled_route.shape, 'time tokes', time_tokens.shape, 'act tokens', act_tokens.shape) # time_traveled_route torch.Size([64, 5]) act_traveled_route torch.Size([64, 5])
        # print('^^^^^^^^^^^')

        # 出力を保存する変数 (l_maxに合わせたサイズに最終的に揃える) # 64*18なので元のバッチの形状と同じ
        time_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device) # 1個減らしてみた
        act_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device) 
        loc_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device)
        context_save_tokens = torch.full((context_tokens.shape[0], context_tokens.shape[1]), fill_value= 999, dtype=torch.long).to(device) # contextは最初999で埋める

        # 教師データの保存
        time_teacher_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device) # lmaxが21なので20列になる
        loc_teacher_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device)
        act_teacher_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device)
        
        '''
        time_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device) # lmaxが21なので20列になる
        act_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device) 
        loc_save_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device)
        context_save_tokens = torch.full((context_tokens.shape[0], context_tokens.shape[1]), fill_value= 999, dtype=torch.long).to(device) # contextは最初999で埋める

        # 教師データの保存
        time_teacher_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device) # lmaxが21なので20列になる
        loc_teacher_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device)
        act_teacher_route = torch.zeros((batch_size, l_max-1), dtype=torch.long).to(device)
        '''

        context_teacher_tokens = torch.full((context_tokens.shape[0], context_tokens.shape[1]), fill_value= 999, dtype=torch.long).to(device) # contextは最初999で埋める

        # バッチ内サンプルのインデックス管理 ####### 揃ってないといけない ########
        idx_lis = torch.arange(batch_size, device=device)
        
        i = initial_len - 1
        # print('i', i)
        while i <= l_max - 3: # timestep+2(前後トークンのはず)→変？
            # print(f'---- inference ste .   .p at {i} ----')
            # time_next_zone_logits, act_next_zone_logits = generate_next_zone_logits(
            #                                                 context_tokens, # 終了した分は削除されてる
            #                                                 model, 
            #                                                 # time_batch, loc_batch, act_batch, disc_feature_mat, ## 
            #                                                 time_tokens, # loc_tokens, 
            #                                                 act_tokens,# disc_feature_mat,
            #                                                 time_traveled_route, # loc_traveled_route, 
            #                                                 act_traveled_route #, traveled_feats
            #                                                 )
            time_output, loc_output, act_output = model( # decoderの出力
                                            context_tokens,
                                            # time_batch, # loc_batch, act_batch, disc_feats, # -> encoder 
                                            time_traveled_route, loc_traveled_route, act_traveled_route
                                            ) #, traveled_feats) # -> decoder
            
            time_next_zone_logits = time_output[:, -1, :] 
            loc_next_zone_logits = loc_output[:, -1, :] # loc_output[:, -1, :] # locはloc_traveled_routeから取得
            act_next_zone_logits = act_output[:, -1, :]

            # softmax + サンプリング 
            time_next_zone = sample_next_zone(time_next_zone_logits) 
            act_next_zone = sample_next_zone(act_next_zone_logits) 
            loc_next_zone = sample_next_zone(loc_next_zone_logits)

            # traveled_route の更新 # 次のsoftmax最大のトークンを用いてupdate
            l1 = act_traveled_route.shape
            time_traveled_route, loc_traveled_route, act_traveled_route = update_traveled_route(
                                                                            time_traveled_route, loc_traveled_route, act_traveled_route,
                                                                            time_next_zone, loc_next_zone, act_next_zone
                                                                            )
            l2 = act_traveled_route.shape

            ####### 重要 ####### 終了トークン <e> を出したサンプルを確認して保存＆削除
            (context_tokens, 
             context_save_tokens, # context_padded,
             time_traveled_route, loc_traveled_route, act_traveled_route,# traveled_feats, 
             idx_lis,
             time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
             #  time_batch, loc_batch, act_batch, disc_feature_mat,
             time_tokens, loc_tokens, act_tokens, #disc_feature_mat,
             time_save_route, loc_save_route, act_save_route,
             time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_tokens,
             global_indices # 他のtraveled routeなどと同様に入力を更新して出力
             ) = check_and_save_completed_routes(
                    context_tokens, 
                    context_save_tokens,# context_padded,
                    time_traveled_route, loc_traveled_route, act_traveled_route, #traveled_feats, 
                    l_max, tokenizer, idx_lis, # loc_idx_lis,
                    time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
                    # time_batch, loc_batch, act_batch, disc_feature_mat, 
                    time_tokens, loc_tokens, act_tokens, #disc_feature_mat,
                    time_save_route, loc_save_route, act_save_route,
                    time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_tokens,
                    time_clone, loc_clone, act_clone, context_clone,
                    global_indices
                )

            # バッチ内サンプルがなくなったら終了
            if act_traveled_route.size(0) == 0:
                print("All samples in this batch have finished.", time_traveled_route.size(0), 'これも0のはず')
                break
            i += 1

        time_all_results.append(time_save_route)
        act_all_results.append(act_save_route)
        loc_all_results.append(loc_save_route)
        context_all_results.append(context_save_tokens)

        time_all_teacher.append(time_teacher_route)
        act_all_teacher.append(act_teacher_route)
        loc_all_teacher.append(loc_teacher_route)
        context_all_teacher.append(context_teacher_tokens)

    # 結果をまとめて返す
    time_final_result = torch.cat(time_all_results, dim=0)
    act_final_result = torch.cat(act_all_results, dim=0)
    loc_final_result = torch.cat(loc_all_results, dim=0)
    context_final_result = torch.cat(context_all_results, dim=0)

    time_teacher_final_result = torch.cat(time_all_teacher, dim=0)
    act_teacher_final_result = torch.cat(act_all_teacher, dim=0)
    loc_teacher_final_result = torch.cat(loc_all_teacher, dim=0)
    context_teacher_final_result = torch.cat(context_all_teacher, dim=0)

    print('!!!!!!!finall!!!!!!')
    print('time traveled route', time_traveled_route.shape, 'act traveled route', act_traveled_route.shape) # time traveled route torch.Size([256, 20]) act traveled route torch.Size([256, 20])
    print('time traveled route', time_traveled_route[0], 'act traveled route', act_traveled_route[0]) # time traveled route tensor([[26
    print('time traveled route', time_traveled_route[21], 'act traveled route', act_traveled_route[21]) # time traveled route tensor([[26

    return time_final_result, loc_final_result, act_final_result, context_batch, context_final_result, time_teacher_final_result, loc_teacher_final_result, act_teacher_final_result, context_teacher_final_result, time_traveled_route,loc_traveled_route, act_traveled_route # loc_traveled_route


########################################
# 6. 実際に推論を実行して結果を保存
########################################

time_result, loc_result, act_result, context_batch, context_result, time_teacher_result, loc_teacher_result, act_teacher_result, context_teacher_result, time_traveled_route, loc_traveled_route, act_traveled_route = run_inference(test_loader, model, tokenizer, l_max, initial_len)

time_traveled_route_df = pd.DataFrame(time_traveled_route.cpu().numpy())
act_traveled_route_df = pd.DataFrame(act_traveled_route.cpu().numpy())
loc_traveled_route_df = pd.DataFrame(loc_traveled_route.cpu().numpy())

# CSVへ保存
time_result_df = pd.DataFrame(time_result.cpu().numpy())
act_result_df = pd.DataFrame(act_result.cpu().numpy())
loc_result_df = pd.DataFrame(loc_result.cpu().numpy())
context_df = pd.DataFrame(context_result.cpu().numpy())

time_teacher_result_df = pd.DataFrame(time_teacher_result.cpu().numpy())
act_teacher_result_df = pd.DataFrame(act_teacher_result.cpu().numpy())
loc_teacher_result_df = pd.DataFrame(loc_teacher_result.cpu().numpy())
context_teacher_df = pd.DataFrame(context_teacher_result.cpu().numpy())

time_result_df.to_csv(os.path.join(base_path0, f'actgpt0512/output/0527time_inference_res_{initial_len}.csv'))
act_result_df.to_csv(os.path.join(base_path0, f'actgpt0512/output/0527act_infer_res_{initial_len}.csv'))
loc_result_df.to_csv(os.path.join(base_path0, f'actgpt0512/output/0527loc_infer_res_{initial_len}.csv'))
context_df.to_csv(os.path.join(base_path0, f'actgpt0512/output/0527context_infer_res_{initial_len}.csv'))
# result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/inference_result.csv')
print("推論結果を保存しました！")
