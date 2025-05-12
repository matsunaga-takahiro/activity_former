## 推論 decoder only transformer

import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from network import Network
from tokenization import *
from ActFormer.RoutesFormer.actgpt0512.decoderonly import *
# from utils.logger import logger
import os
import seaborn as sns

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
print(device)


loadfile = torch.load('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/ACTGPT_20250505_151610_GPT_f9crnqk3.pth')
config_path = os.path.join('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/actformer0411/wandb/offline-run-20250505_151610-f9crnqk3/files', "config.yaml") # /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/actformer0411/wandb
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
print(config)

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
initial_len = 10 # 1始まりなので
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

print('context_arr', context_arr.shape)

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
# print('-------context_data------', context_data[0], context_data.shape)
# context_data = torch.zeros_like(torch.from_numpy(context_arr))

time_vocab_size = TT + 4 # 時間数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
# loc_vocab_size = Z + 4
act_vocab_size = A + 4
# context_dim = context_data.shape[1] - 1 # 個人特徴量の次元数（個人IDは除外）
# feature_dim = network.node_features.shape[1] # 特徴量の次元数 


df_act_clean.to_csv(f'/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0506act_teacher_{initial_len}.csv')
df_time_clean.to_csv(f'/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0506time_teacher_{initial_len}.csv')
df_indivi.to_csv(f'/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0506context_teacher_{initial_len}.csv')

tokenizer = TokenizationGPT(network = None, TT = TT, A = A)

class MultiModalDataset(Dataset):
    def __init__(self, time_data, act_data, context_data):
        """
        act_data :torch.Tensor or np.ndarray, shape = [N, seq_len]
        loc_data : 同上
        """

        # 一旦torch.Tensorに変換しておくと後段が楽
        if not isinstance(act_data, torch.Tensor):
            act_data = torch.tensor(act_data, dtype=torch.long)
        # if not isinstance(loc_data, torch.Tensor):
        #     loc_data = torch.tensor(loc_data, dtype=torch.long)
        if not isinstance(time_data, torch.Tensor):
            time_data = torch.tensor(time_data, dtype=torch.long)
        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float32)
        
        self.time_data = time_data
        self.act_data = act_data
        # self.loc_data = loc_data
        self.context_data = context_data
        
        # 念のため長さが全部同じかチェック
        assert self.act_data.shape[0] == self.time_data.shape[0], \
            "act and loc must have the same number of samples"
        # seq_lenは自由にしてOK

    def __len__(self):
        return self.act_data.shape[0]

    def __getitem__(self, idx): # 結局ここは同じ
        return self.time_data[idx], self.act_data[idx], self.context_data[idx] 


########################################
# 3. モデルの作成・読み込み
########################################

B_de = config['B_de'] #デコーダのブロック数
head_num = config['head_num'] #ヘッド数
d_ie_time = config['d_ie_time'] #トークンの埋め込み次元数
# d_ie_loc = config['d_ie_loc']
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
# print('d_ie_loc', d_ie_loc)
print('d_ie_act', d_ie_act)
print('d_fe', d_fe)
print('d_ff', d_ff)
print('batch_size', batch_size)
print('********************************************')

# バッチ化
dataset = MultiModalDataset(time_data, act_data, context_data) # classのインスタンス化: initしか実行されない
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

model = GPT( # インスタンス生成→以降modelで呼び出すとforward関数が呼ばれる
            context_vocab_sizes = context_vocab_sizes,
            time_vocab_size = time_vocab_size, # どれくらいの時間数があるか
            # loc_vocab_size = loc_vocab_size, 
            act_vocab_size = act_vocab_size, 
            time_emb_dim = d_ie_time, 
            # loc_emb_dim = d_ie_loc, 
            act_emb_dim = d_ie_act,

            #feature_dim = feature_dim,
            # feature_emb_dim = d_fe,
            d_ff = d_ff,
            head_num = head_num,
            # B_en = B_en,
            B_de = B_de).to(device)

model.load_state_dict(loadfile['model_state_dict'])
model.eval()



# 2. 任意のユーザデータ（1バッチ）を val_loader などから取得
for time_batch, act_batch, context_batch in test_loader:
    time_s, act_s = tokenizer.tokenization(time_batch[:1], 
                                           act_batch[:1], 
                                           mode="simple")
    break

# 3. hook を仕込んで forward 実行
# attn_maps = []
# def attn_hook(mod, inp, out):
#     attn_maps.append(out[0].detach().cpu())

# model.decoder.blocks[0].MMHA.register_forward_hook(attn_hook)
# _ = model(context_batch[:1].to(device),
#           time_s.to(device), act_s.to(device))

# # 4. heatmap 可視化
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.heatmap(attn_maps[0][0, 0], cmap="magma")
# plt.show()


attn_maps = []

def attn_hook(mod, inp, out):
    if isinstance(out, tuple) and len(out) > 1:
        attn_maps.append(out[1].detach().cpu())  # (B, H, L, L)
    else:
        print("Warning: attention weights not found in output")

# hook 登録
model.decoder.blocks[0].MMHA.register_forward_hook(attn_hook)

# 推論実行
_ = model(context_batch[:1].to(device),
 time_s.to(device), act_s.to(device))

for i in range(10):
    # 可視化
    if len(attn_maps) > 0 and attn_maps[0].ndim == 4:
        w = attn_maps[i][0, 0]  # 最初のサンプル、最初のhead
        sns.heatmap(w, cmap='magma', square=True)
        plt.title(f"Attention Map (Head 0, Sample 0)")
        plt.xlabel("Query Position")
        plt.ylabel("Key Position")
        plt.savefig(f"/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/image/attention_map_{i}.png")
        plt.close()
    else:
        print("Attention map is empty or not 4D. Check your MultiHeadAttention implementation.")

