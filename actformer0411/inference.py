## 推論
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from network import Network
from tokenization import Tokenization
from actformer import Actformer
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

# torch.cuda.init()  /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/actformer0411/wandb/offline-run-20250424_031435-leu73f04/files/config.yaml
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
base_path = '/Users/matsunagatakahiro/Desktop/res2025/ActFormer'
# savefilename = 'ACT_20250423_220328_nrow10000_hs4nib0w.pth' # ca no, ctx no
savefilename = 'ACT_20250424_031435_nrow10000_leu73f04.pth' # model 2
offline_run_path = 'wandb/offline-run-20250424_031435-leu73f04/files' # model2
# savefilename = 'ACT_20250423_050157_nrow10000_0l2ebds5.pth' # model 4
# offline_run_path = 'wandb/offline-run-20250423_050157-0l2ebds5/files' # model4
# savefilename = 'ACT_20250423_205338_nrow10000_eilb0g12.pth'  # contextのみある　　　offline-run-20250423_205338-eilb0g12
# savefilename = 'ACT_20250423_051212_nrow10000_horyepz2.pth' # model 3
# offline_run_path = 'wandb/offline-run-20250423_051211-horyepz2/files' # model3
# savefilename = 'ACT_20250420_225229_nrow7500_04eicnql.pth' # ここは適宜変更
model_weights_path = os.path.join(base_path, 'RoutesFormer/output', savefilename) # 学習済みmodelのアウトプット
loadfile = torch.load(model_weights_path)
print('------check------')
print(loadfile.keys())  # 'model_state_dict', 'optimizer_state_dict', 'config', ...
# print(loadfile['config'])  # 当時の語彙サイズやパラメータが入っているかも

#input data
k = 1 # 10分割してるので
nrow = 10000 # 1000行だけ読み込む

df_time_arr = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/time_cleaned_{k}.csv'), index_col = 0, nrows=nrow)
df_loc_arr = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/loc_cleaned_{k}.csv'), index_col = 0, nrows=nrow) 
df_act_traj = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/act_cleaned_{k}.csv'), index_col = 0, nrows=nrow) 
df_indivi = pd.read_csv(os.path.join(base_path, f'tokyoPT2019/actformer_input/indivi_cleaned_{k}.csv'), index_col = False, nrows=nrow)
df_indivi.drop(columns=['household_size', 'age', 'work_start_am_pm', 'work_start_hour', 'work_start_minute', 'household_income', 'survey_month', 'survey_day'], inplace=True)
print(df_indivi.columns)



loc_columns = df_loc_arr.columns.tolist() # loc_1~6までの列を取得
time_columns = df_time_arr.columns.tolist() # time_1~6までの列を取得
act_columns = df_act_traj.columns.tolist() # act_1~6までの列を取得
context_columns = df_indivi.columns.tolist() # context_1~6までの列を取得   

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
print('context_vocab_sizes', context_vocab_sizes)

# map
time2id = {time: i for i, time in enumerate(unique_time)}  # mapping辞書作成
print('time2id', time2id)
loc2id = {loc: i for i, loc in enumerate(unique_locs)}  # mapping辞書作成
act2id = {1: 0, 2: 0, 
          3: 1, 4: 1, 5: 1, 6: 1, 
          7: 2, 8: 2, 
          9: 3, 10: 3, 
          11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 
          16: 5, 17: 5, 18: 5, # 
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
# print('df_indivi columns', df_indivi.columns, df_indivi.columns[1:])
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


df_time_arr.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411time_teacher_inference_result_crossmodalatt.csv')
df_act_traj.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411act_teacher_inference_result_crossmodalatt.csv')
df_loc_arr.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411loc_teacher_inference_result_crossmodalatt.csv')
df_indivi.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411context_teacher_inference_result_crossmodalatt.csv')

# adj_matrix_np = adj_matrix.numpy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 以下がバッチ分割されてtransformerに入る # 変換自体は成功しているから問題はない
time_data = torch.from_numpy(time_arr)
loc_data = torch.from_numpy(loc_arr)
act_data = torch.from_numpy(act_arr)
context_data = torch.from_numpy(context_arr)

# adj_matrix = torch.load(os.path.join(base_path, 'toy_data_generation/grid_adjacency_matrix.pt'), weights_only=True)
df_node = pd.read_csv(os.path.join(base_path, 'tokyoPT2019/actformer_input/node.csv'))
df_node['node_id'] = df_node['node_id'].apply(lambda x: loc2id.get(int(float(x)), Z + 3) if pd.notna(x) and str(x) != '99.0' else Z + 3)
df_node = df_node.sort_values(by='node_id')
df_node = df_node.drop(columns=['node_id'])  # もし使わないなら削除（あるいは保持してもOK）

node_features = torch.tensor(df_node.to_numpy(), dtype=torch.float32)
network = Network(node_features)
node_features_np = node_features.numpy()

#教師データを保存しておく
time_teacher_df = pd.DataFrame(time_arr)
loc_teacher_df = pd.DataFrame(loc_arr)
act_teacher_df = pd.DataFrame(act_arr)
context_teacher_df = pd.DataFrame(context_arr)
time_teacher_df.to_csv(os.path.join(base_path, 'tokyoPT2019/actformer_input/time_teacher.csv'))
loc_teacher_df.to_csv(os.path.join(base_path, 'tokyoPT2019/actformer_input/loc_teacher.csv'))
act_teacher_df.to_csv(os.path.join(base_path, 'tokyoPT2019/actformer_input/act_teacher.csv'))
context_teacher_df.to_csv(os.path.join(base_path, 'tokyoPT2019/actformer_input/context_teacher.csv'))

#前処理
time_vocab_size = TT + 4 # 時間数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
loc_vocab_size = Z + 4
act_vocab_size = A + 4
context_dim = context_data.shape[1] - 1 # 個人特徴量の次元数（個人IDは除外）
feature_dim = network.node_features.shape[1] # 特徴量の次元数 
print('TT', TT, 'Z', Z, 'A', A)

feature_dim = network.node_features.shape[1] # 特徴量の次元数
tokenizer = Tokenization(network, TT, Z, A)

class MultiModalDataset(Dataset):
    def __init__(self, time_data, loc_data, act_data, context_data):
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
        if not isinstance(time_data, torch.Tensor):
            time_data = torch.tensor(time_data, dtype=torch.long)
        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float32)
        
        self.time_data = time_data
        self.act_data = act_data
        self.loc_data = loc_data
        self.context_data = context_data
        
        # 念のため長さが全部同じかチェック
        assert self.act_data.shape[0] == self.loc_data.shape[0], \
            "act and loc must have the same number of samples"
        # seq_lenは自由にしてOK

    def __len__(self):
        return self.act_data.shape[0]

    def __getitem__(self, idx): # 結局ここは同じ
        return self.time_data[idx], self.loc_data[idx], self.act_data[idx], self.context_data[idx] 


########################################
# 3. モデルの作成・読み込み
########################################

#ハイパーパラメータの取得 /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/actformer0411/wandb/offline-run-20250414_223315-h7djad9a/run-h7djad9a.wandb
# api = wandb.Api() # https://wandb.ai/   ACT_20250423_220328_nrow10000_hs4nib0w     offline-run-20250423_220328-hs4nib0w
# offline_run_path = "wandb/offline-run-20250419_222720-i1qyesqn/files" # 04eicnql
# offline_run_path = "wandb/offline-run-20250423_220328-hs4nib0w/files" 
# offline_run_path = "wandb/offline-run-20250423_050157-0l2ebds5/files" # best model

config_path = os.path.join(base_path, 'RoutesFormer/actformer0411', offline_run_path, "config.yaml") # /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/actformer0411/wandb
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
print(config)  # オフラインでもハイパーパラメータを確認可能
# run = api.run("takahiromtf958-the-university-of-tokyo/ActFormer0411/runs/i1qyesqn") # 04eicnql
# run = api.run("takahiromtf958-the-university-of-tokyo/ActFormer0411/runs/04eicnql") # 04eicnql
# print(f"Run ID: {run.id}, Run Name: {run.name}")
# config = run.config

#RoutesFormerのハイパーパラメータ
l_max = config['l_max'] #シークエンスの最大長さ
B_en = config['B_en'] #エンコーダのブロック数
B_de = config['B_de'] #デコーダのブロック数
head_num = config['head_num'] #ヘッド数
d_ie_time = config['d_ie_time'] #トークンの埋め込み次元数
d_ie_loc = config['d_ie_loc']
d_ie_act = config['d_ie_act']
d_fe = config['d_fe'] #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
d_ff = config['d_ff'] #フィードフォワード次元数
batch_size = config['batch_size'] #バッチサイズ
l_max = config['l_max'] #シークエンスの最大長さ
print('************* Hyperparameters *************')
print('l_max', l_max)
print('B_en', B_en)
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
dataset = MultiModalDataset(time_data, loc_data, act_data, context_data) # classのインスタンス化: initしか実行されない
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

model = Actformer( # インスタンス生成→以降modelで呼び出すとforward関数が呼ばれる
                    context_vocab_sizes = context_vocab_sizes,
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
                    B_de = B_de).to(device)

print('check the file name', savefilename)
model.load_state_dict(loadfile['model_state_dict'])
model.eval()
tokenizer = Tokenization(network, TT, Z, A)


########################################
# 4. 推論用の関数を分割して定義
########################################

def generate_next_zone_logits(
                            context_tokens, 
                            model, 
                            time_batch, loc_batch, act_batch, disc_feats, 
                            time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feats):
    """
    モデルから次に出力するトークンのlogitsを取り出す関数。
    """
    for name, tok in tokenizer.time_SPECIAL_TOKENS.items():
        print('ssssss')
        print(name, tok, "freq_in_teacher=",
            (time_batch == tok).float().mean().item())
        
    time_output, loc_output, act_output = model(
                                                context_tokens,
                                                time_batch, loc_batch, act_batch, disc_feats, # -> encoder # 
                                                time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feats) # -> decoder

    cur_time = time_traveled_route[:, -1]  # shape: (batch_size,)
    print('rrrrrrshape', time_output.shape, loc_output.shape, act_output.shape) # rrrrrrshape torch.Size([256, 1, 53]) torch.Size([256, 1, 271]) torch.Size([256, 1, 13])

    '''
    # マスクを作成: 自分より小さい時間をすべてマスク（-infをかける）
    mask = torch.arange(time_vocab_size, device=device).unsqueeze(0) < cur_time.unsqueeze(1)  # shape: (batch_size, time_vocab_size)
    last_time_logits[mask] = float('-inf')
    '''
    # モデルの特殊トークン辞書を使って <m> を取得
    m_time = tokenizer.time_SPECIAL_TOKENS["<m>"]
    m_loc = tokenizer.loc_SPECIAL_TOKENS["<m>"]
    m_act = tokenizer.act_SPECIAL_TOKENS["<m>"]

    # print('@@@@@@@@@@@@@inside generate@@@@@@@@@@@@@@')

    # それぞれ logits の <m> トークンのスコアを -inf にして、選ばれないようにする
    time_output[:, -1, m_time] = float('-inf')
    loc_output[:, -1, m_loc] = float('-inf')
    act_output[:, -1, m_act] = float('-inf')

    last_time_logits = time_output[:, -1, :]  # time_output から最後のステップのlogitsのみ取得 # softmaxに入れる値が入ってる# shape: (batch_size, time_vocab_size)
    
    # マスク：現在時刻までを禁止（前進のみに制限）
    if time_output.shape[1] != 1:
        time_range = torch.arange(time_vocab_size, device=device).unsqueeze(0)
        mask = time_range <= cur_time.unsqueeze(1)
        last_time_logits = last_time_logits.masked_fill(mask, float('-inf'))

        # fallback: 全部 -inf の場合はゼロベクトルにしてsoftmaxエラー回避
        all_inf_mask = torch.isinf(last_time_logits).all(dim=-1)
        last_time_logits[all_inf_mask] = torch.zeros_like(last_time_logits[all_inf_mask])
        time_output[:, -1, :] = last_time_logits
    
    # print('last time logit', last_time_logits[:, tokenizer.time_SPECIAL_TOKENS["<e>"]], last_time_logits.shape) # last time logit tensor([0.], device='cuda:0') torch.Size([256, 53])
    # sys.exit()

    return time_output[:, -1, :], loc_output[:, -1, :], act_output[:, -1, :]  # sequenceの最後のステップの出力のみ返す # 8はemb-dimなので保存

def apply_neighbor_mask(logits, neighbor, newest_zone, d_tensor):
    """
    neighbor マスクを生成して logits に加算する。
    """
    neighbor_mask = neighbor.make_neighbor_mask(newest_zone, d_tensor).to(device)
    # マスクを加算
    masked_logits = logits + neighbor_mask
    return masked_logits

def sample_next_zone(masked_logits): # softmax のサンプリング
    """
    softmaxしてトークンをサンプリングする（multinomial）。
    """
    # 数値安定化
    masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
    output_softmax = F.softmax(masked_logits, dim=-1)
    next_zone = torch.multinomial(output_softmax, num_samples=1).squeeze(-1) # 確率分布に従ってトークンをランダムにサンプリング(確定的ではない！！)
    return next_zone

# これまで通った経路＋次のゾーン→結合したい
def update_traveled_route(tokenizer, time_traveled_route, loc_traveled_route, act_traveled_route, time_next_zone, loc_next_zone, act_next_zone): #, time_batch, img_dic, time_is_day):
    """
    traveled_route に next_zone を追加し、特徴行列 (features) も更新する。
    """
    l1  = act_traveled_route.shape
    time_traveled_route = torch.cat([time_traveled_route, time_next_zone.unsqueeze(1)], dim=1)
    act_traveled_route = torch.cat([act_traveled_route, act_next_zone.unsqueeze(1)], dim=1)
    loc_traveled_route = torch.cat([loc_traveled_route, loc_next_zone.unsqueeze(1)], dim=1)

    traveled_feature_mat = tokenizer.make_feature_mat(loc_traveled_route).to(device)
    l2 = act_traveled_route.shape

    return time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feature_mat


####### 重要 ########
def check_and_save_completed_routes( 
    context_tokens,# 未終了のバッチのためのコンテキスト
    context_save_route, # 終了済みのバッチのためのコンテキスト
    time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feats, l_max, tokenizer, idx_lis,
    time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
    time_batch, loc_batch, act_batch, disc_feats, 
    time_save_route, loc_save_route, act_save_route,
    time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_route,
    time_clone, loc_clone, act_clone, context_clone,
    global_indices
    ):

    context_tokens_original = context_tokens.clone()
    time_tokens_original = time_batch[:, :-1].clone()
    loc_tokens_original = loc_batch[:, :-1].clone()
    act_tokens_original = act_batch[:, :-1].clone()
    """
    最新トークンが <e> のものを最終的な出力として保存し、バッチから除去して返す
    """
    # 終了トークンが出たサンプルを取得
    time_true_indices = torch.where(time_traveled_route[:, -1] == tokenizer.time_SPECIAL_TOKENS["<e>"])[0]
    act_true_indices = torch.where(act_traveled_route[:, -1] == tokenizer.act_SPECIAL_TOKENS["<e>"])[0]
    loc_true_indices = torch.where(loc_traveled_route[:, -1] == tokenizer.loc_SPECIAL_TOKENS["<e>"])[0]

    # 1. 各系列の末尾が <e> かどうか
    # <e>トークンが生成されなかったものは999もしくは0000のままになっている
    time_is_end = time_traveled_route[:, -1] == tokenizer.time_SPECIAL_TOKENS["<e>"]
    act_is_end = act_traveled_route[:, -1] == tokenizer.act_SPECIAL_TOKENS["<e>"]
    loc_is_end = loc_traveled_route[:, -1] == tokenizer.loc_SPECIAL_TOKENS["<e>"]

    # 2. どれか1つでも <e> が出ていれば True
    any_is_end = time_is_end | act_is_end | loc_is_end

    # 3. <e> が出ているサンプルのインデックス
    end_indices = torch.where(any_is_end)[0]

    if len(end_indices) > 0:
        # 4. 強制的に他の系列も <e> にする 
        for i in end_indices:
            if not time_is_end[i]:
                time_traveled_route[i, -1] = tokenizer.time_SPECIAL_TOKENS["<e>"]
            if not act_is_end[i]:
                act_traveled_route[i, -1] = tokenizer.act_SPECIAL_TOKENS["<e>"]
            if not loc_is_end[i]:
                loc_traveled_route[i, -1] = tokenizer.loc_SPECIAL_TOKENS["<e>"]
            

        # 5. パディングして保存
        '''
        act_padded_routes = torch.nn.functional.pad(
            act_traveled_route[end_indices], (0, l_max - act_traveled_route.size(1) - 1), # +1?????
            value=tokenizer.act_SPECIAL_TOKENS["<p>"]
        )
        loc_padded_routes = torch.nn.functional.pad(
            loc_traveled_route[end_indices], (0, l_max - loc_traveled_route.size(1) - 1),
            value=tokenizer.loc_SPECIAL_TOKENS["<p>"]
        )
        time_padded_routes = torch.nn.functional.pad(
            time_traveled_route[end_indices], (0, l_max - time_traveled_route.size(1) - 1),
            value=tokenizer.time_SPECIAL_TOKENS["<p>"]
        )
        '''
        act_padded_routes = torch.nn.functional.pad(
            act_traveled_route[end_indices], (0, l_max - act_traveled_route.size(1) + 1), # +1?????
            value=tokenizer.act_SPECIAL_TOKENS["<p>"]
        )
        loc_padded_routes = torch.nn.functional.pad(
            loc_traveled_route[end_indices], (0, l_max - loc_traveled_route.size(1) + 1),
            value=tokenizer.loc_SPECIAL_TOKENS["<p>"]
        )
        time_padded_routes = torch.nn.functional.pad(
            time_traveled_route[end_indices], (0, l_max - time_traveled_route.size(1) + 1),
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

        print('true_global_indices', true_global_indices.shape) 
        print('time_teacher_route', time_teacher_route.shape)
        print('time clone', time_clone.shape) 

        print('time save route', time_save_route.shape, 'time padded route', time_padded_routes.shape)

        time_save_route[true_global_indices] = time_padded_routes # 推論結果
        act_save_route[true_global_indices] = act_padded_routes
        loc_save_route[true_global_indices] = loc_padded_routes
        context_save_route[true_global_indices] = context_tokens_original[end_indices]

        time_teacher_route[true_global_indices] = time_tokens_original[end_indices]
        loc_teacher_route[true_global_indices] = loc_tokens_original[end_indices]
        act_teacher_route[true_global_indices] = act_tokens_original[end_indices]
        context_teacher_route[true_global_indices] = context_clone[end_indices]
        # save ファイルは256分ある，だんだん完成版で埋まっていく感じ

        # 6. マスクで除去
        mask_del = torch.ones(time_traveled_route.size(0), dtype=torch.bool, device=device)
        mask_del[end_indices] = False
        # mask_delは終了してないところだけTrue→終了してないところが残る

        # 7. バッチ更新（以下元のコードと同様）
        time_traveled_route = time_traveled_route[mask_del]
        act_traveled_route = act_traveled_route[mask_del]
        loc_traveled_route = loc_traveled_route[mask_del]
        traveled_feats = traveled_feats[mask_del]
        idx_lis = idx_lis[mask_del]

        time_infer_start_indices = time_infer_start_indices[mask_del]
        act_infer_start_indices = act_infer_start_indices[mask_del]
        loc_infer_start_indices = loc_infer_start_indices[mask_del]
        time_batch = time_batch[mask_del]
        act_batch = act_batch[mask_del]
        loc_batch = loc_batch[mask_del]
        context_tokens = context_tokens[mask_del]
        disc_feats = disc_feats[mask_del]
    
    '''
        # actとlocの両方が終了しているサンプルのみを対象にする
    common_true_indices = torch.tensor(
        list(
            set(act_true_indices.tolist())
            & set(loc_true_indices.tolist())
            & set(time_true_indices.tolist())
        ),
        dtype=torch.long,
        device=act_traveled_route.device
    )

    # min_end_indices = min(time_true_indices, loc_true_indices, act_true_indices)
    # print(min_end_indices)

    # sys.exit()


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
        # print('l_max - act_traveled_route.size(1)', l_max - act_traveled_route.size(1))
        act_padded_routes = torch.nn.functional.pad( # act_traveled_route: size 1 から始まる 19から始まる # (left, right)で左側にパディングする数，右側にパディングする数
            act_traveled_route[common_true_indices], (0, l_max - act_traveled_route.size(1)-1), value=tokenizer.act_SPECIAL_TOKENS["<p>"]
        ) # 長さは19 
        loc_padded_routes = torch.nn.functional.pad(
            loc_traveled_route[common_true_indices], (0, l_max - loc_traveled_route.size(1)-1), value=tokenizer.loc_SPECIAL_TOKENS["<p>"]
        )
        time_padded_routes = torch.nn.functional.pad(
            time_traveled_route[common_true_indices], (0, l_max - time_traveled_route.size(1)-1), value=tokenizer.time_SPECIAL_TOKENS["<p>"]
        )

        # save_route の該当箇所に書き込む
        # idx = idx_lis[true_indices] 
        actloc_idx = idx_lis[common_true_indices]
        # loc_idx = loc_idx_lis[common_true_indices]
        # print('act_padded_routes', act_padded_routes.shape) # 64*20
        time_save_route[actloc_idx] = time_padded_routes
        act_save_route[actloc_idx] = act_padded_routes
        loc_save_route[actloc_idx] = loc_padded_routes

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
        time_traveled_route = time_traveled_route[mask_del]
        act_traveled_route = act_traveled_route[mask_del]
        loc_traveled_route = loc_traveled_route[mask_del]

        # idx_lis = idx_lis[mask_del]
        idx_lis = idx_lis[mask_del]
        # loc_idx_lis = loc_idx_lis[mask_del]
        # time_is_day = time_is_day[mask_del]
        # d_tensor = d_tensor[mask_del]
        time_d_tensor = time_d_tensor[mask_del]
        act_d_tensor = act_d_tensor[mask_del]
        loc_d_tensor = loc_d_tensor[mask_del]

        # infer_start_indices = infer_start_indices[mask_del]
        time_infer_start_indices = time_infer_start_indices[mask_del]
        act_infer_start_indices = act_infer_start_indices[mask_del]
        loc_infer_start_indices = loc_infer_start_indices[mask_del]
        
        # disc_tokens = disc_tokens[mask_del]
        time_batch = time_batch[mask_del]
        act_batch = act_batch[mask_del]
        loc_batch = loc_batch[mask_del]
        disc_feats = disc_feats[mask_del]

        # traveled_route = traveled_route[mask_del]
        # act_traveled_route = act_traveled_route[act_mask_del]
        # loc_traveled_route = loc_traveled_route[loc_mask_del]
        traveled_feats = traveled_feats[mask_del]
    '''
    return (
        context_tokens,
        context_save_route,
        time_traveled_route,loc_traveled_route,act_traveled_route,traveled_feats,idx_lis,
        # time_d_tensor, loc_d_tensor, act_d_tensor,
        time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
        time_batch, loc_batch, act_batch, disc_feats, time_save_route, loc_save_route, act_save_route,
        time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_route,
        global_indices
    ) 


########################################
# 5. 推論の実行（メイン部分）
########################################

def run_inference(test_loader, model, tokenizer, l_max):
    """
    実際にtest_loaderからバッチを読み出し、推論を行うメイン関数。
    """
    # ignore_value_list = [tokenizer.SPECIAL_TOKENS["<p>"], tokenizer.SPECIAL_TOKENS["<m>"]]
    time_all_results = []
    loc_all_results = []
    act_all_results = []
    context_all_results = []

    time_all_teacher = []
    loc_all_teacher = []
    act_all_teacher = []
    context_all_teacher = []

    # global indicesの初期化
    global_indices = torch.arange(batch_size, device = device)

    print('len(loder):', len(test_loader))
    batch_counter = 0
    for time_batch, loc_batch, act_batch, context_batch in test_loader:
        batch_counter += 1

        if batch_counter == 40:
            break

        # # time batchのうち<e>の位置を取得
        # time_batch_end = time_batch[:, ]

        time_clone = time_batch[:, :-1].clone()
        loc_clone = loc_batch[:, :-1].clone()
        act_clone = act_batch[:, :-1].clone()
        context_clone = context_batch.clone()

        time_batch = time_batch.to(device)
        act_batch = act_batch.to(device)
        loc_batch = loc_batch.to(device)
        context_tokens = context_batch.to(device)

        print(f'------------ test batch :: {batch_counter}th batch  ------------')

        # 終了トークン <e> の直前のトークン（=目的地）を d_tensor として取得　# inputデータはbeginやendの処理がされてないのでここでする（completeにする）
        time_tokens, loc_tokens, act_tokens = tokenizer.tokenization(time_batch, loc_batch, act_batch, mode = "complete")

        time_end_indices = (time_tokens == tokenizer.time_SPECIAL_TOKENS["<e>"]).float().argmax(dim=1)
        act_end_indices = (act_tokens == tokenizer.act_SPECIAL_TOKENS["<e>"]).float().argmax(dim=1)
        loc_end_indices = (loc_tokens == tokenizer.loc_SPECIAL_TOKENS["<e>"]).float().argmax(dim=1)
        
        time_begin_indices = (time_tokens == tokenizer.time_SPECIAL_TOKENS["<b>"]).float().argmax(dim=1)
        act_begin_indices = (act_tokens == tokenizer.act_SPECIAL_TOKENS["<b>"]).float().argmax(dim=1)
        loc_begin_indices = (loc_tokens == tokenizer.loc_SPECIAL_TOKENS["<b>"]).float().argmax(dim=1)        

        time_infer_start_indices = time_begin_indices + 1 # 全部1
        act_infer_start_indices = act_begin_indices + 1
        loc_infer_start_indices = loc_begin_indices + 1
        disc_feature_mat = tokenizer.make_feature_mat(loc_tokens).to(device) ## ここ直したけど

        # 推論中のルートを初期化 # 最初は開始トークンになるのでbを入れる
        time_traveled_route = torch.full((batch_size, 1), tokenizer.time_SPECIAL_TOKENS["<b>"], dtype=torch.long).to(device)
        act_traveled_route = torch.full((batch_size, 1), tokenizer.act_SPECIAL_TOKENS["<b>"], dtype=torch.long).to(device)
        loc_traveled_route = torch.full((batch_size, 1), tokenizer.loc_SPECIAL_TOKENS["<b>"], dtype=torch.long).to(device)
        traveled_feats = tokenizer.make_feature_mat(loc_traveled_route).to(device) # paddingに対応した特徴量ベクトルになる
        print('traveled_feats', traveled_feats.shape, 'time traveled route', time_traveled_route.shape) # traveled_feats torch.Size([256, 1, 271]) time traveled route torch.Size([256, 1])
        # 出力を保存する変数 (l_maxに合わせたサイズに最終的に揃える) # 64*18なので元のバッチの形状と同じ
        time_save_route = torch.zeros((batch_size, l_max+1), dtype=torch.long).to(device) # lmaxが21なので20列になる
        act_save_route = torch.zeros((batch_size, l_max+1), dtype=torch.long).to(device) 
        loc_save_route = torch.zeros((batch_size, l_max+1), dtype=torch.long).to(device)
        context_save_tokens = torch.full((context_tokens.shape[0], context_tokens.shape[1]), fill_value= 999, dtype=torch.long).to(device) # contextは最初999で埋める

        # 教師データの保存
        time_teacher_route = torch.zeros((batch_size, l_max+1), dtype=torch.long).to(device) # lmaxが21なので20列になる
        loc_teacher_route = torch.zeros((batch_size, l_max+1), dtype=torch.long).to(device)
        act_teacher_route = torch.zeros((batch_size, l_max+1), dtype=torch.long).to(device)
        
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

        i = 0
        while i <= l_max - 3: # timestep+2(前後トークンのはず)→変？
            print(f'---- inference step at {i} ----')

            time_next_zone_logits, loc_next_zone_logits, act_next_zone_logits = generate_next_zone_logits(
                                                                                    context_tokens, # 終了した分は削除されてる
                                                                                    model, 
                                                                                    # ''' もしかしてここ・？？？？？'''
                                                                                    # time_batch, loc_batch, act_batch, disc_feature_mat, ## 
                                                                                    time_tokens, loc_tokens, act_tokens, disc_feature_mat,
                                                                                    time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feats
                                                                                )

            # softmax + サンプリング 
            time_next_zone = sample_next_zone(time_next_zone_logits) 
            act_next_zone = sample_next_zone(act_next_zone_logits) 
            loc_next_zone = sample_next_zone(loc_next_zone_logits)

            # traveled_route の更新 # 次のsoftmax最大のトークンを用いてupdate
            l1 = act_traveled_route.shape
            time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feats = update_traveled_route(
                tokenizer,
                time_traveled_route, loc_traveled_route, act_traveled_route,
                time_next_zone, loc_next_zone, act_next_zone
            )
            l2 = act_traveled_route.shape

            ####### 重要 ####### 終了トークン <e> を出したサンプルを確認して保存＆削除
            (
             context_tokens, 
             context_save_tokens, # context_padded,
             time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feats, idx_lis,
             time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
             #  time_batch, loc_batch, act_batch, disc_feature_mat,
             time_tokens, loc_tokens, act_tokens, disc_feature_mat,
             time_save_route, loc_save_route, act_save_route,
             time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_tokens,
             global_indices # 他のtraveled routeなどと同様に入力を更新して出力
             ) = check_and_save_completed_routes(
                context_tokens, 
                context_save_tokens,# context_padded,
                time_traveled_route, loc_traveled_route, act_traveled_route, traveled_feats, l_max, tokenizer, idx_lis, # loc_idx_lis,
                time_infer_start_indices, loc_infer_start_indices, act_infer_start_indices,
                # time_batch, loc_batch, act_batch, disc_feature_mat, 
                time_tokens, loc_tokens, act_tokens, disc_feature_mat,
                time_save_route, loc_save_route, act_save_route,
                time_teacher_route, loc_teacher_route, act_teacher_route, context_teacher_tokens,
                time_clone, loc_clone, act_clone, context_clone,
                global_indices
            )

            # バッチ内サンプルがなくなったら終了
            if act_traveled_route.size(0) == 0:
                print("All samples in this batch have finished.", loc_traveled_route.size(0), 'これも0のはず')
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

    return time_final_result, loc_final_result, act_final_result, context_batch, context_final_result, time_teacher_final_result, loc_teacher_final_result, act_teacher_final_result, context_teacher_final_result


########################################
# 6. 実際に推論を実行して結果を保存
########################################

# 推論実行
# dddprint('------ inference start ------') # kokomade ha kita
time_result, loc_result, act_result, context_batch, context_result, time_teacher_result, loc_teacher_result, act_teacher_result, context_teacher_result = run_inference(test_loader, model, tokenizer, l_max)

# CSVへ保存
time_result_df = pd.DataFrame(time_result.cpu().numpy())
act_result_df = pd.DataFrame(act_result.cpu().numpy())
loc_result_df = pd.DataFrame(loc_result.cpu().numpy())
context_df = pd.DataFrame(context_result.cpu().numpy())

time_teacher_result_df = pd.DataFrame(time_teacher_result.cpu().numpy())
act_teacher_result_df = pd.DataFrame(act_teacher_result.cpu().numpy())
loc_teacher_result_df = pd.DataFrame(loc_teacher_result.cpu().numpy())
context_teacher_df = pd.DataFrame(context_teacher_result.cpu().numpy())

time_result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411time_inference_result_crossmodalatt_model3.csv')
act_result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411act_inference_result_crossmodalatt_model3.csv')
loc_result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411loc_inference_result_crossmodalatt_model3.csv')
context_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411context_inference_result_crossmodalatt_model3.csv')
# result_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/inference_result.csv')
print("推論結果を保存しました！")

# time_teacher_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411time_teacher_inference_result_crossmodalatt.csv')
# act_teacher_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411act_teacher_inference_result_crossmodalatt.csv')
# loc_teacher_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411loc_teacher_inference_result_crossmodalatt.csv')
# context_teacher_df.to_csv('/Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/output/0411context_teacher_inference_result_crossmodalatt.csv')



'''
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
'''