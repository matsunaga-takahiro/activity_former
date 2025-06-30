import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn as nn

from ActFormer.RoutesFormer.simple_routeformer.toysrc.network import Network
from ActFormer.RoutesFormer.simple_routeformer.toysrc.tokenization import Tokenization
from ActFormer.RoutesFormer.simple_routeformer.toysrc.routesformer import Routesformer
from utils.logger import logger
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, random_split, DataLoader, Subset
import torch.nn.functional as F
import os
from datetime import datetime, timedelta
import pickle
from captum.attr import GradientShap

torch.cuda.init()
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=torch.device('cuda'), abbreviated=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################
# 1:input dataの作成
############################################


#データ読み込み
adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)
node_features = torch.load("/mnt/okinawa/9月BLEデータ/route_input/network/node_features_matrix.pt", weights_only=True)
trip_arrz = np.load('/mnt/okinawa/9月BLEデータ/route_input/reduced_route_input_0928_all.npz') #GT complete training path set
trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']

# 時刻に応じたVAE入力データをロード（例として抜粋）
start_time = datetime(2024, 9, 28, 10, 0, 0)
end_time = datetime(2024, 9, 28, 15, 0, 0)
current_time = start_time
time_lis = []
while current_time < end_time:
    time_str = current_time.strftime("%Y%m%d%H")
    time_lis.append(int(time_str))
    current_time += timedelta(hours=1)

start_time = datetime(2024, 9, 28, 18, 0, 0)
end_time = datetime(2024, 9, 29, 2, 0, 0)
current_time = start_time
while current_time < end_time:
    time_str = current_time.strftime("%Y%m%d%H")
    time_lis.append(int(time_str))
    current_time += timedelta(hours=1)
print(time_lis)

img_dic = {int(time * 100): torch.load(f"/mnt/okinawa/camera/VAE_input_1to1/{time}.pt") for time in time_lis}


#前処理に必要なパラメータを設定
timestep = len(trip_arr[0])
network = Network(adj_matrix, node_features)
route = torch.from_numpy(trip_arr)
time_pt = torch.from_numpy(time_arr)
vocab_size = network.N + 4
feature_dim = network.node_features.shape[1] + 1
tokenizer = Tokenization(network)


#データセットの作成
class MyDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]

#データセットの分割(データを読み込んで固定する)
dataset = MyDataset(route, time_pt)
num_samples = route.shape[0]
with open("/home/kurasawa/master_code/transformer/train_val_indices.pkl", "rb") as file:
    loaded_arrays = pickle.load(file)

train_indices = np.hstack([loaded_arrays[0], loaded_arrays[1], loaded_arrays[2], loaded_arrays[3]])
val_indices = loaded_arrays[4]

train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)

#データローダを作成
#SHAP計算のためにトレーニングデータは100サンプル
#テストデータは50サンプル取得する

train_batch = 100
val_batch = 30

train_loader = DataLoader(dataset=train_data,
                          batch_size=train_batch,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)
val_loader = DataLoader(dataset=val_data,
                        batch_size=val_batch,
                        shuffle=True,
                        num_workers=4,
                        drop_last=True)

#############################################
# 2:modelの作成
#############################################

#モデルをインポートする
#ハイパーパラメータの取得
api = wandb.Api()
#run = api.run("tkwnmdr-utokyo/RoutesFormer_test/yngqjjsg")#画像なしの場合
run = api.run("tkwnmdr-utokyo/RoutesFormer_test/7sxark22")#画像ありの場合
#run = api.run("tkwnmdr-utokyo/RoutesFormer_test/t5wt1reu")#動画の場合
print(f"Run ID: {run.id}, Run Name: {run.name}")

config = run.config
#RoutesFormerのハイパーパラメータ
batch_size = config['batch_size']
l_max = config['l_max'] #シークエンスの最大長さ
B_en = config['B_en'] #エンコーダのブロック数
B_de = config['B_de'] #デコーダのブロック数
head_num = config['head_num'] #ヘッド数
d_ie = config['d_ie'] #トークンの埋め込み次元数
d_fe = config['d_fe'] #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
d_ff = config['d_ff'] #フィードフォワード次元数
z_dim = config['z_dim'] #潜在変数の次元数
l_max = 62
print(d_ie, d_fe)

model = Routesformer(enc_vocab_size= vocab_size,
                            dec_vocab_size = vocab_size,
                            token_emb_dim = d_ie,
                            feature_dim = feature_dim + z_dim,
                            feature_emb_dim = d_fe,
                            d_ff = d_ff,
                            head_num = head_num,
                            B_en = B_en,
                            B_de = B_de).to(device)

#model_weights_path = "model_weights_all_4.pth" #画像なしの場合
model_weights_path = "model_weights_withfigure_all_1to1_4.pth" #画像ありの場合
#model_weights_path = "model_3DCNN.pth" #動画の場合

loadfile = torch.load(model_weights_path)
model.load_state_dict(loadfile['model_state_dict'])
model.eval()


####################
# 3:SHAPの計算の準備
####################


# 前処理関数の定義
def preprocess_data(route_batch, time_batch):
    time_is_day = (time_batch < 202409281500).to(device)
    discontinuous_route_tokens = tokenizer.tokenization(route_batch, mode="discontinuous").long().to(device)
    discontinuous_feature_mat = tokenizer.make_feature_mat(discontinuous_route_tokens).to(device)
    time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(discontinuous_feature_mat.shape[0], discontinuous_feature_mat.shape[1], 1)
    discontinuous_VAE_mat = tokenizer.make_VAE_input(discontinuous_route_tokens, time_batch, img_dic).to(device)
    discontinuous_feature_mat = torch.cat((discontinuous_feature_mat, time_feature_mat), dim=2)
    complete_route_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
    complete_feature_mat = tokenizer.make_feature_mat(complete_route_tokens).to(device)
    time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(complete_feature_mat.shape[0], complete_feature_mat.shape[1], 1)
    complete_VAE_mat = tokenizer.make_VAE_input(complete_route_tokens, time_batch, img_dic).to(device)
    complete_feature_mat = torch.cat((complete_feature_mat, time_feature_mat), dim=2)
    next_route_tokens = tokenizer.tokenization(route_batch, mode = "next").long().to(device)

    return (discontinuous_route_tokens, discontinuous_feature_mat, discontinuous_VAE_mat, complete_route_tokens, complete_feature_mat, complete_VAE_mat, next_route_tokens)

# 1) SHAPを計算する際のforward関数を再定義
def forward_for_shap(
    discontinuous_VAE_mat,       # 1) inputs の最初
    complete_VAE_mat,            # 2) inputs の2番目
    discontinuous_route_tokens,  # 3) additional_forward_args の最初
    discontinuous_feature_mat,   # 4) additional_forward_args の2番目
    complete_route_tokens,       # 5) ...
    complete_feature_mat,        # 6)
    next_route_tokens            # 7)
):
    """
    SHAP用のフォワード関数
      - inputs に相当する (discontinuous_feature_mat, complete_feature_mat)
      - additional_forward_args に相当する (discontinuous_route_tokens, complete_route_tokens)
    """
    print(discontinuous_feature_mat.shape)
    print(discontinuous_VAE_mat.shape)
    discontinuous_feature_mat = torch.cat((discontinuous_feature_mat, discontinuous_VAE_mat), dim= -1)
    complete_feature_mat = torch.cat((complete_feature_mat, complete_VAE_mat), dim= -1)
    # モデルの出力 [batch_size, seq_len, vocab_size]
    outputs = model(
        discontinuous_route_tokens,        # route tokens は追加引数で固定
        discontinuous_feature_mat,         # ここは勾配を取りたい
        complete_route_tokens,             # route tokens は追加引数で固定
        complete_feature_mat               # ここは勾配を取りたい
    )

     # 2) log_softmax しておく [batch_size, seq_len, vocab_size]
    log_probs = F.log_softmax(outputs, dim=-1)
    
    # 3) 正解ラベル y_true に対応する log_prob を gather で取得
    #    y_true shape: [batch_size, seq_len]
    #    log_probs shape: [batch_size, seq_len, vocab_size]
    #    -> gather(dim=-1, index=y_true) したいので、unsqueeze(-1) が必要
    correct_log_probs = log_probs.gather(dim=-1, index=next_route_tokens.unsqueeze(-1)).squeeze(-1)
    # correct_log_probs shape: [batch_size, seq_len]
    
    # 4) シーケンス全体の対数確率を「各ステップのlog_probを合計」してスカラー化
    #    shape: [batch_size]
    seq_log_prob = correct_log_probs.sum(dim=1)
    
    return seq_log_prob


####################
# 4:アトリビューションの計算
####################

accum_discontinuous_feature_mat = []
accum_complete_feature_mat = []
accum_complete_route_tokens = []
accum_discontinuous_route_tokens = []
accum_time = []

for i, ((train_batch, train_time), (val_batch, val_time)) in enumerate(zip(train_loader, val_loader)):
    print(i)
    if i > 50:
        break

    #背景データの準備
    route_background = train_batch.to(device)  # (batch_size, seq_len)
    time_background = train_time.to(device) 
    background_inputs = preprocess_data(route_background, time_background)
    discontinuous_VAE_mat_baseline = background_inputs[2]
    complete_VAE_mat_baseline = background_inputs[5]


    route_val = val_batch.to(device)  # (batch_size, seq_len)
    time_val = val_time.to(device)    # (batch_size,)

    

    # テストデータの準備
    test_routes = route_val.to(device)  # (batch_size, seq_len)
    test_times = time_val.to(device)    # (batch_size,)

    # テストデータの前処理
    test_inputs = preprocess_data(test_routes, test_times)  # タプル
    #print(test_inputs)

    # 2) route tokens は additional_forward_args として渡し、feature_matのみを inputs として渡す
    discontinuous_route_tokens, discontinuous_feature_mat, discontinuous_VAE_mat, complete_route_tokens, complete_feature_mat, complete_VAE_mat, next_route_tokens = test_inputs

    # route tokens を固定 (勾配不要) にしたいなら requires_grad=False にしてもよい
    discontinuous_route_tokens.requires_grad_(False)
    complete_route_tokens.requires_grad_(False)
    discontinuous_feature_mat.requires_grad_(False)
    complete_feature_mat.requires_grad_(False)


    '''
    #ベースラインを0にする
    discontinuous_VAE_mat_baseline = torch.zeros_like(discontinuous_VAE_mat).to(device)
    complete_VAE_mat_baseline = torch.zeros_like(complete_VAE_mat).to(device)
    
    #ベースラインをノイズにする
    noize = torch.tensor([ 0.4694,  1.1881,  1.0939, -3.8492, -3.3292,  0.7185, -2.9811,  3.4711,
         -1.0711,  0.6311,  1.1561,  1.8994,  3.0611, -1.0981, -1.6463, -0.3577,
         -3.0078,  3.4375, -2.2085,  0.9386,  1.2634,  3.8274, -3.3683, -2.8018,
         -3.5401, -0.0980,  2.6051, -1.3093, -1.7096, -1.5132, -2.0302,  2.8370])
    discontinuous_VAE_mat_baseline = noize.repeat(discontinuous_VAE_mat.shape[0] * discontinuous_VAE_mat.shape[1], 1).reshape(discontinuous_VAE_mat.shape[0], discontinuous_VAE_mat.shape[1], 32).to(device)
    complete_VAE_mat_baseline = noize.repeat(complete_VAE_mat.shape[0] * complete_VAE_mat.shape[1], 1).reshape(complete_VAE_mat.shape[0], complete_VAE_mat.shape[1], 32).to(device)

    #黒画像をベースラインにする
    black_image = torch.tensor([-0.3224, -0.1259, -0.3761,  0.2854,  0.3394, -0.3418,  0.5425,  0.3221,
          0.5131, -0.3128, -0.3020, -0.4208, -0.4286,  0.3817,  0.4438, -0.0140,
          0.3344,  0.1181,  0.2354, -0.2112,  0.1314, -0.7063,  0.8605,  0.6001,
          0.1790,  0.0805, -0.0927, -0.0271,  0.7871, -0.0185, -0.0963, -0.3001])
    discontinuous_VAE_mat_baseline = black_image.repeat(discontinuous_VAE_mat.shape[0] * discontinuous_VAE_mat.shape[1], 1).reshape(discontinuous_VAE_mat.shape[0], discontinuous_VAE_mat.shape[1], 32).to(device)
    complete_VAE_mat_baseline = black_image.repeat(complete_VAE_mat.shape[0] * complete_VAE_mat.shape[1], 1).reshape(complete_VAE_mat.shape[0], complete_VAE_mat.shape[1], 32).to(device)
    '''


    # Gradient SHAPのインスタンスを作成
    gradient_shap = GradientShap(forward_for_shap)

    '''
    # 4) ターゲット（クラス）の指定
    with torch.no_grad():
        # もしターゲットクラスを argmax でとりたい場合
        outputs = model(
            discontinuous_route_tokens,
            discontinuous_feature_mat,
            complete_route_tokens,
            complete_feature_mat
        )
        probs = F.softmax(outputs, dim=-1)
        probs_sum = probs.sum(dim=1)  # [batch_size, vocab_size]
        preds = torch.argmax(probs_sum, dim=1)  # [batch_size]
    '''
    print(discontinuous_VAE_mat.shape)
    print(complete_VAE_mat.shape)
    print(discontinuous_VAE_mat_baseline.shape)

    # 5) アトリビューション計算
    #    inputs=(discontinuous_feature_mat, complete_feature_mat)
    #    baselines=(discontinuous_feature_mat_baseline, complete_feature_mat_baseline)
    #    additional_forward_args=(discontinuous_route_tokens, complete_route_tokens)
    attributions = gradient_shap.attribute(
        inputs=(discontinuous_VAE_mat, complete_VAE_mat),
        baselines=(discontinuous_VAE_mat_baseline, complete_VAE_mat_baseline),
        target=None,  # forward_for_shapが [batch_size] を返すので target=None でOK
        additional_forward_args=(discontinuous_route_tokens, discontinuous_feature_mat, complete_route_tokens, complete_feature_mat, next_route_tokens),
        n_samples=50
    )

    # attributions はタプル (attr_discontinuous_feature_mat, attr_complete_feature_mat) の形になる
    attr_discontinuous_feature_mat, attr_complete_feature_mat = attributions

    print(attr_discontinuous_feature_mat)
    print(attr_complete_feature_mat.shape)

    #GPUの負担を減らすためにnumpyにデータを移行
    attr_discontinuous_feature_mat = attr_discontinuous_feature_mat.cpu().detach().numpy()
    attr_complete_feature_mat = attr_complete_feature_mat.cpu().detach().numpy()
    complete_route_tokens = complete_route_tokens.cpu().detach().numpy()
    discontinuous_route_tokens = discontinuous_route_tokens.cpu().detach().numpy()
    time_val = time_val.cpu().detach().numpy()

    accum_discontinuous_feature_mat.append(attr_discontinuous_feature_mat)
    accum_complete_feature_mat.append(attr_complete_feature_mat)
    accum_complete_route_tokens.append(complete_route_tokens)
    accum_discontinuous_route_tokens.append(discontinuous_route_tokens)
    accum_time.append(time_val)



# 6) アトリビューションの保存
accum_discontinuous_feature_mat = np.concatenate(accum_discontinuous_feature_mat, axis=0)
accum_complete_feature_mat = np.concatenate(accum_complete_feature_mat, axis=0)
accum_complete_route_tokens = np.concatenate(accum_complete_route_tokens, axis=0)
accum_discontinuous_route_tokens = np.concatenate(accum_discontinuous_route_tokens, axis=0)
accum_time = np.concatenate(accum_time, axis=0)

np.savez("result/attr_fig_all_ave5.npz", attr_discontinuous_feature_mat=accum_discontinuous_feature_mat, attr_complete_feature_mat=accum_complete_feature_mat,
complete_route_tokens = accum_complete_route_tokens, discontinuous_route_tokens = accum_discontinuous_route_tokens, time = accum_time)
