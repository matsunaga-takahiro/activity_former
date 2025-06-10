# toy activity dataを使った学習実験用

import torch
import pandas as pd
import numpy as np
from torch import nn
from network_toyact import Network
from tokenization_toyact import Tokenization
# from ActFormer.RoutesFormer.toyact_src.actformer_toyact import Actformer
from actformer_toyact import Actformer
# from utils.logger import logger
import matplotlib.pyplot as plt
# import wandb
from torch.utils.data import Dataset, random_split, DataLoader, Subset
import torch.nn.functional as F
import os
import json
import pickle
import sys
import networkx as nx

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="RoutesFormer_toydata",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.0001,
#     "architecture": "Normal",
#     "dataset": "0928_day_night",
#     "epochs": 10,
#     "batch_size": 256,
#     "l_max" : 14 + 2,
#     "B_en" : 6,
#     "B_de" : 6,
#     "head_num" : 4,
#     "d_ie" : 22,
#     "d_fe" : 2,
#     "d_ff" : 32,
#     "eos_weight" : 3.0,
#     "stay_weight" : 1,
#     "savefilename": "toydata.pth"
#     }
# )

base_path = '/Users/matsunagatakahiro/Desktop/res2025/ActFormer'

# adj_matrix = torch.load(os.path.join(base_path, 'toy_data_generation/grid_adjacency_matrix.pt'), weights_only=True)
df_node = pd.read_csv(os.path.join(base_path, 'toyact_gen/input/node.csv'), index_col= 0)
df_node = df_node.iloc[:, 1:] # nodeidの列を削除
# node_features = torch.load(os.path.join(base_path, 'toy_data_generation/node_features_matrix.pt'), weights_only=True)
node_features_np = df_node.to_numpy()
node_features = torch.tensor(node_features_np, dtype=torch.float32) # torch.FloatTensor(node_features_np)

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

# print(adj_matrix_np.shape)
# print(node_features_np.shape)
# print(adj_matrix_np)
# print(node_features_np)

# # networkx で可視化
# # G = nx.from_numpy_matrix(adj_matrix_np)
# # G = nx.convert_matrix.from_numpy_matrix()
# G = nx.Graph()
# G.add_nodes_from(range(len(adj_matrix_np)))
# for i in range(len(adj_matrix_np)):
#     for j in range(len(adj_matrix_np)):
#         if adj_matrix_np[i][j] == 1:
#             G.add_edge(i, j)
# nx.draw(G, with_labels=True)
# plt.show()

# sys.exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 前処理
timestep = 19 # 24-6+1 = 19
network = Network(node_features)
route = torch.from_numpy(diary_arr)
loc_data = torch.from_numpy(loc_arr)
act_data = torch.from_numpy(act_arr)

### 状態数＋特殊トークン
A = 4 # activity数
N = 4 # node数
# print('no node', network.N)
SA = A * N # state*activityの同時分布数
act_vocab_size = A + 4 # network.N + 4 # node数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
loc_vocab_size = N + 4 # network.N + 4 # node数 + 4　経路予測やシーケンス学習で使うトークンの総数（状態数＋ダミーノード数など）：パディング，開始・終了，マスク用
feature_dim = network.node_features.shape[1] # 特徴量の次元数 


# 学習のハイパーパラメータ
num_epoch = 10 # wandb.config.epochs #エポック数
eta = 0.0001 # wandb.config.learning_rate #学習率
batch_size = 64 #wandb.config.batch_size

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
# mask_rate = wandb.config.mask_rate #マスク率

# class MyDataset(Dataset):
#     def __init__(self, data1):
#         self.data1 = data1

#     def __len__(self):
#         return len(self.data1)

#     def __getitem__(self, idx):
#         return self.data1[idx]

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

# num_samples = route.shape[0]
num_samples = len(dataset) # = len(loc_dataset)
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# print('num_samples: ', num_samples)
'''
#データを保存したインデックスから呼び出したい場合のコード
with open("/home/kurasawa/master_code/transformer/train_val_indices.pkl", "rb") as file:
    loaded_arrays = pickle.load(file)

train_indices = np.hstack([loaded_arrays[0], loaded_arrays[1], loaded_arrays[3], loaded_arrays[4]])
val_indices = loaded_arrays[2]


train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)
'''

'''
#今後の利用のために使ったデータのインデックスを保存する
train_indices = train_data.indices
val_indices = val_data.indices

np.savez('/home/kurasawa/master_code/transformer/train_val_indices_day.npz', train_indices=train_indices, val_indices=val_indices)
'''

num_batches = num_samples // batch_size

# 3. DataLoaderでバッチ化
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# train_dataset = torch.utils.data.TensorDataset(act_train_data, loc_train_data)
# val_dataset = torch.utils.data.TensorDataset(act_val_data, loc_val_data)

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           #num_workers=4, ### うまくいかないときは消す or 1
#                           drop_last=True)

# val_loader = DataLoader(dataset=val_dataset,
#                         batch_size=batch_size,
#                         shuffle=False,
#                         #num_workers=4,
#                         drop_last=True)

# print("act train data size: ",len(act_train_data) * batch_size)   #train data size:  
# print("act train iteration number: ",len(act_train_data))   #train iteration number: 
# print("act val data size: ",len(act_val_data) * batch_size)   #val data size: 
# print("act val iteration number: ",len(act_val_data))   #val iteration number: 

# print("loc train data size: ",len(loc_train_data) * batch_size)   #train data size:
# print("loc train iteration number: ",len(loc_train_data))   #train iteration number:
# print("loc val data size: ",len(loc_val_data) * batch_size)   #val data size:   
# print("loc val iteration number: ",len(loc_val_data))   #val iteration number:

#モデル
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
                    B_de = B_de)
model = model.to(device)

#criterion
# criterion = nn.CrossEntropyLoss(ignore_index=network.N) 
# self.num_nodes = network.N
# self.SPECIAL_TOKENS = { # valueはトークンID（nodeid）
        #     "<p>": self.num_nodes,  # パディングトークン　### ここに対応する．padingを無視してCEを計算するということ．今paddingをなくすから関係ない
        #     "<e>": self.num_nodes + 1,  # 終了トークン
        #     "<b>": self.num_nodes + 2,  # 開始トークン
        #     "<m>": self.num_nodes + 3,  # 非隣接ノードトークン
        # }

criterion_act = nn.CrossEntropyLoss(ignore_index=A)
criterion_loc = nn.CrossEntropyLoss(ignore_index=N)

optimizer = torch.optim.Adam(model.parameters(), lr = eta) 

history = {"train_loss": [], "val_loss": []} 
for epoch in range(num_epoch): # 各エポックで学習と評価を繰り返す
    model.train()
    epoch_loss = 0
    num_batches = 0
    # for i, act_batch in enumerate(train_loader): # 各エポックでバッチごとに学習
    # for act_batch, loc_batch in train_loader: # 各エポックでバッチごとに学習
    #     print('len(act_batch): ', len(act_batch)) # 64
    #     print('len(loc_batch): ', len(loc_batch)) # 64
    
    for act_batch, loc_batch in train_loader:
        # print('act_batch type111: ', type(act_batch)) # tensorになってるここまでは
        # print('loc_batch type111: ', type(loc_batch)) # tensor
        # # route_batch = batch.to(device)
        # act_batch, loc_batch = batch
        tokenizer = Tokenization(network, A, N)

        # print('len(act_batch): ', len(act_batch)) # 64
        # print('len(loc_batch): ', len(loc_batch)) # 64

        #### トークナイゼーションの結果はまとめていいのか，分けたほうがいいのか？？？
        ##### 分けたほうが良さそう
        # discontinuous_route_tokens = tokenizer.tokenization(route_batch, mode = "discontinuous").long().to(device)
        ### encoderに入れる
        act_tokens, loc_tokens = tokenizer.tokenization(act_batch, loc_batch, mode = "discontinuous")
        act_discontinuous_route_tokens = act_tokens.long().to(device)
        loc_discontinuous_route_tokens = loc_tokens.long().to(device)
        # print('loc_discontinuous_route_tokens: ', loc_discontinuous_route_tokens.shape) # 64*20
        discontinuous_feature_mat = tokenizer.make_feature_mat(loc_discontinuous_route_tokens).to(device) # 特徴量はノードデータのみなのでloc_dataに対応させて読み込む
        # print('discontinuous_feature_matは回った')
        # complete_route_tokens = tokenizer.tokenization(route_batch, mode = "simple").long().to(device) # long: 変数形式の変換
        # decoderに入れる
        act_tokens2, loc_tokens2 = tokenizer.tokenization(act_batch, loc_batch, mode = "simple")
        act_complete_route_tokens = act_tokens2.long().to(device) # long: 変数形式の変換
        loc_complete_route_tokens = loc_tokens2.long().to(device) # long: 変数形式の変換
        # print('act_complete_route_tokens: ', act_complete_route_tokens.shape) # 64*20
        # print('loc_complete_route_tokens: ', loc_complete_route_tokens.shape) # 64*20
        complete_feature_mat = tokenizer.make_feature_mat(loc_complete_route_tokens).to(device)
        # print('complete_feature_matは回った')
        # next_route_tokens = tokenizer.tokenization(route_batch, mode = "next").long().to(device)
        # 正解データ（クロスエントロピー計算用）
        act_tokens3, loc_tokens3 = tokenizer.tokenization(act_batch, loc_batch, mode = "next")
        act_next_route_tokens = act_tokens3.long().to(device)
        loc_next_route_tokens = loc_tokens3.long().to(device)
        # print('act_next_route_tokens: ', act_next_route_tokens.shape) # 64*20
        #id = (complete_route_tokens == 10).nonzero(as_tuple=True)
        #print(complete_feature_mat[id[0], id[1], :])

        ## input for routesformer
        ## ここまできた
        # print('kokomadekita')
        act_output, loc_output = model(act_discontinuous_route_tokens, loc_discontinuous_route_tokens, discontinuous_feature_mat,  # for encoder
                                       act_complete_route_tokens, loc_complete_route_tokens, complete_feature_mat) # for decoder
        # print('actformerのforwardは回った')
        act_output_copy = act_output.clone()
        loc_output_copy = loc_output.clone()
        # outputs = outputs.view(-1, vocab_size)
        # print('act_output_copy: ', act_output_copy.shape) # 64*20*4
        # print("              act_output.shape:", act_output.shape) # 64-20-4        # → should be [B, T, vocab_size]
        # print("   act_next_route_tokens.shape:", act_next_route_tokens.shape) # 64-20 # → should be [B, T]
        # print(" act_next_route_tokens.numel():", act_next_route_tokens.numel()) # 1280
        # print("            act_output.numel():", act_output.shape[0] * act_output.shape[1]) # 1280
                        
        # print('                           act_vocab_size:', act_vocab_size) # 8 # 活動数+4
        # print("act_output.view(-1, act_vocab_size).shape", act_output.view(-1, act_vocab_size).shape) # 64*20-8
        # print("              act_next_route_tokens.shape", act_next_route_tokens.shape)
        # print(     "act_next_route_tokens.view(-1).shape", act_next_route_tokens.view(-1).shape)

        # loss = criterion(outputs, next_route_tokens.view(-1))
        loss_act = criterion_act(act_output.view(-1, act_vocab_size), act_next_route_tokens.view(-1))
        loss_loc = criterion_loc(loc_output.view(-1, loc_vocab_size), loc_next_route_tokens.view(-1))
        # print('loss_act: ', loss_act) # 0.0001
        # print('loss_loc: ', loss_loc) # 0.0001
        # 合成損失関数
        loss = loss_act + loss_loc
        print('test-loss: ', loss) # 0.0001
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
        for act_batch, loc_batch in val_loader:
            act_batch = act_batch.to(device)
            loc_batch = loc_batch.to(device)
            # route_batch = batch.to(device)
            tokenizer = Tokenization(network, A, N)
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
            act_tokens, loc_tokens = tokenizer.tokenization(act_batch, loc_batch, mode = "discontinuous")
            act_discontinuous_route_tokens = act_tokens.long().to(device)
            loc_discontinuous_route_tokens = loc_tokens.long().to(device)
            # print('loc_discontinuous_route_tokens: ', loc_discontinuous_route_tokens.shape) # 64*20
            discontinuous_feature_mat = tokenizer.make_feature_mat(loc_discontinuous_route_tokens).to(device) # 特徴量はノードデータのみなのでloc_dataに対応させて読み込む
            # print('discontinuous_feature_matは回った')
            # complete_route_tokens = tokenizer.tokenization(route_batch, mode = "simple").long().to(device) # long: 変数形式の変換
            # decoderに入れる
            act_tokens2, loc_tokens2 = tokenizer.tokenization(act_batch, loc_batch, mode = "simple")
            act_complete_route_tokens = act_tokens2.long().to(device) # long: 変数形式の変換
            loc_complete_route_tokens = loc_tokens2.long().to(device) # long: 変数形式の変換
            # print('act_complete_route_tokens: ', act_complete_route_tokens.shape) # 64*20
            # print('loc_complete_route_tokens: ', loc_complete_route_tokens.shape) # 64*20
            complete_feature_mat = tokenizer.make_feature_mat(loc_complete_route_tokens).to(device)
            # print('complete_feature_matは回った')
            # next_route_tokens = tokenizer.tokenization(route_batch, mode = "next").long().to(device)
            # 正解データ（クロスエントロピー計算用）
            act_tokens3, loc_tokens3 = tokenizer.tokenization(act_batch, loc_batch, mode = "next")
            act_next_route_tokens = act_tokens3.long().to(device)
            loc_next_route_tokens = loc_tokens3.long().to(device)
            # discontinuous_route_tokens = tokenizer.tokenization(route_batch, mode = "discontinuous").long().to(device)
            # discontinuous_feature_mat = tokenizer.make_feature_mat(discontinuous_route_tokens).to(device)
            # complete_route_tokens = tokenizer.tokenization(route_batch, mode = "simple").long().to(device)
            # complete_feature_mat = tokenizer.make_feature_mat(complete_route_tokens).to(device)
            # next_route_tokens = tokenizer.tokenization(route_batch, mode = "next").long().to(device)
            
            act_output, loc_output = model(act_discontinuous_route_tokens, loc_discontinuous_route_tokens, discontinuous_feature_mat,  # for encoder
                                        act_complete_route_tokens, loc_complete_route_tokens, complete_feature_mat) # for decoder

            # outputs = model(discontinuous_route_tokens, discontinuous_feature_mat, complete_route_tokens, complete_feature_mat)
            # outputs = outputs.view(-1, vocab_size)
            # loss = criterion(outputs, next_route_tokens.view(-1))
            
            # loss_act = criterion_act(act_output.view(-1, vocab_size), act_next_route_tokens.view(-1)) # viewの部分は，テンソル loc_output の形状を変換（reshape）
            # loss_loc = criterion_loc(loc_output.view(-1, vocab_size), loc_next_route_tokens.view(-1))
            # loss = loss_act + loss_loc
            loss_act = criterion_act(act_output.view(-1, act_vocab_size), act_next_route_tokens.view(-1))
            loss_loc = criterion_loc(loc_output.view(-1, loc_vocab_size), loc_next_route_tokens.view(-1))
            # print('val-loss_act: ', loss_act) # 0.0001
            # print('val-loss_loc: ', loss_loc) # 0.0001
            # 合成損失関数
            loss = loss_act + loss_loc
            print('val-loss: ', loss) # 0.0001
            epoch_loss += loss.item()
            num_batches += 1

    val_loss = epoch_loss / num_batches
    history["val_loss"].append(val_loss)
    # logger.info(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {val_loss:.4f}")
    # wandb.log({"total_loss": train_loss, "val_loss": val_loss})
# wandb.finish()

# モデルのパラメータを保存 (多くの行列やテンソルを含んでいるため)
save_data = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
}
torch.save(save_data, os.path.join(base_path, 'RoutesFormer/output', savefilename))
print("Model weights saved successfully") # /Users/matsunagatakahiro/Desktop/res2025/ActFormer/RoutesFormer/training_toydata.py