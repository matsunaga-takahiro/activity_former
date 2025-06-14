import torch
import pandas as pd
import numpy as np
from torch import nn

from ActFormer.RoutesFormer.simple_routeformer.toysrc.network import Network
from ActFormer.RoutesFormer.simple_routeformer.toysrc.tokenization import Tokenization
from ActFormer.RoutesFormer.simple_routeformer.toysrc.routesformer import Routesformer
from utils.logger import logger
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, random_split, DataLoader, Subset
import torch.nn.functional as F
import os
import json
import pickle

## 後で登録
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="RoutesFormer_test",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.0001,
#     "architecture": "Normal",
#     "dataset": "0928_day_night",
#     "epochs": 200,
#     "batch_size": 256,
#     "l_max" : 60 + 2,
#     "B_en" : 6,
#     "B_de" : 6,
#     "head_num" : 4,
#     "d_ie" : 23,
#     "d_fe" : 41,
#     "d_ff" : 32,
#     "eos_weight" : 3.0,
#     "stay_weight" : 1,
#     "savefilename": "model_weights_all_4_ordinary_2.pth"
#     }
# )

#input data
adj_matrix = torch.load('/adjacency_matrix.pt', weights_only=True)
node_features = torch.load("/node_features_matrix.pt", weights_only=True)
trip_arrz = np.load('/reduced_route_input_0928_all.npz') #GT complete training path set
trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(node_features.shape)
#前処理
timestep = len(trip_arr[0])
network = Network(adj_matrix, node_features)
route = torch.from_numpy(trip_arr)
time_pt = torch.from_numpy(time_arr)
vocab_size = network.N + 4
feature_dim = network.node_features.shape[1] + 1


#学習のハイパーパラメータ
num_epoch = 200 #  wandb.config.epochs #エポック数
eta = 0.0001 # wandb.config.learning_rate #学習率
batch_size = 256 # wandb.config.batch_size

#RoutesFormerのハイパーパラメータ
l_max = 60+2 # wandb.config.l_max #シークエンスの最大長さ
B_en = 6 # wandb.config.B_en #エンコーダのブロック数 ### Nx：ブロック数
B_de = 6 # wandb.config.B_de #デコーダのブロック数
head_num = 4 # wandb.config.head_num #ヘッド数
d_ie = 23 # wandb.config.d_ie #トークンの埋め込み次元数
d_fe = 41 # wandb.config.d_fe #特徴の埋め込み次元数 d_ie+d_feがhead_numで割り切れるように設定
d_ff = 32 # wandb.config.d_ff #フィードフォワード次元数
eos_weight = 3.0 # wandb.config.eos_weight #EOSトークンの重み
savefilename = "model_weights_all_4_ordinary_2.pth" # wandb.config.savefilename #モデルの保存ファイル名
stay_weight = 1 # wandb.config.stay_weight
#mask_rate = wandb.config.mask_rate #マスク率

class MyDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]


# バッチ化
dataset = MyDataset(route, time_pt)
num_samples = route.shape[0]
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])


with open(os.path.join(base_path, "/train_val_indices.pkl"), "rb") as file:
    loaded_arrays = pickle.load(file)

train_indices = np.hstack([loaded_arrays[0], loaded_arrays[1], loaded_arrays[3], loaded_arrays[4]])
val_indices = loaded_arrays[2]


train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)


#今後の利用のために使ったデータのインデックスを保存する
train_indices = train_data.indices
val_indices = val_data.indices

num_batches = num_samples // batch_size

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)
val_loader = DataLoader(dataset=val_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        drop_last=True)

print("train data size: ",len(train_data) * batch_size)   #train data size:  
print("train iteration number: ",len(train_data))   #train iteration number: 
print("val data size: ",len(val_data) * batch_size)   #val data size: 
print("val iteration number: ",len(val_data))   #val iteration number: 

#モデル
model = Routesformer(enc_vocab_size= vocab_size,
                            dec_vocab_size = vocab_size,
                            token_emb_dim = d_ie,
                            feature_dim = feature_dim,
                            feature_emb_dim = d_fe,
                            d_ff = d_ff,
                            head_num = head_num,
                            B_en = B_en,
                            B_de = B_de)
model = model.to(device)

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        """
        ラベルスムージング付きクロスエントロピー損失
        :param smoothing: ラベルスムージングの係数 (0~1)
        :param ignore_index: 無視するインデックス (例: paddingトークン)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """
        :param predictions: モデルの出力 (logits) [batch_size, num_classes, ...]
        :param targets: 正解ラベル [batch_size, ...]
        """
        num_classes = predictions.size(1)  # クラス数
        log_probs = F.log_softmax(predictions, dim=1)  # ソフトマックス後のlog値

        # ワンホットラベルのスムージング
        true_dist = torch.full_like(predictions, self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # ignore_indexを適用
        if self.ignore_index is not None:
            mask = targets == self.ignore_index
            true_dist = true_dist.masked_fill(mask.unsqueeze(1), 0)
            log_probs = log_probs.masked_fill(mask.unsqueeze(1), 0)

        # 損失計算
        loss = torch.sum(-true_dist * log_probs, dim=1)
        return loss.mean()

class WeightedCrossEntropyWithIgnoreIndex(nn.Module):
    def __init__(self, eos_token_id, eos_weight=2.0, 
                 ignore_index=-100,
                 stay_weight=0.5):  # <-- 新たに追加
        """
        CrossEntropyLoss with weighted EOS token loss, ignore_index, 
        and an optional stay_weight to handle 'stay' transitions.
        """
        super().__init__()
        self.eos_token_id = eos_token_id
        self.eos_weight = eos_weight
        self.ignore_index = ignore_index
        self.stay_weight = stay_weight  # stayの重み(小さめにする)

        # ベースのCEは reduction='none' で呼び出す
        self.base_loss_fn = nn.CrossEntropyLoss(
            reduction='none', 
            ignore_index=ignore_index
        )
        
    def forward(self, logits, targets, former_targets):
        """
        :param logits: [batch_size, seq_len, vocab_size]
        :param targets: [batch_size, seq_len]
        :param former_targets: [batch_size, seq_len]
        """
        # 形状変換
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

        # stay (とどまる) 判定: (target == former_target) かつ ignore_index でない
        stay_mask = (targets == former_targets) & (targets != self.ignore_index)
        # stay の重み (例: 0.5)
        weights[stay_mask] = self.stay_weight

        # 最終的な重み付き損失
        weighted_loss = (loss_per_token * weights).sum() / (weights[targets != self.ignore_index].sum())

        return weighted_loss

#criterion
#criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=network.N)
#criterion = nn.CrossEntropyLoss(ignore_index=network.N)
criterion = WeightedCrossEntropyWithIgnoreIndex(
    eos_token_id=network.N + 1,
    eos_weight=2.0,
    ignore_index=network.N,
    stay_weight=stay_weight
)
optimizer = torch.optim.Adam(model.parameters(), lr = eta)

history = {"train_loss": [], "val_loss": []} 
for epoch in range(num_epoch):
    model.train()
    epoch_loss = 0
    num_batches = 0
    for i,  (batch, time_batch) in enumerate(train_loader):
        route_batch = batch.to(device)
        time_batch = time_batch.to(device)
        time_is_day = (time_batch < 202409281500).to(device)
        tokenizer = Tokenization(network)
        discontinuous_route_tokens = tokenizer.tokenization(route_batch, mode = "discontinuous").long().to(device)
        discontinuous_feature_mat = tokenizer.make_feature_mat(discontinuous_route_tokens).to(device)
        time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(discontinuous_feature_mat.shape[0], discontinuous_feature_mat.shape[1], 1)
        discontinuous_feature_mat = torch.cat([discontinuous_feature_mat, time_feature_mat], dim=-1)
        complete_route_tokens = tokenizer.tokenization(route_batch, mode = "simple").long().to(device)
        complete_feature_mat = tokenizer.make_feature_mat(complete_route_tokens).to(device)
        time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(complete_feature_mat.shape[0], complete_feature_mat.shape[1], 1)
        complete_feature_mat = torch.cat([complete_feature_mat, time_feature_mat], dim=-1)
        next_route_tokens = tokenizer.tokenization(route_batch, mode = "next").long().to(device)
        #id = (complete_route_tokens == 10).nonzero(as_tuple=True)
        #print(complete_feature_mat[id[0], id[1], :])

        outputs = model(discontinuous_route_tokens, discontinuous_feature_mat, complete_route_tokens, complete_feature_mat)
        outputs_copy = outputs.clone()
        outputs = outputs.view(-1, vocab_size)
        loss = criterion(outputs, next_route_tokens.view(-1), complete_route_tokens.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 損失を累積
        epoch_loss += loss.item()
        num_batches += 1

        if (i + 1) % 10 == 0:  # 10バッチごとにログ
            logger.info(f"Epoch [{epoch+1}/{num_epoch}]")
            logger.info(f"  Loss: {loss.item():.4f}")
            logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"  Sample Prediction: {outputs_copy[0].argmax(dim=-1).tolist()}")
            logger.info(f"  Sample Target: {next_route_tokens[0].tolist()}")
    # 平均損失を計算
    train_loss = epoch_loss / num_batches
    history["train_loss"].append(train_loss)
    logger.info(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        num_batches = 0
        for i, (batch, time_batch) in enumerate(val_loader):
            route_batch = batch.to(device)
            time_batch = time_batch.to(device)
            time_is_day = (time_batch < 202409281500).to(device)
            tokenizer = Tokenization(network)
            discontinuous_route_tokens = tokenizer.tokenization(route_batch, mode = "discontinuous").long().to(device)
            discontinuous_feature_mat = tokenizer.make_feature_mat(discontinuous_route_tokens).to(device)
            time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(discontinuous_feature_mat.shape[0], discontinuous_feature_mat.shape[1], 1)
            discontinuous_feature_mat = torch.cat([discontinuous_feature_mat, time_feature_mat], dim=-1)
            complete_route_tokens = tokenizer.tokenization(route_batch, mode = "simple").long().to(device)
            complete_feature_mat = tokenizer.make_feature_mat(complete_route_tokens).to(device)
            time_feature_mat = time_is_day.unsqueeze(1).unsqueeze(2).expand(complete_feature_mat.shape[0], complete_feature_mat.shape[1], 1)
            complete_feature_mat = torch.cat([complete_feature_mat, time_feature_mat], dim=-1)
            next_route_tokens = tokenizer.tokenization(route_batch, mode = "next").long().to(device)

            outputs = model(discontinuous_route_tokens, discontinuous_feature_mat, complete_route_tokens, complete_feature_mat)
            outputs = outputs.view(-1, vocab_size)
            loss = criterion(outputs, next_route_tokens.view(-1), complete_route_tokens.view(-1))
                
            epoch_loss += loss.item()
            num_batches += 1

    val_loss = epoch_loss / num_batches
    history["val_loss"].append(val_loss)
    logger.info(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {val_loss:.4f}")
    wandb.log({"total_loss": train_loss, "val_loss": val_loss})
wandb.finish()

# モデルのパラメータを保存
save_data = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
    "train_indices": train_indices,
    "val_indices": val_indices
}
torch.save(save_data, savefilename)
print("Model weights saved successfully")