import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import math
from collections import Counter


class Routesformer(nn.Module):
  def __init__(self, enc_vocab_size, dec_vocab_size, token_emb_dim, feature_dim, feature_emb_dim, d_ff, head_num, B_en, B_de):
    super().__init__() 
    self.encoder = Encoder(enc_vocab_size, token_emb_dim, feature_dim, feature_emb_dim, d_ff, head_num, B_en)
    self.decoder = Decoder(dec_vocab_size, token_emb_dim, feature_dim, feature_emb_dim, d_ff, head_num, B_de)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward(self, enc_input, enc_feature, dec_input, dec_feature):
    mask = nn.Transformer.generate_square_subsequent_mask(dec_input.shape[1]).to(self.device)
    enc_input = enc_input.long().to(self.device)
    dec_input = dec_input.long().to(self.device)
    enc_output = self.encoder(enc_input, enc_feature)
    output = self.decoder(dec_input, dec_feature, enc_output, mask)
    return output

# 特徴量埋め込み
class EmbeddingWithFeatures(nn.Module):
  def __init__(self, vocab_size, token_dim, feature_dim=None, feature_emb_dim=None, dropout=0.1):
    super(EmbeddingWithFeatures, self).__init__()
    # トークン埋め込み層
    self.token_embedding = nn.Embedding(vocab_size, token_dim) # vocab sizeをtoken dimに埋め込む：ここではtoken dim: 22次元

    # 特徴埋め込み層（特徴がある場合）
    if feature_dim and feature_emb_dim:
        self.feature_projection = nn.Linear(feature_dim, feature_emb_dim) # 特徴量埋め込み次元は2
        self.use_features = True
    else:
        self.feature_projection = None
        self.use_features = False
  
    self.dropout = nn.Dropout(dropout)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward(self, tokens, features=None):
    # トークン埋め込み
    tokens = tokens.to(self.device)
    if features is not None:
        features = features.to(self.device)
    token_emb = self.token_embedding(tokens)  # [batch_size, seq_len,  token_dim]
    
    # 特徴埋め込みがある場合
    if self.use_features and features is not None:
        feature_emb = self.feature_projection(features)  # [batch_size, seq_len, feature_emb_dim]
        emb = torch.cat((token_emb, feature_emb), dim=-1)  #　トークンと特徴量を 結合 [batch_size, seq_len, token_dim + feature_emb_dim]
    else:
        emb = token_emb
    emb = self.dropout(emb)
    return emb


class PositionalEncoding(nn.Module):

  def __init__(self, dim, dropout = 0.1, max_len = 500):
    super().__init__()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = device
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)).to(device)
    pe = torch.zeros(1, max_len, dim).to(device)
    pe[0, :, 0::2] = torch.sin(position * div_term) # 0から始まる，偶数
    pe[0, :, 1::2] = torch.cos(position * div_term) # 1から始まる，奇数
    self.register_buffer('pe', pe)

  def forward(self, x, mask):
    batch_size, seq_len, dim = x.size()
    # パディングでないトークンのインデックスを計算
    positions = torch.zeros_like(mask, dtype=torch.long, device=self.device)  # (batch_size, seq_len)
    for i in range(batch_size):
        non_pad_positions = torch.arange(seq_len, device=self.device)[~mask[i]]  # パディングでないインデックスを取得
        positions[i, ~mask[i]] = torch.arange(len(non_pad_positions), device=self.device)  # 0, 1, 2, ... を割り当て
        # torch.arrange は１づつ増える数列！ # unsqueeze(1)で次元を追加

    #first_one_indices = torch.argmax((positions == 1).long(), dim=1).to(self.device) 
    # 非パディング部分の位置エンコーディングを取得
    positions = positions.unsqueeze(-1).expand(-1, -1, self.pe.size(-1))  # unsqueeze(1)で次元を追加 # 各位置（0〜max_len-1）を縦に並べた列ベクトルになります
    pe_use = self.pe.expand(batch_size, -1, -1).gather(1, positions) 
    x = x + pe_use
    return self.dropout(x)

class MultiHeadAttention(nn.Module):
  
  def __init__(self, dim, head_num, dropout = 0.1):
    super().__init__() 
    self.dim = dim
    self.head_num = head_num
    self.linear_Q = nn.Linear(dim, dim, bias = True)
    self.linear_K = nn.Linear(dim, dim, bias = True)
    self.linear_V = nn.Linear(dim, dim, bias = True)
    self.linear = nn.Linear(dim, dim, bias = False)
    self.soft = nn.Softmax(dim = 3)
    self.dropout = nn.Dropout(dropout)
  
  def split_head(self, x):
    x = torch.tensor_split(x, self.head_num, dim = 2) # Q, K, Vをhead数に分割
    x = torch.stack(x, dim = 1)
    return x
  
  def concat_head(self, x):
    x = torch.tensor_split(x, x.size()[1], dim = 1)
    x = torch.concat(x, dim = 3).squeeze(dim = 1)
    return x

  def forward(self, Q, K, V, mask = None):
    Q = self.linear_Q(Q)   #(BATCH_SIZE,word_count,dim)
    K = self.linear_K(K) # 線形化
    V = self.linear_V(V)
    
    Q = self.split_head(Q)   #(BATCH_SIZE,head_num,word_count,dim//head_num)
    K = self.split_head(K) # 分割
    V = self.split_head(V)

    QK = torch.matmul(Q, torch.transpose(K, 3, 2)) # matmulは内積
    QK = QK/((self.dim//self.head_num)**0.5) # 1headあたりの次元数のルートで割る

    if mask is not None:
      QK = QK + mask

    softmax_QK = self.soft(QK) # softmax (マスクしたやつを入れる)
    softmax_QK = self.dropout(softmax_QK) # A

    QKV = torch.matmul(softmax_QK, V) # AV: Y
    QKV = self.concat_head(QKV)
    QKV = self.linear(QKV)
    return QKV

class FeedForward(nn.Module):

  def __init__(self, dim, d_ff, dropout = 0.1):
    super().__init__() 
    self.dropout = nn.Dropout(dropout)
    self.linear_1 = nn.Linear(dim, d_ff)
    self.relu = nn.ReLU()
    self.linear_2 = nn.Linear(d_ff, dim)

  def forward(self, x):
    x = self.linear_1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear_2(x)
    return x

class EncoderBlock(nn.Module):

  def __init__(self, dim, head_num, d_ff, dropout = 0.1):
    super().__init__() 
    self.MHA = MultiHeadAttention(dim, head_num)
    self.layer_norm_1 = nn.LayerNorm([dim])
    self.layer_norm_2 = nn.LayerNorm([dim])
    self.FF = FeedForward(dim, d_ff)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, x):
    Q = K = V = x
    x = self.MHA(Q, K, V)
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    _x = x
    x = self.FF(x)
    x = self.dropout_2(x)
    x = x + _x # Residual Connection 残差
    x = self.layer_norm_2(x)
    return x

class Encoder(nn.Module):

  def __init__(self, enc_vocab_size, token_emb_dim, feature_dim, feature_emb_dim, d_ff, head_num, B_en, dropout = 0.1):
    super().__init__() 
    self.dim = token_emb_dim + feature_emb_dim
    self.embed = EmbeddingWithFeatures(enc_vocab_size, token_emb_dim, feature_dim, feature_emb_dim)
    self.PE = PositionalEncoding(self.dim) # 位置エンコーディング
    self.dropout = nn.Dropout(dropout)
    self.EncoderBlocks = nn.ModuleList([EncoderBlock(self.dim, head_num, d_ff) for _ in range(B_en)])
    self.B_en = B_en

  def forward(self, tokens, features = None):
    #パディングマスクを作成
    padding_mask = (tokens == 19)
    x = self.embed(tokens, features) # トークンと特徴量の埋め込み
    x = x*(self.dim**0.5)
    x = self.PE(x , padding_mask) # 位置エンコーディング-> EncoderBlockに入力->Attentionから処理してく
    x = self.dropout(x)
    for i in range(self.B_en): # B_en回EncoderBlockを通すということのはず
      x = self.EncoderBlocks[i](x)
    return x

class DecoderBlock(nn.Module):

  def __init__(self, dim, head_num, d_ff, dropout = 0.1):
    super().__init__() 
    self.MMHA = MultiHeadAttention(dim, head_num)
    self.MHA = MultiHeadAttention(dim, head_num)
    self.layer_norm_1 = nn.LayerNorm([dim])
    self.layer_norm_2 = nn.LayerNorm([dim])
    self.layer_norm_3 = nn.LayerNorm([dim])
    self.FF = FeedForward(dim, d_ff)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)

  def forward(self, x, y, mask):
    Q = K = V = x # self-attention 
    x = self.MMHA(Q, K, V, mask) # 過去のみ参照
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    Q = x
    K = V = y # yはエンコーダの出力: cross-attention
    x = self.MHA(Q, K, V) # 全部参照
    x = self.dropout_2(x)
    x = x + Q
    x = self.layer_norm_2(x)
    _x = x
    x = self.FF(x) # FeedForward
    x = self.dropout_3(x)
    x = x + _x
    x = self.layer_norm_3(x)
    return x

class Decoder(nn.Module):

  def __init__(self, dec_vocab_size, token_emb_dim, feature_dim, feature_emb_dim, d_ff, head_num, B_de, dropout = 0.1):
    super().__init__() 
    self.dim = token_emb_dim + feature_emb_dim
    self.embed = EmbeddingWithFeatures(dec_vocab_size, token_emb_dim, feature_dim, feature_emb_dim)
    self.PE = PositionalEncoding(self.dim)
    self.DecoderBlocks = nn.ModuleList([DecoderBlock(self.dim, head_num, d_ff) for _ in range(B_de)])
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(self.dim, dec_vocab_size)
    self.B_de = B_de

  def forward(self, tokens, features, y, mask):
    padding_mask = (tokens == 19)
    x = self.embed(tokens, features)
    x = x*(self.dim**0.5)
    x = self.PE(x, padding_mask)
    x = self.dropout(x)
    for i in range(self.B_de):
      x = self.DecoderBlocks[i](x, y, mask)
    x = self.linear(x)   #損失の計算にnn.CrossEntropyLoss()を使用する為、Softmax層を挿入しない
    return x

