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


class Actformer(nn.Module):
  def __init__(self, 
               # enc_vocab_size, 
               # dec_vocab_size, # token_emb_dim, 
               # time_vocab_size, 
               loc_vocab_size, act_vocab_size, 
               # token_dim, time_emb_dim, 
               loc_emb_dim, act_emb_dim,
               feature_dim, feature_emb_dim, d_ff, head_num, B_en, B_de): ### ここの特徴量はlocについてのみ
    
    super().__init__() 
    self.encoder = Encoder(# enc_vocab_size, #token_emb_dim, 
                           # time_vocab_size, 
                           loc_vocab_size, act_vocab_size, 
                           # token_dim, time_emb_dim, 
                           loc_emb_dim, act_emb_dim,
                           feature_dim, feature_emb_dim, d_ff, head_num, B_en)
    # print('encoder')
    self.decoder = Decoder(# dec_vocab_size, #token_emb_dim, 
                           # time_vocab_size, 
                           loc_vocab_size, act_vocab_size, 
                           # token_dim, time_emb_dim, 
                           loc_emb_dim, act_emb_dim,
                           feature_dim, feature_emb_dim, d_ff, head_num, B_de)
    # print('decoder')
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.act_output = None
    self.loc_output = None

  # インスタンス生成後はforward()が実施される
  def forward(self, 
              enc_input_act, enc_input_loc, enc_feature, 
              dec_input_act, dec_input_loc, dec_feature): 
    # print('forward kaisi')
    # encoderには不連続トークンを，decoderには連続トークンを入れる
    mask = nn.Transformer.generate_square_subsequent_mask(dec_input_act.shape[1]).to(self.device) # 上三角が -inf（もしくは非常に大きな負値） の行列: 未来のトークンを参照しないようにする
    ## dec_input_actの形状のみみてるので中身は不要
    # decoderに入れる→Multiheadmattentionに行く->QKに加算される

    enc_input_act = enc_input_act.long().to(self.device)
    enc_input_loc = enc_input_loc.long().to(self.device)
    dec_input_act = dec_input_act.long().to(self.device)
    dec_input_loc = dec_input_loc.long().to(self.device)

    # print('kokomade')

    enc_output = self.encoder(enc_input_act, enc_input_loc, enc_feature) # エンコーダ：不完全トークンで実行，出力を完全トークンと共にデコーダに渡す
    # print('enc_output')
    act_output, loc_output = self.decoder(dec_input_act, dec_input_loc, dec_feature, enc_output, mask)
    ### ここでenc_feature, dec_featureはそれぞれのトークン列に対応した特徴量が抽出済み（tokenizationでその操作を実施しているはず）
    self.act_output = act_output
    self.loc_output = loc_output

    return act_output, loc_output


# 特徴量埋め込み
class EmbeddingWithFeatures(nn.Module):
  def __init__(self, # vocab_size, 
               loc_vocab_size, act_vocab_size, # token_dim, 
               loc_emb_dim, act_emb_dim, 
               # out_dim,
               feature_dim=None, feature_emb_dim=None, dropout=0.1):
    super().__init__()
    # トークン埋め込み層
    # self.token_embedding = nn.Embedding(vocab_size, token_dim) # vocab sizeをtoken dimに埋め込む：ここではtoken dim: 22次元
    # self.time_emb = nn.Embedding(time_vocab_size, time_emb_dim)
    self.loc_token_emb = nn.Embedding(loc_vocab_size+4, loc_emb_dim)
    self.act_token_emb = nn.Embedding(act_vocab_size+4, act_emb_dim)
    
    # self.concat_dim = time_emb_dim + loc_emb_dim + act_emb_dim
    # self.concat_dim = loc_emb_dim + act_emb_dim

    # 特徴埋め込み層（特徴がある場合）# 特徴量はlocのみ
    if feature_dim and feature_emb_dim:
        self.feature_projection = nn.Linear(feature_dim, feature_emb_dim) #　特徴量次元数→特徴量埋め込み次元数への線形変換 # 学習対象ね
        self.use_features = True
    else:
        self.feature_projection = None
        self.use_features = False
  
    # concat後を out_dim (＝ d_model相当) に射影
    # self.proj = nn.Linear(self.concat_dim, out_dim = time_emb_dim + loc_emb_dim + act_emb_dim) # 結合ベクトルを線形変換するため
    # self.proj = nn.Linear(self.concat_dim, out_dim = loc_emb_dim + act_emb_dim) # 結合ベクトルを線形変換するため

    # Dropout層
    self.dropout = nn.Dropout(dropout)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #self.to(self.device)

  def forward(self, loc_tokens, act_tokens, features=None): #  time_ids, loc_ids, act_ids: token
    # tokens = tokens.to(self.device)
    loc_tokens = loc_tokens.to(self.device)
    act_tokens = act_tokens.to(self.device)

    print('loc_tokens', loc_tokens[0])
    print('act_tokens', act_tokens[0])

    if features is not None:
        features = features.to(self.device)
    # print("loc_tokens.max():", loc_tokens.max().item())
    # print("embedding table size:", self.loc_token_emb.num_embeddings)
    # print("unique loc_tokens:", torch.unique(loc_tokens))
    # トークン埋め込み
    # token_emb = self.token_embedding(tokens)  # [batch_size, seq_len,  token_dim]
    l_vec = self.loc_token_emb(loc_tokens)    # => [B, seq_len, loc_emb_dim]
    a_vec = self.act_token_emb(act_tokens)    # => [B, seq_len, act_emb_dim]    
    # print(f'l_vec shape: {l_vec.shape}') # [64, 20, 8]
    # print(f'a_vec shape: {a_vec.shape}')
    # 特徴埋め込みがある場合

    if self.use_features and features is not None:
        # print('きてる？？？')
        feature_emb = self.feature_projection(features)  # [batch_size, seq_len, feature_emb_dim]
        # print(f"token_emb shape: {feature_emb.shape}")
        # print(f"feature_emb shape: {feature_emb.shape}")
        # print('ここは？？？')
        # 特徴量の埋め込み結果と共に全部結合する
        # emb = torch.cat((token_emb, feature_emb), dim=-1)  #　トークンと特徴量を 結合 [batch_size, seq_len, token_dim + feature_emb_dim]
        #concat_vec = torch.cat([l_vec, a_vec, feature_emb], dim=-1)
        #emb = self.proj(concat_vec)      # => [B, S, d_model]
        emb = torch.cat([l_vec, a_vec, feature_emb], dim=-1) # => [B, S, loc_emb_dim + act_emb_dim + feature_emb_dim]
        # print('ここは？？？？？')
        # print(f'emb_shape = {emb.shape}') # 8+8+2

    else:
        # concat_vec = torch.cat([l_vec, a_vec], dim=-1)
        # emb = token_emb
        # emb = self.proj(concat_vec)
        emb = torch.cat([l_vec, a_vec], dim=-1) ### is this OK????
    # Dropoutを適用
    
    emb = self.dropout(emb)
    # print('ここは？？？？？？？？？？')
    #print(f'emb_shape = {emb.shape}')
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
    #print(f'pe_shape = {pe.shape}')
    #print(f'pe = {pe}')

  def forward(self, x, mask): # x: 埋め込まれたトークン列, mask: パディングマスクの有無
    #print(f'x_shape = {x.shape}')
    #print(f'pe[:x.size(0)] = {self.pe[:, :x.size(1)].shape}')
    batch_size, seq_len, dim = x.size() ## 64 * 18 * 10?
    # print(f'batch_size = {batch_size}, seq_len = {seq_len}, dim = {dim}')
    # パディングでないトークンのインデックスを計算
    # print(f'mask = {mask.shape}')
    positions = torch.zeros_like(mask, dtype=torch.long, device=self.device)  # (batch_size, seq_len)
    for i in range(batch_size):
        non_pad_positions = torch.arange(seq_len, device=self.device)[~mask[i]]  # パディングでないインデックスを取得
        positions[i, ~mask[i]] = torch.arange(len(non_pad_positions), device=self.device)  # 0, 1, 2, ... を割り当て
        # torch.arrange は１づつ増える数列！ # unsqueeze(1)で次元を追加

    #first_one_indices = torch.argmax((positions == 1).long(), dim=1).to(self.device) 
    #print(f'first_one_indices = {first_one_indices}')
    # 非パディング部分の位置エンコーディングを取得
    positions = positions.unsqueeze(-1).expand(-1, -1, self.pe.size(-1))  # unsqueeze(1)で次元を追加 # 各位置（0〜max_len-1）を縦に並べた列ベクトルになります
    #print(positions.shape)
    #print(positions)
    #pe_use = self.pe[:, positions, :]  # (batch_size, seq_len, dim)
    pe_use = self.pe.expand(batch_size, -1, -1).gather(1, positions) 
    x = x + pe_use
    #print(f'x_re_shape = {x.shape}')
    return self.dropout(x)


class MultiHeadAttention(nn.Module): # maskが与えられるのはDecoderBlockの中のALTが実行される時のみ．
  
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
    x = torch.stack(x, dim = 1) # 
    return x
  
  def concat_head(self, x):
    x = torch.tensor_split(x, x.size()[1], dim = 1)
    x = torch.concat(x, dim = 3).squeeze(dim = 1)
    return x

  def forward(self, Q, K, V, mask = None): # maskはDecoderBlockの中のALTでのみ使用される
    Q = self.linear_Q(Q) #(BATCH_SIZE,word_count,dim)
    K = self.linear_K(K) # 線形化
    V = self.linear_V(V)
    
    Q = self.split_head(Q) #(BATCH_SIZE,head_num,word_count,dim//head_num)
    K = self.split_head(K) # 分割
    V = self.split_head(V)

    QK = torch.matmul(Q, torch.transpose(K, 3, 2)) # matmulは内積
    QK = QK/((self.dim//self.head_num)**0.5) # 1headあたりの次元数のルートで割る

    if mask is not None: ## paddingにはマスクをかける
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


class Encoder(nn.Module):

  def __init__(self, 
               # enc_vocab_size, # 状態数＋特殊トークン数
               # token_emb_dim, 
               # vocab_size, 
               # time_vocab_size, 
               loc_vocab_size, act_vocab_size, # token_dim, time_emb_dim, 
               loc_emb_dim, act_emb_dim,
               feature_dim, feature_emb_dim, d_ff, head_num, B_en, dropout = 0.1):
    super().__init__() 
    # self.dim = time_emb_dim + loc_emb_dim + act_emb_dim + feature_emb_dim
    self.dim = loc_emb_dim + act_emb_dim + feature_emb_dim # どうやらこれで良さそう
    # 8 + 8 + 2 = 18
    self.embed = EmbeddingWithFeatures( # ここはこれでOKのはず
                                      #  enc_vocab_size, # 状態数+特殊トークン数
                                      # token_emb_dim, feature_dim, feature_emb_dim
                                      # vocab_size, 
                                      # time_vocab_size, 
                                       loc_vocab_size, act_vocab_size, 
                                      # token_dim, time_emb_dim, 
                                       loc_emb_dim, act_emb_dim,
                                       feature_dim, feature_emb_dim, dropout=0.1
                                       )
    self.PE = PositionalEncoding(self.dim) # 位置エンコーディング
    self.dropout = nn.Dropout(dropout)
    # print('head_num = ', head_num, 'self.dim = ', self.dim)
    self.EncoderBlocks = nn.ModuleList([EncoderBlock(self.dim, head_num, d_ff) for _ in range(B_en)])
    self.B_en = B_en

  ### 不連続トークンを入力
  def forward(self, act_tokens, loc_tokens, features = None):
    #パディングマスクを作成
    # 今paddingは4なので
    padding_mask_loc = (loc_tokens == 4) # パディングトークンはなし？みんな時間揃っているので
    padding_mask = (act_tokens == 4) # true or false を返す：対応するはず
    if not torch.equal(padding_mask_loc, padding_mask):
      print('padding_mask_loc !!!!!!!!= padding_mask')

    # print('unique loc_tokens = ', torch.unique(loc_tokens))
    # print('unique act_tokens = ', torch.unique(act_tokens))
    # print(f'padding_mask_loc = {padding_mask_loc}')
    x = self.embed(loc_tokens, act_tokens, features) # 複数トークンと特徴量の埋め込み
    # print('ここまではきている')
    # print(f'x_shape = {x.shape}')
    x = x*(self.dim**0.5)
    # print(f'x_shape = {x.shape}')
    x = self.PE(x, padding_mask) # 位置エンコーディング-> EncoderBlockに入力->Attentionから処理してく
    x = self.dropout(x)
    for i in range(self.B_en): # B_en回EncoderBlockを通す
      x = self.EncoderBlocks[i](x)

    # print(f'x_shape = {x.shape}')
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
    # print('head_num = ', head_num)
    # print('dim = ', dim)

  def forward(self, x):
    Q = K = V = x
    x = self.MHA(Q, K, V) # maskはなし maskされるのはDecoderBlock内のALT=attention to link traveled
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    _x = x
    x = self.FF(x)
    x = self.dropout_2(x)
    x = x + _x # Residual Connection 残差結合 # FF処理出力に元のxを足すことで入力との結合を保証
    x = self.layer_norm_2(x)
    return x


class Decoder(nn.Module):

  def __init__(self, 
               # dec_vocab_size, # token_emb_dim, 
               # time_vocab_size, 
               loc_vocab_size, act_vocab_size, # token_dim, time_emb_dim, 
               loc_emb_dim, act_emb_dim,
               feature_dim, feature_emb_dim, 
               d_ff, head_num, B_de, dropout = 0.1):
    super().__init__() 
    # self.dim = time_emb_dim + loc_emb_dim + act_emb_dim + feature_emb_dim
    self.dim = loc_emb_dim + act_emb_dim + feature_emb_dim

    self.embed = EmbeddingWithFeatures(
      # dec_vocab_size, # token_emb_dim, 
      # time_vocab_size, 
      loc_vocab_size, act_vocab_size, 
      # token_dim, time_emb_dim, 
      loc_emb_dim, act_emb_dim,
      feature_dim, feature_emb_dim)
    self.PE = PositionalEncoding(self.dim)
    self.DecoderBlocks = nn.ModuleList([DecoderBlock(self.dim, head_num, d_ff) for _ in range(B_de)])
    self.dropout = nn.Dropout(dropout)

    self.act_logits = None
    self.loc_logits = None

    # self.linear = nn.Linear(self.dim, dec_vocab_size) # dec_vocab_sizeが状態数＋特殊トークン数
    # 活動と場所それぞれにLinear層を用意する

    ### ここがダメかも
    # self.linear_act = nn.Linear(self.dim, act_vocab_size)  
    # self.linear_loc = nn.Linear(self.dim, loc_vocab_size)

    self.linear_act = nn.Linear(self.dim, act_vocab_size + 4)  
    self.linear_loc = nn.Linear(self.dim, loc_vocab_size + 4)
    
    self.B_de = B_de # DecoderBlockの数

  def forward(self, act_tokens, loc_tokens, features, y, mask):
    padding_mask = (act_tokens == 4) # true or false を返す：対応するはず
    x = self.embed(loc_tokens, act_tokens, features)
    x = x*(self.dim**0.5)
    x = self.PE(x, padding_mask)
    x = self.dropout(x)
    for i in range(self.B_de):
      x = self.DecoderBlocks[i](x, y, mask)
    # x = self.linear(x)   #損失の計算にnn.CrossEntropyLoss()を使用する為、Softmax層を挿入しない
    act_logits = self.linear_act(x) # まとまって
    loc_logits = self.linear_loc(x)

    self.act_logits = act_logits
    self.loc_logits = loc_logits

    return act_logits, loc_logits
  

class DecoderBlock(nn.Module):

  def __init__(self, dim, head_num, d_ff, dropout = 0.1):
    super().__init__() 
    
    self.MMHA = MultiHeadAttention(dim, head_num) # MHAをデーコーダでは二つ使う（ALP=cross-attention, ALT=masked self-attention）
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
    # print(f'Q.shape = {Q.shape}')
    print('in decoder')
    x = self.MMHA(Q, K, V, mask) # 過去のみ参照=ALT attention to link traveled
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    Q = x
    K = V = y # yはエンコーダの出力: cross-attention
    print('in decoder2')
    x = self.MHA(Q, K, V) # 全部参照 =ALP attention to link-to-pass
    x = self.dropout_2(x)
    x = x + Q
    x = self.layer_norm_2(x)
    _x = x
    x = self.FF(x) # FeedForward
    x = self.dropout_3(x)
    x = x + _x
    x = self.layer_norm_3(x)
    return x
