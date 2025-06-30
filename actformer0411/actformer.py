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
import sys


class Actformer(nn.Module):
  def __init__(self, # enc_vocab_size, # dec_vocab_size, # token_emb_dim, 
               context_vocab_sizes,
               time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim, 
               time_emb_dim, loc_emb_dim, act_emb_dim,
               feature_dim, feature_emb_dim, d_ff, head_num, B_en, B_de): ### ここの特徴量はlocについてのみ
    
    super().__init__() 
    self.encoder = Encoder(# enc_vocab_size, #token_emb_dim, 
                           context_vocab_sizes,
                           time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim, 
                           time_emb_dim, loc_emb_dim, act_emb_dim,
                           feature_dim, feature_emb_dim, d_ff, head_num, B_en,
                           use_tri_modal=False # TriModalBlockを使うかどうか
                           )
    self.decoder = Decoder(# dec_vocab_size, #token_emb_dim, 
                           context_vocab_sizes,
                           time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim,
                           time_emb_dim, loc_emb_dim, act_emb_dim,
                           feature_dim, feature_emb_dim, d_ff, head_num, B_de, 
                           use_tri_modal=False) # cross attentionを使うかどうか
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.time_output = None
    self.loc_output = None
    self.act_output = None

  def forward(self, 
             context_tokens,
              enc_input_time, enc_input_loc, enc_input_act, enc_feature,  # encoder: discontinuous token
              dec_input_time, dec_input_loc, dec_input_act, dec_feature): # decoder: continuous token

    mask = nn.Transformer.generate_square_subsequent_mask(dec_input_act.shape[1]).to(self.device) 
    # mask = self.build_causal_mask_with_context(dec_input_act.shape[1], context_tokens.shape[1], self.device) # maskはcontext_len + seq_lenの大きさ
    # print('mask.shape', mask.shape) # mask.shape torch.Size([23, 23]) # 23はseq_len
    enc_input_time = enc_input_time.long().to(self.device)
    enc_input_act = enc_input_act.long().to(self.device)
    enc_input_loc = enc_input_loc.long().to(self.device) # long.に変換しないとtorchできない
    dec_input_act = dec_input_act.long().to(self.device)
    dec_input_loc = dec_input_loc.long().to(self.device)
    dec_input_time = dec_input_time.long().to(self.device)
    context_tokens = context_tokens.long().to(self.device) # これでいいか？

    enc_output = self.encoder(
                              context_tokens, 
                              enc_input_time, enc_input_loc, enc_input_act, enc_feature) # エンコーダ：不完全トークンで実行，出力を完全トークンと共にデコーダに渡す
    # encoderでは圧縮されてるのが，decoderでは展開される？？
    time_output, loc_output, act_output = self.decoder(
                                                      context_tokens, 
                                                      dec_input_time, dec_input_loc, dec_input_act, dec_feature, 
                                                      enc_output, mask)

    # outputにcontextはなくて良い
    self.time_output = time_output
    self.loc_output = loc_output
    self.act_output = act_output

    return time_output, loc_output, act_output
  
  def build_causal_mask_with_context(self, seq_len, context_len, device): # 全体の長さは context_len + seq_len
    # context部分：context_len * context_len（すべて参照可）
    top_left = torch.zeros((context_len, context_len), device=device)
    
    # contextからsequence：context_len * seq_len（すべて参照可）
    top_right = torch.zeros((context_len, seq_len), device=device)
    
    # sequenceからcontext：seq_len * context_len（すべて参照可）
    bottom_left = torch.zeros((seq_len, context_len), device=device)
    
    # sequence部分：通常の causal mask（下三角）
    bottom_right = torch.triu(torch.ones((seq_len, seq_len), device=device) * float('-inf'), diagonal=1)
    # mask = nn.Transformer.generate_square_subsequent_mask(dec_input_act.shape[1]).to(self.device) 

    # 結合して最終マスク：(context_len + seq_len, context_len + seq_len)
    top = torch.cat([top_left, top_right], dim=1)
    bottom = torch.cat([bottom_left, bottom_right], dim=1)
    full_mask = torch.cat([top, bottom], dim=0)

    return full_mask


# decoder only transformer
class GPT(nn.Module):
  def __init__(self,
                context_vocab_sizes,
                time_vocab_size, # loc_vocab_size, 
                act_vocab_size, 
                time_emb_dim, #loc_emb_dim, 
                act_emb_dim,
                # feature_dim, feature_emb_dim, 
                d_ff, head_num, B_de, 
                use_tri_modal=False
                ):
     
    super().__init__()
      
    self.decoder = DecoderforGPT(
                  context_vocab_sizes,
                  time_vocab_size, # loc_vocab_size, 
                  act_vocab_size, # token_dim,
                  time_emb_dim, # loc_emb_dim, 
                  act_emb_dim,
                  # feature_dim, feature_emb_dim, 
                  d_ff, head_num, B_de, 
                  use_tri_modal=False
                  ) # cross attentionを使うかどうか
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.time_output = None
    # self.loc_output = None
    self.act_output = None

  # decoder only 
  def forward(self, 
             context_tokens,
              # enc_input_time, enc_input_loc, enc_input_act, enc_feature,  # encoderへの入力は全カット
              dec_input_time, # dec_input_loc, 
              dec_input_act, dec_feature): # decoder: continuous token
    dec_input_act = dec_input_act.long().to(self.device)
    # dec_input_loc = dec_input_loc.long().to(self.device)
    dec_input_time = dec_input_time.long().to(self.device)
    context_tokens = context_tokens.long().to(self.device) 
    mask = nn.Transformer.generate_square_subsequent_mask(dec_input_act.shape[1]).to(self.device) 
    time_output, act_output = self.decoder(context_tokens, dec_input_time, dec_input_act, dec_feature, mask)

    # outputにcontextはなくて良い
    self.time_output = time_output
    # self.loc_output = loc_output
    self.act_output = act_output

    return time_output, act_output # これがdecoderのインプットにもなる必要．．


  def build_causal_mask_with_context(self, seq_len, context_len, device): # 全体の長さは context_len + seq_len
    # context部分：context_len * context_len（すべて参照可）
    top_left = torch.zeros((context_len, context_len), device=device)
    
    # contextからsequence：context_len * seq_len（すべて参照可）
    top_right = torch.zeros((context_len, seq_len), device=device)
    
    # sequenceからcontext：seq_len * context_len（すべて参照可）
    bottom_left = torch.zeros((seq_len, context_len), device=device)
    
    # sequence部分：通常の causal mask（下三角）
    bottom_right = torch.triu(torch.ones((seq_len, seq_len), device=device) * float('-inf'), diagonal=1)
    # mask = nn.Transformer.generate_square_subsequent_mask(dec_input_act.shape[1]).to(self.device) 

    # 結合して最終マスク：(context_len + seq_len, context_len + seq_len)
    top = torch.cat([top_left, top_right], dim=1)
    bottom = torch.cat([bottom_left, bottom_right], dim=1)
    full_mask = torch.cat([top, bottom], dim=0)

    return full_mask
  

class Encoder(nn.Module):

  def __init__(self, # enc_vocab_size, # token_emb_dim, # vocab_size, 
               context_vocab_sizes, 
               time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim, 
               time_emb_dim, loc_emb_dim, act_emb_dim,
               feature_dim, feature_emb_dim, d_ff, head_num, B_en, dropout = 0.1,
               use_tri_modal=True # TriModalBlockを使うかどうか
               ):
    super().__init__() 
    # self.dim = time_emb_dim + loc_emb_dim + act_emb_dim + feature_emb_dim
    self.use_tri_modal = use_tri_modal
    self.dim = time_emb_dim + feature_emb_dim
    self.embed = EmbeddingWithFeatures( #  enc_vocab_size, # 状態数+特殊トークン数# token_emb_dim, feature_dim, feature_emb_dim# vocab_size,
                                      context_vocab_sizes,
                                      time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim, 
                                      time_emb_dim, loc_emb_dim, act_emb_dim,
                                      feature_dim, feature_emb_dim,
                                      d_model=36, head_num=head_num, dropout=0.1 # d_modelは
                                      )
    self.PE = PositionalEncoding(self.dim) # PEに渡されるのはt_dim+l_dim+a_dim(+f_dim)
    self.dropout = nn.Dropout(dropout)
    # self.EncoderBlocks = nn.ModuleList([EncoderBlock(self.dim, head_num, d_ff) for _ in range(B_en)])
    self.B_en = B_en
    self.time_vocab_size = time_vocab_size
    self.loc_vocab_size = loc_vocab_size
    self.act_vocab_size = act_vocab_size

    if self.use_tri_modal:  # フラグがTrueならTriModalを使用
        self.blocks = nn.ModuleList([
            TriModalBlock(self.dim, head_num, d_ff) for _ in range(B_en)
        ])
    else:  # フラグがFalseなら通常のEncoderBlockを使用
        self.blocks = nn.ModuleList([
            EncoderBlock(self.dim, head_num, d_ff) for _ in range(B_en)
        ])

    # self.blocks = nn.ModuleList([
    #     TriModalBlock(self.dim, head_num, d_ff)
    #     for _ in range(B_en)
    # ])

  def forward(self, 
              context_tokens,
              time_tokens, loc_tokens, act_tokens, features = None
              # use_tri_modal = True
              ): ### encoder: discontinuous token input

    # print('---encoder check---', 'context_tokens', context_tokens[0], 'time', time_tokens[0], 'loc', loc_tokens[0], 'act', act_tokens[0])
    padding_mask_loc = (loc_tokens == self.loc_vocab_size-4) # 今paddingは4なので # パディングトークンはなし？みんな時間揃っているので
    padding_mask_act = (act_tokens == self.act_vocab_size-4) # true or false を返す：対応するはず
    padding_mask_time = (time_tokens == self.time_vocab_size-4) # true or false を返す：対応するはず
    # print('padding_mask_time', padding_mask_time[0], 'padding_mask_loc', padding_mask_loc[0], 'padding_mask_act', padding_mask_act[0])

    context_mask = torch.zeros((time_tokens.shape[0], context_tokens.shape[1]), dtype=torch.bool, device=padding_mask_time.device)
    full_padding_mask = torch.cat([context_mask, padding_mask_time], dim=1)  

    if not torch.equal(padding_mask_loc, padding_mask_act):
      print('********* padding_mask_loc != padding_mask *********')
    pad_mask_seq = padding_mask_time

    '''ここでcontextとsequencial tokensを結合している multimodal embedding で消した部分
    x = self.embed( # 256*23*100 B*(S+C)*dim
                  context_tokens, # context_tokensはエンコーダの入力  
                  time_tokens, loc_tokens, act_tokens, features) # 複数トークンと特徴量の埋め込み # 256*23*100 B*(S+C)*dim
    '''

    # print('context_shape', context_tokens.shape, 'time_tokens_shape', time_tokens.shape, 'loc_tokens_shape', loc_tokens.shape, 'act_tokens_shape', act_tokens.shape)
    ctx_emb, time_emb, loc_emb, act_emb = self.embed(context_tokens, time_tokens, loc_tokens, act_tokens, features) # 埋め込み後には埋め込み次元数を持つようになる # 全部36に揃える
    # print('ctx_emb.shape', ctx_emb.shape, 'time_emb.shape', time_emb.shape, 'loc_emb.shape', loc_emb.shape, 'act_emb.shape', act_emb.shape) # 256*23*100

    time_emb = self.PE(time_emb * (self.dim**0.5), pad_mask_seq)
    loc_emb = self.PE(loc_emb * (self.dim**0.5), pad_mask_seq)
    act_emb = self.PE(act_emb * (self.dim**0.5), pad_mask_seq)

    # print('after PE', 'ctx_emb', ctx_emb.shape, 'time_emb.shape', time_emb.shape, 'loc_emb.shape', loc_emb.shape, 'act_emb.shape', act_emb.shape) # 256*23*100
    
    if self.use_tri_modal:
        for blk in self.blocks:
            time_emb, loc_emb, act_emb = blk(time_emb, loc_emb, act_emb, attn_mask=None, pad_mask=pad_mask_seq)
        fused = torch.cat([ctx_emb, time_emb + loc_emb + act_emb], dim=1)
    else:
        fused = torch.cat([ctx_emb, time_emb + loc_emb + act_emb], dim=1)
        for blk in self.blocks:
            fused = blk(fused)

    # for blk in self.blocks:
    #   time_emb, loc_emb, act_emb = blk(time_emb, loc_emb, act_emb, attn_mask = None, pad_mask = pad_mask_seq) ## encoderなのでattention maskはなし

    # 最後に全結合
    fused = torch.cat([ctx_emb, time_emb + loc_emb + act_emb], dim=1)
    # print('fused.shape', fused.shape) # 256*23*100

    '''
    ### ここで埋め込まれたやつがencoderに入っていく
    # x = x*(self.dim**0.5)
    # x = self.PE(x, full_padding_mask) # 位置エンコーディング-> EncoderBlockに入力->Attentionから処理してく
    # x = self.dropout(x)
    # for i in range(self.B_en): # B_en回EncoderBlockを通す
    #   x = self.EncoderBlocks[i](x)
    '''
    # print('encoder block finished')
    return fused


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
    x = self.MHA(Q, K, V) # maskなし maskされるのはDecoderBlock内のALT=attention to link traveled
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

  def __init__(self, # dec_vocab_size, # token_emb_dim, 
               context_vocab_sizes,
               time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim, 
               time_emb_dim, loc_emb_dim, act_emb_dim,
               feature_dim, feature_emb_dim, 
               d_ff, head_num, B_de, dropout = 0.1,
               use_tri_modal=True # TriModalBlockを使うかどうか
               ):
    super().__init__() 
    self.use_tri_modal = use_tri_modal
    self.dim = time_emb_dim + feature_emb_dim
    self.embed = EmbeddingWithFeatures( # dec_vocab_size, # token_emb_dim, 
      context_vocab_sizes,
      time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim, 
      time_emb_dim, loc_emb_dim, act_emb_dim,
      feature_dim, feature_emb_dim) 
    self.PE = PositionalEncoding(self.dim)
    self.dropout = nn.Dropout(dropout)
    self.time_logits = None
    self.loc_logits = None
    self.act_logits = None
    if self.use_tri_modal:
        self.blocks = nn.ModuleList([
            TriModalBlock(self.dim, head_num, d_ff) for _ in range(B_de)
        ])
    else:
        self.blocks = nn.ModuleList([
            DecoderBlock(self.dim, head_num, d_ff) for _ in range(B_de)
        ])

    # Tri‑Modal で自己注意したあと、y へクロスする層
    # self.ca_time = nn.MultiheadAttention(self.dim, head_num, batch_first=True)
    # self.ca_loc  = nn.MultiheadAttention(self.dim, head_num, batch_first=True)
    # self.ca_act  = nn.MultiheadAttention(self.dim, head_num, batch_first=True)

    # self.norm_ca_T = nn.LayerNorm(self.dim)
    # self.norm_ca_L = nn.LayerNorm(self.dim)
    # self.norm_ca_A = nn.LayerNorm(self.dim)

    # if self.use_cross_attention:  # ←cross-attentionを使う場合のみ初期化
    self.ca_time = nn.MultiheadAttention(self.dim, head_num, batch_first=True)
    self.ca_loc  = nn.MultiheadAttention(self.dim, head_num, batch_first=True)
    self.ca_act  = nn.MultiheadAttention(self.dim, head_num, batch_first=True)

    self.norm_ca_T = nn.LayerNorm(self.dim)
    self.norm_ca_L = nn.LayerNorm(self.dim)
    self.norm_ca_A = nn.LayerNorm(self.dim)

    # 活動と場所それぞれにLinear層を用意する
    self.linear_time = nn.Linear(self.dim, time_vocab_size) # +4: special tokens    
    self.linear_loc = nn.Linear(self.dim, loc_vocab_size)
    self.linear_act = nn.Linear(self.dim, act_vocab_size)  
    self.time_vocab_size = time_vocab_size
    self.loc_vocab_size = loc_vocab_size
    self.act_vocab_size = act_vocab_size
    self.B_de = B_de 

    # cross-modal attention
    # self.blocks = nn.ModuleList([TriModalBlock(self.dim, head_num, d_ff) for _ in range(B_de)])

  def forward(self, 
              context_tokens,
              time_tokens, loc_tokens, act_tokens, features, y, attn_mask): # simpleなtoken列を入れる y: enc_output
    padding_mask_loc = (loc_tokens == self.loc_vocab_size-4) # 今paddingは4なので # パディングトークンはなし？みんな時間揃っているので
    padding_mask_act = (act_tokens == self.act_vocab_size-4) # true or false を返す：対応するはず
    padding_mask_time = (time_tokens == self.time_vocab_size-4) # true or false を返す：対応するはず
    
    if not torch.equal(padding_mask_loc, padding_mask_act):
      print('********* padding_mask_loc != padding_mask *********')
    pad_mask_seq = padding_mask_time

    ctx_emb, time_emb, loc_emb, act_emb = self.embed(context_tokens, time_tokens, loc_tokens, act_tokens, features) # にする？？？？
    time_emb = self.PE(time_emb * (self.dim**0.5), pad_mask_seq) 
    loc_emb = self.PE(loc_emb * (self.dim**0.5), pad_mask_seq)
    act_emb = self.PE(act_emb * (self.dim**0.5), pad_mask_seq)
    
    # if self.use_tri_modal:
    #     for blk in self.blocks:
    #         time_emb, loc_emb, act_emb = blk(time_emb, loc_emb, act_emb, attn_mask, pad_mask_seq)
    #     final_emb = time_emb + loc_emb + act_emb
    # else:
    #     final_emb = time_emb + loc_emb + act_emb
    #     for blk in self.blocks:
    #         final_emb = blk(final_emb, y, attn_mask)

    # for blk in self.blocks: # ここの入植がみんな同じ
    #   time_emb, loc_emb, act_emb = blk(time_emb, loc_emb, act_emb, attn_mask, pad_mask_seq) ## ここサイズ変えなくていいか？

    # -------------------------------------------------
    #  NEW ✨  Cross‑Attention  (Q = T/L/A,  K = V = y)
    # -------------------------------------------------
    # 注意：y のパディングマスクは enc 側で 0/1 作って register_buffer しておくとベター
    # time_ca, _ = self.ca_time(time_emb, y, y)   # Q,K,V の順
    # loc_ca,  _ = self.ca_loc (loc_emb,  y, y)
    # act_ca,  _ = self.ca_act (act_emb,  y, y)

    # time_emb = self.norm_ca_T(time_emb + time_ca)   # 残差＋LayerNorm
    # loc_emb  = self.norm_ca_L(loc_emb  + loc_ca)
    # act_emb  = self.norm_ca_A(act_emb  + act_ca)

    # if self.use_cross_attention:  # ←cross-attentionを使う場合
    time_ca, _ = self.ca_time(time_emb, y, y)
    loc_ca,  _ = self.ca_loc(loc_emb,  y, y)
    act_ca,  _ = self.ca_act(act_emb,  y, y)

    time_emb = self.norm_ca_T(time_emb + time_ca)
    loc_emb  = self.norm_ca_L(loc_emb  + loc_ca)
    act_emb  = self.norm_ca_A(act_emb  + act_ca)
    # else:
        # pass  # cross-attentionを行わない場合、そのままスキップ
    
    # --- 出力線形層 ---
    time_logits = self.linear_time(time_emb) # 毎回シーケンスが伸びていく
    loc_logits  = self.linear_loc(loc_emb)
    act_logits  = self.linear_act(act_emb)

    # return time_logits, loc_logits, act_logits

    # x = x*(self.dim**0.5)

    # x = self.PE(x, full_padding_mask)
    # x = self.dropout(x)
    # ### ここがアテンション含んだブロックをB_de回通す部分
    # for i in range(self.B_de):
    #   x = self.DecoderBlocks[i](x, y, mask) # yはエンコーダの出力

    # # context部分を除外してから出力
    # time_logits = time_logits[:, context_tokens.shape[1]:, :]
    # loc_logits = loc_logits[:, context_tokens.shape[1]:, :]
    # act_logits = act_logits[:, context_tokens.shape[1]:, :]

    return time_logits, loc_logits, act_logits


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
    x = self.MMHA(Q, K, V, mask) # masked-MHA 過去のみ参照=ALT attention to link traveled
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    Q = x
    K = V = y # yはエンコーダの出力: cross-attention
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


class CrossAttention(nn.Module):
    def __init__(self, dim, head_num, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(dim, head_num, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_seq, key_value_seq):
        # query_seqからkey_value_seqへattention
        attended = self.cross_attention(query_seq, key_value_seq, key_value_seq)
        return self.norm(query_seq + attended)


class EmbeddingWithFeatures(nn.Module):
  def __init__(self, # vocab_size, 
               context_vocab_sizes, 
               time_vocab_size, loc_vocab_size, act_vocab_size, # token_dim, 
               time_emb_dim, loc_emb_dim, act_emb_dim, # out_dim,
               feature_dim=None, feature_emb_dim=None, 
               d_model = 32, head_num=4, dropout=0.1):
    super().__init__()

    # トークン埋め込み層 
    print('context_vocab_sizes', context_vocab_sizes, 'embedding_dim', time_emb_dim, loc_emb_dim, act_emb_dim)
    self.time_token_emb = nn.Embedding(time_vocab_size, time_emb_dim) # +4は特殊トークンの数
    self.loc_token_emb = nn.Embedding(loc_vocab_size, loc_emb_dim)
    self.act_token_emb = nn.Embedding(act_vocab_size, act_emb_dim)
    self.ctx_embedding = ContextEmbedding(context_vocab_sizes=context_vocab_sizes, d_model = d_model) # あとでくっつけるのでtime, loc, actと埋め込み次元は揃える

    # Cross Attention layers # previous version
    # self.cross_attn_time_loc = CrossAttention(d_model, head_num, dropout)
    # self.cross_attn_time_act = CrossAttention(d_model, head_num, dropout)
    # self.cross_attn_loc_act = CrossAttention(d_model, head_num, dropout)
    
    # 特徴埋め込み層（特徴がある場合）# 特徴量は今はlocのみ
    if feature_dim and feature_emb_dim:
        self.feature_projection = nn.Linear(feature_dim, feature_emb_dim) #　特徴量次元数→特徴量埋め込み次元数への線形変換 # 学習対象ね
        self.use_features = True
    else:
        self.feature_projection = None
        self.use_features = False
  
    self.project_time = nn.Linear(time_emb_dim, time_emb_dim)
    self.project_loc = nn.Linear(loc_emb_dim, time_emb_dim)
    self.project_act = nn.Linear(act_emb_dim, time_emb_dim)
    
    self.dropout = nn.Dropout(dropout)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # ここでcontextを埋め込み＆結合
  def forward(self, 
              context_tokens, # batch_size * context_len
              time_tokens, loc_tokens, act_tokens, features=None): #  time_ids, loc_ids, act_ids: token# tokens = tokens.to(self.device)
    time_tokens = time_tokens.to(self.device)
    loc_tokens = loc_tokens.to(self.device)
    act_tokens = act_tokens.to(self.device)
    context_tokens = context_tokens.to(self.device)

    # print('inside embedding', context_tokens.shape)

    if features is not None:
          features = features.to(self.device)

    time_emb = self.project_time(self.time_token_emb(time_tokens)) # time_dimに射影：B*S -> B*S*dimになる！
    loc_emb = self.project_loc(self.loc_token_emb(loc_tokens))
    act_emb = self.project_act(self.act_token_emb(act_tokens))

    # print('inside embedding', 'time_emb.shape',time_emb.shape, 'loc_emb.shape', loc_emb.shape, 'act_emb.shape', act_emb.shape) # 256*23*100

    # 特徴量を全系列に concat 
    if self.use_features and features is not None: ### ここで4次元分増加する
        feat = self.feature_projection(features)
        # print('inside embedding', 'feat.shape', feat.shape) # 256*23*100
        # print('inside embedding', 'time_emb.shape',time_emb.shape, 'loc_emb.shape', loc_emb.shape, 'act_emb.shape', act_emb.shape) # 256*23*100
        time_emb = torch.cat([time_emb, feat], -1)
        loc_emb = torch.cat([loc_emb, feat], -1)
        act_emb = torch.cat([act_emb, feat], -1)

    # Cross-attention interactions
    #time_emb_attn_loc = self.cross_attn_time_loc(time_emb, loc_emb)
    #time_emb_attn_act = self.cross_attn_time_act(time_emb, act_emb)
    #loc_emb_attn_act = self.cross_attn_loc_act(loc_emb, act_emb)
    
    # 特徴埋め込みがある場合
    # if self.use_features and features is not None:
    #     feature_emb = self.feature_projection(features)  # [batch_size, seq_len, feature_emb_dim]
    #     emb = torch.cat([time_emb_attn_loc, time_emb_attn_act, loc_emb_attn_act, feature_emb], dim=-1) # => [B, S, loc_emb_dim + act_emb_dim + feature_emb_dim]
    # else:
    #     emb = torch.cat([time_emb_attn_loc, time_emb_attn_act, loc_emb_attn_act], dim=-1) ###縦方向に結合
    # print('cat.shape', emb.shape, 'time_emb_shape', time_emb.shape, 'loc_emb_shape', loc_emb.shape, 'act_emb_shape', act_emb.shape, 'feature_emb_shape', feature_emb.shape)
    # print('after cross-attention')
    # print('t-vec_attn_loc.shape', time_emb_attn_loc.shape, 'time_emb_attn_act.shape', time_emb_attn_act.shape, 'loc_emb_attn_act.shape', loc_emb_attn_act.shape)
    # emb = self.dropout(emb)
    ### embの先頭にくっつける；(Batch_size, seq_len + context_len, dim)
    ctx_emb = self.ctx_embedding(context_tokens) # [batch_size, context_len, d_model)
    # print('inside embedding', ctx_emb.shape)
    # full_emb = torch.cat([ctx_emb, emb], dim=1) # [batch_size, seq_len + context_len, d_model]
    
    return ctx_emb, time_emb, loc_emb, act_emb  # return full_emb 


class ContextEmbedding(nn.Module):
    def __init__(self, context_vocab_sizes, d_model): # context_vocab_sizesは
        """
        Args:
            context_vocab_sizes (list[int]): 各属性列の語彙サイズのリスト
            d_model (int): Transformerに合わせる最終次元
        """
        super().__init__()
        self.ctx_embs = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for vocab_size in context_vocab_sizes # 各コンテキストの語彙数
        ])
        print(',,,,,,,,,,context_vocab_sizes', context_vocab_sizes, 'd_model', d_model)
    
    def forward(self, context_tokens):
        """
        context_tokens.shape: [batch_size, context_len]
          各列が「属性1, 属性2, ..., 属性N」のID
        戻り値: [batch_size, context_len, d_model]
        """
        # 属性ごとに埋め込みを取って1つのトークンにする
        # 注意：属性列ごとに単独トークンとして扱うので、以下のようにループ
        embed_list = []
        for i, emb_layer in enumerate(self.ctx_embs): # context_len
            c_vec = emb_layer(context_tokens[:, i])  # => [batch_size, d_model]
            embed_list.append(c_vec.unsqueeze(1))    # => [batch_size, 1, d_model]
        
        ctx_emb = torch.cat(embed_list, dim=1)

        return ctx_emb


class PositionalEncoding(nn.Module):

  def __init__(self, dim, dropout = 0.1, max_len = 500): # ここのmax_lenは余裕を持っておく
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

  def forward(self, x, mask): # x: 埋め込まれたトークン列, mask: パディングマスクの有無
    batch_size, seq_len, dim = x.size() 
    positions = torch.zeros_like(mask, dtype=torch.long, device=self.device)  # (batch_size, seq_len)
    for i in range(batch_size):
        non_pad_positions = torch.arange(seq_len, device=self.device)[~mask[i]]  # パディングでないインデックスを取得
        positions[i, ~mask[i]] = torch.arange(len(non_pad_positions), device=self.device)  # 0, 1, 2, ... を割り当て # torch.arrange は１づつ増える数列！ # unsqueeze(1)で次元を追加
    positions = positions.unsqueeze(-1).expand(-1, -1, self.pe.size(-1)) # 非パディング部分の位置エンコーディングを取得 # unsqueeze(1)で次元を追加 # 各位置（0〜max_len-1）を縦に並べた列ベクトルになります
    pe_use = self.pe.expand(batch_size, -1, -1).gather(1, positions)
    x = x + pe_use # 埋め込まれたトークン列に位置エンコーディングを加算
    return self.dropout(x)


class MultiHeadAttention(nn.Module):
  
  def __init__(self, dim, head_num, dropout = 0.1):
    super().__init__() 
    self.dim = dim
    self.head_num = head_num
    self.W_q = nn.Linear(dim, dim, bias = True)
    self.W_k = nn.Linear(dim, dim, bias = True)
    self.W_v = nn.Linear(dim, dim, bias = True)
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

  def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask = None): # maskが与えられればMask部分を追加するだけで基本的に同じ
    Q = self.W_q(encodings_for_q) #(BATCH_SIZE,word_count,dim)
    K = self.W_k(encodings_for_k) # 線形化
    V = self.W_v(encodings_for_v)
    
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

# cross-modal attention
class TriModalBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        # 自己注意
        self.sa = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                                 for _ in range(3)])
        # 交差注意
        self.ca = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                                 for _ in range(3)])
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, T, L, A, attn_mask, pad_mask):
        for x, sa, ln in [(T,self.sa[0],self.norm1[0]),
                          (L,self.sa[1],self.norm1[1]),
                          (A,self.sa[2],self.norm1[2])]:
            resid = x # time, loc, actの順にatt層が実行される
            x,_ = sa(x,x,x, attn_mask=attn_mask, key_padding_mask=pad_mask) # QKV=x self attention
            x = ln(resid + self.dropout(x)) # lnはLayerNormで正規化
        T,L,A = x if isinstance(x,tuple) else T,L,A   # noqa

        # --- cross‑modal ---
        T = self.norm2[0](T + self.ca[0](T, torch.cat([L,A],1), torch.cat([L,A],1))[0]) # MHAはQ, K, Vを引数にする→
        L = self.norm2[1](L + self.ca[1](L, torch.cat([T,A],1), torch.cat([T,A],1))[0])
        A = self.norm2[2](A + self.ca[2](A, torch.cat([T,L],1), torch.cat([T,L],1))[0])

        # --- FFN ---
        T = self.ff(T);  L = self.ff(L);  A = self.ff(A)
        return T, L, A


# GPTのデコーダはcross-attentionを必要としないので定義し直す必要がある．．
class DecoderforGPT(nn.Module):
   
  def __init__(self, # dec_vocab_size, # token_emb_dim, 
               context_vocab_sizes,
               time_vocab_size, # loc_vocab_size, 
               act_vocab_size, 
               time_emb_dim, # loc_emb_dim, 
               act_emb_dim,
               # feature_dim, feature_emb_dim, 
               d_ff, head_num, B_de, dropout = 0.1,
               use_tri_modal=False # TriModalBlockを使うかどうか
               ):
    
    super().__init__() 
    self.use_tri_modal = use_tri_modal
    self.dim = time_emb_dim #+ feature_emb_dim
    self.embed = EmbeddingWithFeatures( # ctx, time, loc, actの埋め込み済みテンソルが別々で帰ってくる
      context_vocab_sizes,
      time_vocab_size, #loc_vocab_size, 
      act_vocab_size, # token_dim, 
      time_emb_dim, # loc_emb_dim, 
      act_emb_dim,
      # feature_dim=None, feature_emb_dim=None
      ) 
    self.PE = PositionalEncoding(self.dim)
    self.dropout = nn.Dropout(dropout)

    self.time_logits = None
    self.loc_logits = None
    self.act_logits = None
    # DecoderBlockは使わずこの中で実装する
    # if self.use_tri_modal:
    #     self.blocks = nn.ModuleList([
    #         TriModalBlock(self.dim, head_num, d_ff) for _ in range(B_de)
    #     ])
    # else:
    #     self.blocks = nn.ModuleList([
    #         DecoderBlock(self.dim, head_num, d_ff) for _ in range(B_de)
    #     ])

    # self.sa_time = nn.MultiheadAttention(self.dim, head_num, batch_first=True)
    # self.sa_loc  = nn.MultiheadAttention(self.dim, head_num, batch_first=True)
    # self.sa_act  = nn.MultiheadAttention(self.dim, head_num, batch_first=True)

    # self.norm_ca_T = nn.LayerNorm(self.dim)
    # self.norm_ca_L = nn.LayerNorm(self.dim)
    # self.norm_ca_A = nn.LayerNorm(self.dim)

    # 活動と場所それぞれにLinear層を用意する
    self.linear_time = nn.Linear(self.dim, time_vocab_size) # +4: special tokens    
    # self.linear_loc = nn.Linear(self.dim, loc_vocab_size)
    self.linear_act = nn.Linear(self.dim, act_vocab_size)  
    self.time_vocab_size = time_vocab_size
    # self.loc_vocab_size = loc_vocab_size
    self.act_vocab_size = act_vocab_size
    self.B_de = B_de 

    self.blocks = nn.ModuleList([
        GPTDecoderBlock(self.dim, head_num, d_ff) for _ in range(B_de)
        ])


  def forward(self,
              context_tokens,
              time_tokens, loc_tokens, act_tokens, #features, 
              attn_mask):
    
    padding_mask_loc = (loc_tokens == self.loc_vocab_size-4) # 今paddingは4なので # パディングトークンはなし？みんな時間揃っているので
    padding_mask_act = (act_tokens == self.act_vocab_size-4) # true or false を返す：対応するはず
    padding_mask_time = (time_tokens == self.time_vocab_size-4) # true or false を返す：対応するはず
    
    if not torch.equal(padding_mask_loc, padding_mask_act):
      print('********* padding_mask_loc != padding_mask *********')
    pad_mask_seq = padding_mask_time

    ctx_emb, time_emb, act_emb = self.embed(context_tokens, time_tokens, act_tokens# ,# features
                                            ) # にする？？？？
    
    time_emb = self.PE(time_emb * (self.dim**0.5), pad_mask_seq) 
    # loc_emb = self.PE(loc_emb * (self.dim**0.5), pad_mask_seq)
    act_emb = self.PE(act_emb * (self.dim**0.5), pad_mask_seq)

    '''
    ## まとめるなら：
    final_emb = time_emb + loc_emb + act_emb
    for blk in self.blocks:
        final_emb = blk(final_emb, attn_mask)

    # --- 出力線形層 ---
    time_logits = self.linear_time(final_emb) # 毎回シーケンスが伸びていく
    loc_logits  = self.linear_loc(final_emb)
    act_logits  = self.linear_act(final_emb)
    '''

    ## まとめないなら： ## time, loc, actそれぞれで独立
    for blk in self.blocks:
       time_emb = blk(time_emb, attn_mask)
      #  loc_emb = blk(loc_emb, attn_mask)
       act_emb = blk(act_emb, attn_mask)
    
    # --- 出力線形層 ---
    time_logits = self.linear_time(time_emb) # 毎回シーケンスが伸びていく
    # loc_logits  = self.linear_loc(loc_emb)
    act_logits  = self.linear_act(act_emb)

    return time_logits, act_logits
  

class GPTDecoderBlock(nn.Module):
   
  def __init__(self, dim, head_num, d_ff, dropout = 0.1):
    super().__init__() 
    
    self.MMHA = MultiHeadAttention(dim, head_num) # non-masked cross-attentionは使わない．Masked cross-attentionのみ
    # self.MHA = MultiHeadAttention(dim, head_num)
    self.layer_norm_1 = nn.LayerNorm([dim])
    self.layer_norm_2 = nn.LayerNorm([dim])
    self.FF = FeedForward(dim, d_ff)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, x, mask):
    Q = K = V = x 
    x = self.layer_norm_1(x)
    x = self.MMHA(Q, K, V, mask) # masked-MHA 過去のみ参照=ALT attention to link traveled
    x = self.dropout_1(x)
    x = x + Q
    
    _x = x
    x = self.dropout_2(x)
    x = self.FF(x) 
    x = x + _x
    x = self.layer_norm_2(x)
    return x


    



'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleSelfAttention, self).__init__()
        
        # 重み行列（クエリ、キー、バリューを作る）
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        
        # 出力層
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # クエリ、キー、バリューを作成
        Q = self.W_q(x)  # (batch_size, seq_len, hidden_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # Attentionスコア計算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        attn_scores = attn_scores / (Q.size(-1) ** 0.5)  # スケーリング
        attn_weights = F.softmax(attn_scores, dim=-1)    # ソフトマックスで重み化

        # Attention適用
        context = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        # 出力層
        output = self.out_proj(context)  # (batch_size, seq_len, hidden_dim)
        
        return output
'''