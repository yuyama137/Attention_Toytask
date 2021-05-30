# import model
import copy_model
import torch
from torch import nn
import matplotlib.pyplot as plt
import csv

# constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_BATCHES = 5000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 32
attn_type = "linear_attn_elu"

def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long()
        tgt = torch.cat((prefix, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1] - 1).bool()
        yield (src, tgt, src_mask, tgt_mask)
        
def culc_loss(loss_func, inputs, targets):
    """
    損失関数の計算
    args:
        - loss_func : 損失関数(交差エントロピー)
        - input (B x len x d): 入力データ
        - target (B x len): ターゲットデータ
    
    文章ごとに平均をとって、バッチごとに平均をとる
    pytorchの交差エントロピー使わない方が収束早いし、lossも小さくなる。。。
    どういうことだ？
    計算結果が微妙に違う気がするし、nn.CrossEntropyが所望の計算をしてない可能性ある？
    """
    B, l, d = inputs.shape
    _loss = 0
    loss = 0
    for i in range(B):
        loss += loss_func(inputs[i], targets[i])# 内部的に文章平均
#         _loss += cross_ent(inputs[i], targets[i])
#     _loss /= B# バッチ平均
    loss /= B
    return loss

dim_lst = [256, 512, 1024]
head_lst = [8, 16]
depth_lst = [4]
ff_lst = [256, 512, 1024]
pos_lst = [64, 1000]

count = 1
for dim in dim_lst:
  for h in head_lst:
    for dep in depth_lst:
      for ff in ff_lst:
        for p in pos_lst:
          DIMENTION = dim
          HEAD = h
          # DEPTH_enc = 1
          # DEPTH_dec = 3
          DEPTH = dep ##
          DEPTH_enc = DEPTH
          DEPTH_dec = DEPTH
          ff_hidnum = ff ##
          position_max_len = p ##

          loss_lst = []# 記録用

          # transformer = model.Model(device, DIMENTION, NUM_TOKENS, 0.0, DEPTH_enc, DEPTH_dec, HEAD, HEAD).to(device)
          transformer = copy_model.CopyModel(device, DIMENTION, NUM_TOKENS, attn_type, DEPTH_enc, DEPTH_dec, HEAD, HEAD, ff_hidnum, position_max_len).to(device)
          criterion = nn.CrossEntropyLoss()
          # optimizer

          optim = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)

          # training

          for i in range(NUM_BATCHES):
              transformer.train()
              src, tgt, src_mask, tgt_mask = next(cycle())
              src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
              
              out = transformer(src, tgt)
              loss = culc_loss(criterion, out[:,:-1,:], src)
              loss.backward()
              # print(loss.item())

              optim.step()
              optim.zero_grad()
              

              loss_lst.append(loss.item())

              if i % GENERATE_EVERY == 0:
                  transformer.eval()
                  src, _, src_mask, _ = next(cycle())
                  src, src_mask = src[0:1].to(device), src_mask[0:1].to(device)
                  
                  sample = transformer.generate(src)
          #         import pdb; pdb.set_trace()
          #         incorrects = (src != sample).abs().sum()
                  incorrects = torch.sum(src != sample)

                  # print(f"input:  ", src)
                  # print(f"predicted output:  ", sample)
                  # print(f"incorrects: {incorrects}")

          file_name = "_poslen_{}_dim_{}_ffhidnum_{}_head_{}_depth_{}".format(attn_type,position_max_len, DIMENTION, ff_hidnum, HEAD, DEPTH)

          # lossの推移グラフ
          plt.rcParams["font.size"] = 18
          fig = plt.figure(figsize=(12,8))
          ax = fig.add_subplot(111)
          ax.plot(loss_lst, label="loss")
          ax.set_xlabel("iteration")
          ax.set_ylabel("loss")
          ax.set_title(file_name)
          plt.savefig(f"output_{attn_type}/{file_name}.png")
          plt.clf()
          plt.close()

          # 結果のパラメータにおけるincorrect number + 最後のloss
          sample_num = 10
          txt_lst = []
          for _ in range(sample_num):
            transformer.eval()
            src, _, src_mask, _ = next(cycle())
            src, src_mask = src[0:1].to(device), src_mask[0:1].to(device)

            sample = transformer.generate(src)
            incorrects = torch.sum(src != sample)

            txt_lst.append(f"input  : {src}\npredict: {sample}\nincorrects: {incorrects}\n\n")

          with open(f"output_{attn_type}/{file_name}.txt", mode="w") as f:
            f.writelines(txt_lst)
          with open(f"output_{attn_type}/{file_name}_loss.csv", mode="w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(loss_lst)
          print(f"end {file_name}  {count}/36")
          count += 1
          del transformer