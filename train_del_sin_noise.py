# import model
import copy_model
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import csv
import random

# constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_BATCHES = 5000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 1
# データの範囲は、2<= data <= 220(含む)
NUM_TOKENS = 230
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 32
attn_type = "linear_attn_elu"
SEED = 32
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if device != "cpu":
    torch.cuda.manual_seed(SEED)


def makesin_noise():
    period = np.random.randint(50, 100)# 周期
    A = np.random.randint(50,100)# 振幅
    sft = np.random.randint(period)
    x_lst = np.arange(period) - sft
    s = np.sin(2*np.pi*x_lst/period)
    snoise = s + 0.1*np.random.randn(period)
    snoise = (snoise * A).astype(np.int64)
    s = (s*A).astype(np.int64)
    snoise = snoise + A + 1
    s = s + A + 1
    snoise = np.clip(snoise, 2, 220)
    s = np.clip(s, 2, 220)
    return snoise, s

def check_test_data(train, test):
#     import pdb; pdb.set_trace()
    for t in test:
        if np.all(train==t):
            return False
    return True

def make_data(batch, train=True, test_lst=None):
    src_lst = []
    tgt_lst = []
    # src_mask_lst = []
    # tgt_mask_lst = []
    len_lst = []
    max_len = 0
    count = 0
    while count<batch:
        snoise, s = makesin_noise()
        if train:# trainデータ作成時
            if not check_test_data(snoise, test_lst):
                # print("this data was in test set")
                continue
        l = len(snoise)
        len_lst.append(l)
        if (l>max_len):
            max_len = l
        src_lst.append(snoise)
        tgt_lst.append(s)
        count += 1
    src = np.ones((batch, max_len))
    tgt = np.ones((batch, max_len + 1))
    tgt[:,0] = 1
    for i in range(batch):
        src[i, :len_lst[i]] = src_lst[i]
        tgt[i, 1:len_lst[i]+1] = tgt_lst[i]
    if train:
        return src, tgt
    else:
        return src, tgt, src_lst
        
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
pos_lst = [256, 1000]

test_src, test_tgt, test_src_lst = make_data(10, False)
test_src = torch.from_numpy(test_src).long().to(device)
test_tgt = torch.from_numpy(test_tgt).long().to(device)

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
                        # src, tgt, src_mask, tgt_mask = next(cycle())
                        src, tgt = make_data(BATCH_SIZE, True, test_src_lst)
                        # src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
                        src, tgt = torch.from_numpy(src).long().to(device), torch.from_numpy(tgt).long().to(device)
                        
                        out = transformer(src, tgt)
                        loss = culc_loss(criterion, out[:,:-1,:], src)
                        loss.backward()
                        # print(loss.item())

                        optim.step()
                        optim.zero_grad()

                        loss_lst.append(loss.item())

                        if i % GENERATE_EVERY == 0:
                            transformer.eval()
                            # src, _, src_mask, _ = next(cycle())
                            # src, src_mask = src[0:1].to(device), src_mask[0:1].to(device)
                            print("generationg")
                            
                            sample = transformer.generate(test_src)
                            # import pdb; pdb.set_trace()
                            # incorrects = (src != sample).abs().sum()
                            incorrects = torch.sum(test_tgt[:,1:] != sample)

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
                    for i in range(sample_num):
                        transformer.eval()
                        # src, _, src_mask, _ = next(cycle())
                        # src, src_mask = src[0:1].to(device), src_mask[0:1].to(device)

                        src = test_src[i,:]
                        tgt = test_tgt[i,1:]

                        sample = transformer.generate(src)
                        incorrects = torch.sum(tgt != sample)

                        txt_lst.append(f"input  : {tgt}\npredict: {sample}\nincorrects: {incorrects}\n\n")

                    with open(f"output_{attn_type}/{file_name}.txt", mode="w") as f:
                        f.writelines(txt_lst)
                    with open(f"output_{attn_type}/{file_name}_loss.csv", mode="w") as f:
                        writer = csv.writer(f, lineterminator="\n")
                        writer.writerow(loss_lst)
                    print(f"end {file_name}  {count}/36")
                    count += 1
                    del transformer