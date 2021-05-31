import numpy as np
import matplotlib.pyplot as plt
import torch

A = 100
# データの範囲は、2<= data <= 220(含む)

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
#     np.random.seed(seed=32)
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

if __name__ == "__main__":
    src_test, tgt_test, src_lst  = make_data(10, False)
    src, tgt = make_data(10, True, src_lst)

    plt.plot(src[3])
    plt.plot(tgt[3, 1:])
    plt.show()



