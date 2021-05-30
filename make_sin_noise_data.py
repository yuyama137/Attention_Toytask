import numpy as np
import matplotlib.pyplot as plt


"""
"""
def makesin_noise():
    period = np.random.randint(50, 100)# 周期
    A = np.random.randint(50,100)# 振幅
    sft = np.random.randint(period)
    x_lst = np.arange(period)
    s = np.sin(2*np.pi*x_lst/period)
    snoise = s + 0.1*np.random.randn(period)
    snoise = (snoise * A).astype(np.int64)
    s = (s*A).astype(np.int64)
    return snoise, s

def make_data(batch, train=True, test_src=None):
    np.random.seed(seed=32)
    src_lst = []
    tgt_lst = []
    # src_mask_lst = []
    # tgt_mask_lst = []
    len_lst = []
    max_len = 0
    count = 0
    while count<batch:
        snoise, s = makesin_noise()
        if not train and (snoise not in test_src):# テストデータ作成時
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
    return src, tgt




if __name__ == "__main__":
    snoise, s = make_data(10)

    plt.plot(s[0])
    plt.plot(snoise[0])
    plt.show()




