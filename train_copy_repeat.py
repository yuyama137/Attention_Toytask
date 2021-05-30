import model
import torch
from torch import nn

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 32

"""
src, tgt, src_mask, tgt_maskを生成。
予め、testを作成しておき、もしtestに生成した文字列が含まれていたら違うものを作るようにする。
"""
def cycle(test_src):
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long()
        tgt = torch.cat((prefix, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1] - 1).bool()

        yield (src, tgt, src_mask, tgt_mask)






if __name__ == "__main__":
    src, tgt, src_mask, tgt_mask = cycle()