import torch
from torch import nn, einsum
import torch.optim as optim
import torch.nn.functional as F
import math

# from linear_attention_transformer.linear_attention_transformer import SelfAttention, linear_attn

# from linear_attention_transformer import LinearAttentionTransformerLM
# from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# [ref](https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py)
def linear_attn(q, k, v, kv_mask = None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    q = q * dim ** -0.5

    # import pdb; pdb.set_trace()

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)

# [ref](https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py)
def causal_linear_attn(q, k, v, kv_mask = None, bucket_size = None, eps = 1e-6):
    b, h, n, e, dtype = *q.shape, q.dtype
    bucket_size = default(bucket_size, 64)
    bucket_size = max(bucket_size, 1)
    assert bucket_size == 0 or (n % bucket_size) == 0, f'sequence length {n} must be divisible by the bucket size {bucket_size} for causal linear attention'

    q = q.softmax(dim=-1)
    k = torch.exp(k).type(dtype).clone()

    q = q * e ** -0.5

    if exists(kv_mask):
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, 0.)
        v = v.masked_fill_(~mask, 0.)
        del mask

    bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
    b_q, b_k, b_v = map(bucket_fn, (q, k, v))

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

    context = einsum('bhund,bhune->bhude', b_k, b_v)
    context = context.cumsum(dim = -3).type(dtype)

    if bucket_size > 1:
        context = F.pad(context, (0, 0, 0, 0, 1, 0), value = 0.)
        context, _ = split_at_index(2, -1, context)

        b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value = 0.)
        b_k_cumsum, _ = split_at_index(2, -1, b_k_cumsum)

    D_inv = 1. / einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = einsum('bhund,bhude,bhun->bhune', b_q, context, D_inv)
    return attn.reshape(*q.shape)

def concat_head(x):
    """
    - inputs
        - x (torch.tensor) : (B, n, 3d)(B, h, n, d')
    - outputs
        - out (torch.tensor) : (B, n, d)
    """
    B, h, n, _d = x.shape
    out = x.transpose(1,2).reshape(B, n, _d*h)
    return out


# def full_attention():


def to_eachhead(x, head_num, split_num=3):
    """
    self.qvkの出力をqvkそれぞれに分割をして、それぞれをヘッドごとに分割する。

    (B, n, 3d) -> (B, n, d) x 3 (qvk) -> (B, h, n, d')

    - inputs
        - x (torch.tesor) : (B, n, 3d) output of qvk
        - head_num : head数
        - split_num : 分割数、qvkに分割する場合は、split_num=3
    - outpus
        - out (list)
            - out = [q, v, ...(split num)]
                - q (torch.tensor) : (B, h, n, d')
                - v (torch.tensor) : (B, h, n, d')
                - k (torch.tensor) : (B, h, n, d')
                    - ただしd'はマルチヘッドアテンションを行う時の次元数
    """
    B, n, _d = x.shape
    d = _d//split_num
    assert _d%split_num == 0, f"have to be multiple of {split_num}"
    assert d%head_num == 0, "dim must be divided by head_num"

    tpl = torch.chunk(x, split_num, dim=2)
    out = []
    for t in tpl:
        out.append(t.reshape(B, n, head_num, d//head_num).transpose(1,2))
    # q = q.reshape(B, n, head_num, d//head_num).transpose(1,2)
    # v = v.reshape(B, n, head_num, d//head_num).transpose(1,2)
    # k = k.reshape(B, n, head_num, d//head_num).transpose(1,2)
    # return q, v, k
    return out

class FeedForward(nn.Module):
    """
    単語ごとの全結合層

    - args:
      - d (int) : 次元数
      - hid_dim (int) : 隠れ層の次元数(2048デフォルト)
      - dropout (float) : dropout ratio
    """
    def __init__(self, d, hid_dim=2048, dropout=0.0):
        super().__init__()
        self.l1 = nn.Linear(d, hid_dim)
        self.l2 = nn.Linear(hid_dim, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(F.relu(self.l1(x))))

class EncoderLayer(nn.Module):
    """
    Encoder Layer maskはないことを前提に進める

    - args:
        - dim : 潜在次元数
        - h : head num
    - inputs
        - x (torch.tensor) : (B x n x d)
        - mask : マスク
    - outputs
        - out (torch.tensor) : (B x n x d)

    """
    def __init__(self, dim, h):
        super().__init__()
        self.head_num = h
        self.to_qvk = nn.Linear(dim, dim*3)
        self.ln1 = nn.LayerNorm(dim)
        self.mhattn = linear_attn
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)
    
    def forward(self, x, mask=None):
        res = torch.clone(x)
        qvk = self.to_qvk(x)
        q, v, k = to_eachhead(qvk, self.head_num)
        out = self.mhattn(q, k, v)# out.shape : (B, h, n, d')
        out = concat_head(out)
        out = out + res
        out = self.ln1(out) # out : (B, length, dim)

        res = torch.clone(out)
        out = self.ff(out)
        out = out + res
        out = self.ln2(out)
        return out

"""
encoder

- args:
    - depth
    - dim
    - head_num
- inputs:
    - x
- outpus:
    - out
"""
class Encoder(nn.Module):
    def __init__(self, depth, dim, head_num):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, head_num) for i in range(depth)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    """
    Each Decoder Layer

    - args
        - dim
        - h
    - inputs
        - x (torch.tensor) : 
        - memory (torch.tensor) : 
        - mask (torch.tensor) : 
    - outputs
    """
    def __init__(self, dim, h):
        super().__init__()
        self.head_num = h
        self.to_qvk = nn.Linear(dim, dim*3)
        self.mhsattn = causal_linear_attn
        self.ln1 = nn.LayerNorm(dim)

        self.to_kv = nn.Linear(dim, dim*2)
        self.to_q = nn.Linear(dim, dim)
        self.mhattn = linear_attn
        self.ln2 = nn.LayerNorm(dim)

        self.ff = FeedForward(dim)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x, memory):
        res = torch.clone(x)
        qvk = self.to_qvk(x)
        q, v, k = to_eachhead(qvk, self.head_num)
        # import pdb; pdb.set_trace()
        out = self.mhsattn(q, k, v, bucket_size=1)# out.shape : (B, h, n, d')
                                                  # bucket_sizeは1にしておく。まだよくわかっていない。。。[ref](https://github.com/lucidrains/linear-attention-transformer#usage)
                                                  # のblindspot_sizeのコメントによると、少なくとも1にしておけば「full q(kv) attention of past」らしい。
        out = concat_head(out)
        out = out + res
        out = self.ln1(out) # out : (B, length, dim)

        res = torch.clone(out)
        kv = self.to_kv(memory)
        # import pdb; pdb.set_trace()
        k, v = to_eachhead(kv, self.head_num, 2)
        q = self.to_q(out)
        q = to_eachhead(q, self.head_num, 1)[0]
        # import pdb; pdb.set_trace()
        out = self.mhattn(q,k,v)
        out = concat_head(out)
        out = out + res
        out = self.ln2(out)

        res = torch.clone(out)
        out = self.ff(out)
        out = out + res
        out = self.ln3(out)
        return out

class Decoder(nn.Module):
    def __init__(self, depth, dim, head_num):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, head_num) for i in range(depth)])
    
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x

class PositionalEncoding(nn.Module):
    """
    位置エンコーディング

    args:
      - d_model (int) : ベクトルの次元数
      - dropout (float)
      - device
      - max_len (int) : 許容しうる最大の長さの文章
    """
    def __init__(self, d_model, dropout, device="cpu", max_len = 10000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).type(torch.float32)
        tmp = torch.arange(0,d_model,2)
        den = 1/torch.pow(torch.ones(int(d_model/2))*10000,2*tmp/d_model)
        den = den.unsqueeze(0)
        self.pe[:,0::2] = torch.sin(torch.matmul(pos,den))
        self.pe[:,1::2] = torch.cos(torch.matmul(pos,den))
        self.pe = self.pe.to(device)

    def forward(self, x):
        return x + self.pe[:x.shape[1],:]

class Generator(nn.Module):
    """
    最終層。モデルの出力と文字を対応させる。
    output of model -> linear -> softmax -> output probability

    args:
      - d_model (int) : 
      - vocab_num (int) : 
    """
    def __init__(self, d_model, vocab_num):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 2048)
        self.linear2 = nn.Linear(2048, vocab_num)
    def forward(self, x):
        # return F.softmax(self.linear(x), dim=-1)
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out

class Model(nn.Module):
    """
    モデル全体
    input of model (word of id) -> embedding -> PositionalEncoding
                        -> EncoderDecoder -> Generator -> probability of words

    args:
      - device
      - d_model
      - vocab_num (int) : 全語彙数
      - dropout
      - N_enc (int) : number of encoderlayer
      - N_dec (int) : number of decoderlayer
      - h_enc (int) : number of multihead in encoder
      - h_dec (int) : number of multihead in decoder
    """
    def __init__(self, device, d_model, vocab_num, dropout, N_enc, N_dec, h_enc, h_dec):
        super().__init__()
        self.vocab_num = vocab_num
        self.emb_x = Embedding(vocab_num, d_model)
        self.pos_enc_x = PositionalEncoding(d_model, dropout, device)
        self.emb_y = Embedding(vocab_num, d_model)
        self.pos_enc_y = PositionalEncoding(d_model, dropout, device)
        self.enc_dec = EncoderDecoder(d_model, N_enc, N_dec, h_enc, h_dec)
        self.gen = Generator(d_model, vocab_num)
    
    def forward(self, x, y, mask=None):
        """
        args:
          - x (torch.tensor) (B x len) : それぞれの文章(id)
          - y (torch.tensor) (B x len) : それぞれの文章(id)
          - mask (torch.tensor) (len x len) : マスク(カンニング部は0で埋め、それ以外は1)
        output:
          - x (torch.tensor) (B x len) : 変換後のそれぞれの文章(id)
        """
        x = self.emb_x(x)
        x = self.pos_enc_x(x)
        y = self.emb_y(y)
        y = self.pos_enc_y(y)# 一番最初の要素は1でstart of sequenceだけど、xと同じようにやって良い？ずらさなくてよい？
        out = self.enc_dec(x, y, mask)
        out = self.gen(out)
        return out
    
    def generate(self, x, z_def=1):
        """
        自己回帰的に生成する.
        簡易的に、所望の長さになったら終了するように実装する。本来は<eos>が出るまで。
        args:
            - z_dec (int) : デコーダに入力する最初の文字
        """
        device = x.device
        B, l= x.shape
        x = self.emb_x(x)
        x = self.pos_enc_x(x)
        z = self.enc_dec.encode(x)
        y = torch.ones(size=(B, 1)).long().to(device)
        for i in range(l):
            # mask = make_mask(y.shape[1])
            tmp_y = self.emb_y(y)
            tmp_y = self.pos_enc_y(tmp_y)
            tmp_y = self.enc_dec.decode(tmp_y, z)
            tmp_y = self.gen(tmp_y)
            next_word = torch.max(tmp_y[:,-1,:],dim=-1)[1]
            # import pdb; pdb.set_trace()
            y = torch.cat([y,next_word.unsqueeze(1)],dim = -1)
        return y[:,1:]
        # return y
	
    def check_attention(self, x, h):
        x = self.emb_x(x)
        x = self.pos_enc_x(x)
        atn = self.enc_dec.get_attention(x,h)
        return atn

class EncoderDecoder(nn.Module):
    """
    メインモデル  
    embedded data -> encoder -> decoder -> embededd data

    args:
      - d_model (int) : ベクトルの次元
      - N_enc (int) : encoder layerの数
      - N_dec (int) : decoder layerの数
      - h_enc (int) : エンコーダのマルチヘッドの分割数
      - h_dec (int) : デコーダのマルチヘッドの分割数  
    """
    def __init__(self, d_model=512, N_enc=6, N_dec=6, h_enc=8, h_dec=8):
        super(EncoderDecoder, self).__init__()
        self.encode = Encoder(N_enc, d_model, h_enc)
        self.decode = Decoder(N_dec, d_model, h_dec)
    
    def forward(self, x_emb, y_emb, mask=None):
        """
        args:
          - x_emb (torch.tensor) (B x len x d) : 入力データ(位置エンコーディングまで終わったもの)
          - y_emb (torch.tensor) (B x len x d) : 出力データ(位置エンコーディングまで終わったもの)
        """
        z = self.encode(x_emb)
        y_out = self.decode(y_emb, z)
        return y_out

class Embedding(nn.Module):
    """
    input と output で別のインスタンスにする必要がある
    args:
      - vocab_num (int) : 語彙数
      - d_model (int) : 埋め込みベクトルの次元数
    """
    def __init__(self, vocab_num, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_num, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

if __name__ == "__main__":
    device = "cpu"
    B, n, d = 2, 3, 6
    head_num = 2
    depth = 2
    TOKEN_NUM = 10
    x = torch.randint(0, TOKEN_NUM-1, (2, 3))
    y = torch.clone(x)
    m = Model(device, d, TOKEN_NUM, 0.0, depth, depth, head_num, head_num)
    print(m.generate(x).shape)

