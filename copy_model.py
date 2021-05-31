import torch
from torch import nn, einsum
import torch.optim as optim
import torch.nn.functional as F
import math
from functools import partial

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

def phi(x):
    """
    nonlinear function for linear attention, which is described in the paper.

    $$
    φ(x) = elu(x) + 1
    $$
    """
    return F.elu(x) + 1

def linear_attn_elu(q, k, v, eps = 1e-6):
    dim = q.shape[-1]

    q = phi(q)
    k = phi(k)

    kv = einsum('bhkd,bhve->bhed',k,v)
    up = einsum("bhqd,bhdd->bhqd",q,kv)
    blw = einsum("bhqd,bhkd->bhq",q,k)

    # import pdb; pdb.set_trace()

    _attn = up/(blw.unsqueeze(-1) + eps)
    attn = _attn/math.sqrt(dim)
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


def causal_linear_attn_elu(q, k, v, kv_mask = None, bucket_size = 1, eps = 1e-6):
    b, h, n, e, dtype = *q.shape, q.dtype
    # bucket_size = default(bucket_size, 64)
    # bucket_size = max(bucket_size, 1)
    assert bucket_size == 0 or (n % bucket_size) == 0, f'sequence length {n} must be divisible by the bucket size {bucket_size} for causal linear attention'

    # q = q.softmax(dim=-1)
    # k = torch.exp(k).type(dtype).clone()
    q = phi(q)
    k = phi(k)

    # q = q * e ** -0.5

    # if exists(kv_mask):
    #     mask = kv_mask[:, None, :, None]
    #     k = k.masked_fill_(~mask, 0.)
    #     v = v.masked_fill_(~mask, 0.)
    #     del mask

    # bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
    # b_q, b_k, b_v = map(bucket_fn, (q, k, v))

    # k_sum = k.sum(dim=-2) # bucket方向に和をとっている
    # import pdb; pdb.set_trace()
    
    k_cumsum = k.cumsum(dim = -2).type(dtype)

    kv = torch.einsum("bhnd,bhne->bhned", k, v)
    kv_cumsum = kv.cumsum(dim = -3).type(dtype)# (b, h, n, d e)

    # if bucket_size > 1:
    #     context = F.pad(context, (0, 0, 0, 0, 1, 0), value = 0.)
    #     context, _ = split_at_index(2, -1, context)

    #     b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value = 0.)
    #     b_k_cumsum, _ = split_at_index(2, -1, b_k_cumsum)

    # blw = einsum("bhnd,bhnd->bhn", q, k_cumsum + eps)
    
    # blw = torch.mul(k_cumsum,q).sum(dim=-2)
    blw = einsum("bhnd,bhnd->bhn", q, k_cumsum)
    up = einsum("bhnd,bhnde->bhne", q, kv_cumsum)
    # D_inv = 1. / einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    # attn = einsum('bhund,bhude,bhun->bhune', b_q, context, D_inv)
    _attn = up/(blw.unsqueeze(-1) + eps)
    attn = _attn/math.sqrt(e)
    return attn.reshape(*q.shape)

def full_attention(query, key, value, causal=False, dropout=0.0):
    """
    Scale Dot-Product Attention (論文Fig.2)

    inputs:
      - query (torch.tensor) (B, h, n, d)
      - key (torch.tensor) (B, h, n, d)
      - value (torch.tensor) (B, h, n, d)
      - causal (bool) : Trueの時、時間マスク(三角行列)を使用
      - dropout (float) : ドロップアウトの割合(使用するなら)
    
    return:
      - out (torch.tensor) (B, h, n, d)
    """
    device = key.device
    B_k, h_k, n_k, d_k = key.shape
    B_q, h_q, n_q, d_q = query.shape
    # import pdb; pdb.set_trace()

    scale = einsum("bhqd,bhkd->bhqk", query, key)/math.sqrt(d_k)

    if causal:
        # マスクを作る(下三角行列)
        ones = torch.ones(B_k, h_k, n_q, n_k).to(device)
        mask = torch.tril(ones)
        scale = scale.masked_fill(mask == 0, -1e9)# -infで埋めるイメージ。めちゃめちゃ確率小さくなる
    atn = F.softmax(scale, dim=-1)
    if dropout is not None:# ここにはさむべき？？
        atn = F.dropout(atn, p=dropout)   
    # out = torch.matmul(atn, value)
    out = einsum("bhqk,bhkd->bhqd", atn, value)
    return out

def to_eachhead(x, head_num, split_num=3):
    """
    入力テンソルをsplit_num個に分割(3の時qvk)して、ヘッドに分割

    (B, n, D) -> (B, n, d) x split_num -> (B, h, n, d')

    ただし、D = d x split_num

    - inputs
        - x (torch.tesor) : (B, n, 3d) output of self.qvk
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
    B, n, pre_d = x.shape
    new_d = pre_d//split_num
    assert pre_d%split_num == 0, f"have to be multiple of {split_num}"
    assert new_d%head_num == 0, "dim must be divided by head_num"

    tpl = torch.chunk(x, split_num, dim=2)
    out = []
    for t in tpl:
        out.append(t.reshape(B, n, head_num, new_d//head_num).transpose(1,2))
    return out

def concat_head(x):
    """
    ヘッドをもとに戻す

    - inputs
        - x (torch.tensor) : (B, h, n, d')
    - outputs
        - out (torch.tensor) : (B, n, d) (d = d' x h)
    """
    B, h, n, _d = x.shape
    out = x.transpose(1,2).reshape(B, n, _d*h)
    return out

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
        den = 1/torch.pow(torch.ones(int(d_model/2))*max_len,2*tmp/d_model)
        den = den.unsqueeze(0)
        self.pe[:,0::2] = torch.sin(torch.matmul(pos,den))
        self.pe[:,1::2] = torch.cos(torch.matmul(pos,den))
        self.pe = self.pe.to(device)

    def forward(self, x):
        return x + self.pe[:x.shape[1],:]

class CopyModel(nn.Module):
    """
    コピータスク専用のTransformerモデル。(マスクは考えない)
    position -> encoder -> decoder -> finallayer(最後にsoftmaxしない)

    - args:
        - device (str) : cpu or gpu name
        - ed (int) : 潜在次元数
        - vocab_num (int) : number of vcab
        - attn_type (str) : type of attention ("linear", "full")
        - N_enc (int) : number of encoderlayer
        - N_dec (int) : number of decoderlayer
        - h_enc (int) : number of multihead in encoder
        - h_dec (int) : number of multihead in decoder

    - inputs:
        - x (torch.tensor) : (B, len_x)
        - y (torch.tensor) : (B, len_y) 先頭は<sos>

    - outputs:
        - out (torch.tensor) : (B, len_gen)
    """
    def __init__(self, device, ed, vocab_num, attn_type, N_enc, N_dec, h_enc, h_dec, ff_hidnum, max_len):
        super().__init__()
        self.device = device
        self.x_emb = nn.Embedding(vocab_num, ed)
        self.y_emb = nn.Embedding(vocab_num, ed)
        self.pos = PositionalEncoding(ed, 0.0, device, max_len) # src, tgt共通
        self.enc = Encoder(N_enc, ed, h_enc, attn_type, ff_hidnum)
        self.dec = Decoder(N_dec, ed, h_dec, attn_type, ff_hidnum)
        self.fin = FinalLayer(ed, vocab_num, ff_hidnum)

    def forward(self, x, y):
        # import pdb; pdb.set_trace()
        x_emb = self.x_emb(x)
        y_emb = self.y_emb(y)
        x_emb_pos = self.pos(x_emb)
        y_emb_pos = self.pos(y_emb)
        memory = self.enc(x_emb_pos)
        out = self.dec(y_emb_pos, memory)
        out = self.fin(out)
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
        x = self.x_emb(x)
        x = self.pos(x)
        z = self.enc(x)
        y = torch.ones(size=(B, 1)).long().to(device)
        # import pdb; pdb.set_trace()
        for i in range(l):
            # mask = make_mask(y.shape[1])
            tmp_y = self.y_emb(y)
            tmp_y = self.pos(tmp_y)
            tmp_y = self.dec(tmp_y, z)
            tmp_y = self.fin(tmp_y)
            next_word = torch.max(tmp_y[:,-1,:],dim=-1)[1]
            # import pdb; pdb.set_trace()
            y = torch.cat([y,next_word.unsqueeze(1)],dim = -1)
        return y[:,1:]

class FinalLayer(nn.Module):
    """
    出力の直前の層
    output of transformer -> linear -> output
    nn.CrossEntropyでソフトマックスを行うので、ここでは実装しない

    args:
      - dim (int) : 特徴次元
      - vocab_num (int) : 語彙数
      - hif_dim (int) : 中間層のユニット数
    """
    def __init__(self, dim, vocab_num, hid_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, vocab_num)
    
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out

class Encoder(nn.Module):
    """
    コピータスクのエンコーダ
    EncoderLayerを所望の数積み重ねる

    - args:
        - depth : 層の数
        - dim : 潜在次元数
        - head_num : ヘッド数
        - attn_type : linear -> LinearAttention / full -> Vannila
        - ff_hidnum : feedforwardにおける潜在次元数

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - x : (torch.tensor) : (B, N, D)
    """
    def __init__(self, depth, dim, head_num, attn_type="linear", ff_hidnum=2048):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, attn_type, head_num, ff_hidnum) for i in range(depth)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    """
    コピータスクのエンコーダレイヤー
    selfattention -> feedforward
    residual passとそれに伴ったLayerNormを実装

    - args:
        - dim : 潜在次元数
        - attn_type : attentionのタイプ
        - head_num : ヘッド数
        - ff_hidnum (int) : feedforwardでの隠れ層の次元

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num, ff_hidnum):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim, attn_type, head_num)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidnum)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        res = torch.clone(x)
        out = self.mhsa(x)
        out = out + res
        out = self.ln1(out)
        res = torch.clone(out)
        out = self.ff(out)
        out = out + res
        out = self.ln2(out)
        return out

class FeedForward(nn.Module):
    """
    feedforwad module. 2層のaffine層

    - args:
        - dim (int)
        - hid_dim (int)

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, hid_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hid_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, dim, bias=True)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    multiheadselfattention
    head増やす(B, H, N, D) -> selfattention function -> output

    - args:
        - dim (int) : 
        - attn_type (str) : linear -> LinearAttention / full -> Vannila
        - head_num (int) : 
    
    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim*3)
        self.make_head = partial(to_eachhead, head_num=head_num, split_num=3)
        if attn_type == "linear":
            self.mhsa = linear_attn
        elif attn_type == "full":
            self.mhsa = full_attention
        elif attn_type == "linear_attn_elu":
            self.mhsa = linear_attn_elu
        else:
            NotImplementedError("attention type of {} is not defined. Please set linear or full".format(attn_type))
    
    def forward(self, x):
        qvk = self.to_qvk(x)
        q, v, k = self.make_head(qvk)
        out = self.mhsa(q, k, v)
        out = concat_head(out)
        return out

class MultiHeadCausalAttention(nn.Module):
    """
    Causal attentionをやります。
    head増やす(B, H, N, D) -> causalattention function -> output

    - args:
        - dim (int) : 特徴次元数
        - attn_type (str) : linear -> LinearAttention / full -> Vannila
        - head_num (int) : ヘッド数
    
    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim*3)
        self.make_head = partial(to_eachhead, head_num=head_num, split_num=3)
        if attn_type == "linear":
            self.mhca = partial(causal_linear_attn, bucket_size=1)
        elif attn_type == "full":
            self.mhca = partial(full_attention, causal=True)
        elif attn_type == "linear_attn_elu":
            self.mhca = partial(causal_linear_attn_elu, bucket_size=1)
        else:
            NotImplementedError("attention type of {} is not defined. Please set linear or full".format(attn_type))

    def forward(self, x):
        qvk = self.to_qvk(x)
        q, v, k = self.make_head(qvk)
        out = self.mhca(q, k, v)
        out = concat_head(out)
        return out

class MultiHeadSourceAttention(nn.Module):
    """
    source attention. this is for attention using output of encoder(memory). 

    - args:
        - dim (int) : 特徴次元数
        - attn_type (str) : linear -> LinearAttention / full -> Vannila
        - head_num (int) : ヘッド数

    - inputs:
        - x (torch.tensor) : (B, N, D) input tensor
        - memory (torch.tensor) : (B, N, D) output of encoder

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num):
        super().__init__()
        self.to_kv = nn.Linear(dim, dim*2)
        self.to_q = nn.Linear(dim, dim)
        self.make_head_kv = partial(to_eachhead, head_num=head_num, split_num=2)
        self.make_head_q = partial(to_eachhead, head_num=head_num, split_num=1)
        if attn_type == "linear":
            self.mhsa = linear_attn
        elif attn_type == "full":
            self.mhsa = full_attention
        elif attn_type == "linear_attn_elu":
            self.mhsa = linear_attn_elu
        else:
            raise NotImplementedError("attention type of {} is not defined. Please set linear or full".format(attn_type))

    def forward(self, x, memory):
        mem = self.to_kv(memory)
        x = self.to_q(x)
        k, v = self.make_head_kv(mem)
        q = self.make_head_q(x)[0]
        out = self.mhsa(q, k, v)
        out = concat_head(out)
        return out

class Decoder(nn.Module):
    """
    コピータスクのデコーダ
    DecoderLayerを所望の数積み重ねる

    - args:
        - depth : 層の数
        - dim : 潜在次元数
        - head_num : ヘッド数
        - attn_type : linear -> LinearAttention / full -> Vannila
        - ff_hidnum : feedforwardにおける潜在次元数

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - x : (torch.tensor) : (B, N, D)
    """
    def __init__(self, depth, dim, head_num, attn_type="linear", ff_hidnum=2048):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, attn_type, head_num, ff_hidnum) for i in range(depth)])
    
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x

class DecoderLayer(nn.Module):
    """
    コピータスクのデコーダレイヤー
    (self)causalattention -> sourceattention -> feedforward
    residual passとそれに伴ったLayerNormを実装

    - args:
        - dim (int) : 潜在次元数
        - attn_type (str) : attentionのタイプ
        - head_num (int) : ヘッド数
        - ff_hidnum (int) : feedforwardでの隠れ層の次元

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num, ff_hidnum):
        super().__init__()
        self.mhca = MultiHeadCausalAttention(dim, attn_type, head_num)
        self.ln1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSourceAttention(dim, attn_type, head_num)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidnum)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x, memory):
        res = torch.clone(x)
        out = self.mhca(x)
        out = out + res
        out = self.ln1(out)
        res = torch.clone(out)
        out = self.mhsa(out, memory)
        out = out + res
        out = self.ln2(out)
        res = torch.clone(out)
        out = self.ff(out)
        out = out + res
        out = self.ln3(out)
        return out


        

