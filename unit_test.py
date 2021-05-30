import torch
import torch.optim as optim

from linear_attention_transformer.linear_attention_transformer import SelfAttention, linear_attn

from linear_attention_transformer import LinearAttentionTransformerLM
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper
import model



out = model.to_eachhead()



