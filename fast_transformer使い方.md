# fast-transformerの使い方についてのまとめ

[github](https://github.com/idiap/fast-transformers)

## Attention

```fast_transformers.attention.attention_layer.AttentionLayer(attention, d_model, n_heads, d_keys=None, d_values=None)
```

#### arguments

- attention: attention(インスタンス) attention builderで作成
- d_model: 入力xの次元数
- n_heads: ヘッド数
- d_keys: keyの次元数 (default = d_model/n_heads)
- d_values: valueの次元数 (default = d_model/n_heads)

 ## Builder

 ### Attention Builder

```py
from fast_transformers.builders import AttentionBuilder

builder = AttentionBuilder.from_kwargs(
    attention_dropout=0.1,                   # used by softmax attention
    softmax_temp=1.,                         # used by softmax attention
    feature_map=lambda x: (x>0).float() * x  # used by linear
)
softmax = builder.get("full")
linear = builder.get("linear")
```



