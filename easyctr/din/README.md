
# Deep Interest Network

参考论文: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)


主要是实现了 attention 结构, 通过特征配置文件定义 attention 结构：
```
# 物品 ID 和用户浏览历史做 attention pooling
transform.add_attention(['u_history'], 'i_rowkey', 100)

# 指定 attention 结构细节
transform.add_attention(
    ['u_history'], 'i_rowkey', 100,
    attention_type='mlp',
    attention_args={'attention_hidden_units': [128, 64, 32]}
)
```


模型配置文件参数：
```
model_type='din'
```

# TODO
- 目前的实现是纯 deep 网络，结合 Linear 和 FM 部分。
