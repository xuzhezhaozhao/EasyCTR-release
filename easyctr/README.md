
# DSSM

双塔结构，通过特征配置文件可以指定特征放在哪个特征塔，如下:

```
# 用户 ID 放在左塔
transform.add_categorical_identity('u_omgid', [])
transform.add_embedding('u_omgid.identity', ['dssm1'], 200)

# 物品 ID 放在右塔
transform.add_embedding('i_rowkey.identity', ['dssm2'], 100)
```

最后一层计算相似度有多种计算方法： dot, concat, cosine. 通过下面的方法配置：
```
model_type='dssm'

dssm_mode='dot'
dssm_hidden_units=512,256,128
dssm_activation_fn=relu
dssm_dropout=0.0
dssm_batch_norm=false
dssm_layer_norm=false
dssm_use_resnet=false
dssm_use_densenet=false
```

# DeepX

自由组装模式，提供一些基础模型，用户可以自由组装，达到模型 Ensemble 的效果。

举几个例子：
- Wide&Deep 是 LR 和 DNN 组装;
- DeepFM 是 LR, DNN, FM 模型组装;

有几个注意的点：
- Linear 部分使用单独优化器，其他模型共用优化器;
- Linear 特征不是默认包含的，需要在特征配置文件中将 places 设置为 wide 才会生效;


## 基本模型
首先介绍基本模型。

### LR

```
model_type=linear
```

### FM
不包括线性层, 下面的 FM 系列同样。
```
model_type=fm

fm_use_shared_embedding=true    # embedding 是否共享
fm_use_project=false            # 一个trick, FM 要求所有特征 embedding 维度一致，
                                # 但可以线性变换将 adaptive embedding 投影为相同维度
fm_project_size=32              # 投影维度
```

### FwFM
给 N\*(N-1)/2 个特征交叉添加单独的权重，实现时最后一层使用 FC.
```
model_type=fwfm

fwfm_use_shared_embedding=true
fwfm_use_project=false
fwfm_project_size=32
```

### AFM
在 FwFM 的基础上更进一步，使用 Attention 自动学习 N\*(N-1)/2 个特征交叉的权重。
```
model_type=afm

afm_use_shared_embedding=true
afm_use_project=false
afm_project_size=32
afm_hidden_unit=32
```

### IAFM
在 AFM 的基础上更进一步，不仅使用 Attention 学习特征交叉的权重，还额外学习一个
Field Aspect 向量(实现时分解为两个矩阵相乘)，进一步表征不同特征交叉的权重。
```
model_type=iafm

iafm_use_shared_embedding=true
iafm_use_project=false
iafm_project_size=32
iafm_hidden_unit=32
iafm_field_dim=8
```

### IFM
同 IAFM，但不使用 Attention.
```
model_type=ifm

ifm_use_shared_embedding=true
ifm_use_project=false
ifm_project_size=32
ifm_hidden_unit=32
ifm_field_dim=8
```

### KFM
Kernel FM, 原本的 FM 使用向量点乘表示特征相互作用，KFM 使用 Kernel Product，具体
为将 vi\*vj 改为 vi\*U\*vj，U 为 Kernel 矩阵，Kernel Product 不要求 vi, vj 向量维度
相同，所以可以直接使用 adaptive embedding size.

特征相互作用个数为 N\*N.
```
model_type=kfm

kfm_use_shared_embedding=true
kfm_use_project=false
kfm_project_size=32
```

### WKFM
思路同 FwFM，给 N*N 个特征相互作用添加 weight.
```
model_type=wkfm

wkfm_use_shared_embedding=true
wkfm_use_project=false
wkfm_project_size=32
```

### NIFM
使用 micro network 替代 dot product.
```
model_type=nifm

nifm_use_shared_embedding=true
nifm_use_project=false
nifm_project_size=32
nifm_hidden_units=32,16
nifm_activation_fn=relu
nifm_dropout=0.0
nifm_batch_norm=false
nifm_layer_norm=false
nifm_use_resnet=false
nifm_use_densenet=false
```

### CIN
xDeepFM 提出的结构。
```
model_type=cin

cin_use_shared_embedding=true
cin_use_project=false
cin_project_size=32
cin_hidden_feature_maps=128,128
cin_split_half=true
```

### Cross
Deep&Cross 提出的结构。
```
model_type=cross

cross_use_shared_embedding=true
cross_use_project=false
cross_project_size=32
cross_num_layers=4
```

### AutoInt
使用 Multihead-Attention, 不包括 DNN.
```
model_type=autoint

autoint_use_shared_embedding=true
autoint_use_project=false
autoint_project_size=32
autoint_size_per_head=16
autoint_num_heads=6
autoint_num_blocks=2
autoint_dropout=0.0
autoint_has_residual=true
```

### AutoInt+
包括 DNN 部分, autoint + dnn 组装。
```
model_type=autoint+
```

### DNN
```
model_type=dnn

dnn_use_shared_embedding=true
dnn_use_project=false
dnn_project_size=32
dnn_hidden_units=512,256,128
dnn_activation_fn=relu
dnn_dropout=0.0
dnn_batch_norm=false
dnn_layer_norm=false
dnn_use_resnet=false
dnn_use_densenet=false
```

### NFM
Neural FM
```
model_type=nfm

nfm_use_shared_embedding=true
nfm_use_project=false
nfm_project_size=32
nfm_hidden_units=512,256,128
nfm_activation_fn=relu
nfm_dropout=0.0
nfm_batch_norm=false
nfm_layer_norm=false
nfm_use_resnet=false
nfm_use_densenet=false
```

### NKFM
Neural Kernel FM, 类似于 NFM, 在 N*N 个特征交互值后接多层 MLP.
```
model_type=nkfm

nkfm_use_shared_embedding=true
nkfm_use_project=false
nkfm_project_size=32
nkfm_hidden_units=512,256,128
nkfm_activation_fn=relu
nkfm_dropout=0.0
nkfm_batch_norm=false
nkfm_layer_norm=false
nkfm_use_resnet=false
nkfm_use_densenet=false
```

### CCPM
使用卷积提取特征，结果跟特征的输入顺序有关。
```
model_type=ccpm

ccpm_use_shared_embedding=true
ccpm_use_project=false
ccpm_project_size=32
ccpm_hidden_units=512,256,128
ccpm_activation_fn=relu
ccpm_dropout=0.0
ccpm_batch_norm=false
ccpm_layer_norm=false
ccpm_use_resnet=false
ccpm_use_densenet=false
ccpm_kernel_sizes=3,3,3
ccpm_filter_nums=4,3,2
```


### IPNN
Inner-Product Neural Network
```
model_type=ipnn

ipnn_use_shared_embedding=true
ipnn_use_project=false
ipnn_project_size=32
ipnn_hidden_units=512,256,128
ipnn_activation_fn=relu
ipnn_dropout=0.0
ipnn_batch_norm=false
ipnn_layer_norm=false
ipnn_use_resnet=false
ipnn_use_densenet=false
ipnn_unordered_inner_product=false
ipnn_concat_project=false
```

### KPNN
Kernel-Product Neural Network
```
model_type=kpnn

kpnn_use_shared_embedding=true
kpnn_use_project=false
kpnn_project_size=32
kpnn_hidden_units=512,256,128
kpnn_activation_fn=relu
kpnn_dropout=0.0
kpnn_batch_norm=false
kpnn_layer_norm=false
kpnn_use_resnet=false
kpnn_use_densenet=false
kpnn_concat_project=false
```

### PIN
Product-network In Network
```
model_type=pin

pin_use_shared_embedding=true
pin_use_project=false
pin_project_size=32
pin_hidden_units=512,256,128
pin_activation_fn=relu
pin_dropout=0.0
pin_batch_norm=false
pin_layer_norm=false
pin_use_resnet=false
pin_use_densenet=false
pin_use_concat=false
pin_concat_project=false
pin_subnet_hidden_units=64,32
```

### FiBiNET
```
model_type=fibinet

fibinet_use_shared_embedding=true
fibinet_use_project=false
fibinet_project_size=32
fibinet_hidden_units=512,256,128
fibinet_activation_fn=relu
fibinet_dropout=0.0
fibinet_batch_norm=false
fibinet_layer_norm=false
fibinet_use_resnet=false
fibinet_use_densenet=false
fibinet_use_se=true
fibinet_use_deep=true
fibinet_interaction_type='bilinear'
fibinet_se_interaction_type='bilinear'
fibinet_se_use_shared_embedding=false
```

## 自由组装模式
比如需要 DeepFM 模型，可以由下面的配置得到:
```
model_type=deepx            # 自由组装模式
model_slots=linear,fm,dnn   # 需要组装的模型, 基本模型的参数单独配置
```

xDeepFM 模型:
```
model_type=deepx
model_slots=linear,cin,dnn
```

KFM + KPNN 组装:
```
model_type=deepx
model_slots=linear,kfm,kpnn
```

对一些常用的组装模型做了预设，包括:
- wide/lr/linear
- dnn/deep
- wdl/wide_deep
- fm
- fwfm
- afm
- iafm
- ifm
- kfm
- wkfm
- nifm
- cross
- cin
- autoint
- autoint+
- nfm
- nkfm
- ccpm
- ipnn
- kpnn
- pin
- fibinet
- deepfm
- deepfwfm
- deepfwfm
- deepafm
- deepiafm
- deepifm
- deepkfm
- deepwkfm
- deepnifm
- xdeepfm
- dcn/deepcross

model_type 填上述字段就行。

## Ensemble 模式
默认情况下，组装模式所有模型是分别算出 logits，先求和得到总 logits，再计算 loss，
这样可能会存在大 logits“吃掉 小 logits 的情况，导致某些分类错误的模型由于梯度过小
而没有得到更新。

为了解决这种问题，我们提出单独计算 loss 并更新对应模型的方法，可以通过下面的参数
配置：
```
use_seperated_logits=true
```

## Weighted logits 模式
除了 Ensemble 模式，还提供量一种 Weighted Logits 模式，即各个模型的 logits 不是
简单求和，而是加权求和，权重是一个可学习变量，通过下面的参数配置:
```
use_weighted_logits=true
```

注意：`use_seperated_logits` 和 `use_weighted_logits` 两个参数不能同时设置为 true.
