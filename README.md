# EasyCTR
封装主流 CTR 模型，配置化的训练方式，统一的 serving 接口。

# 特色
- 简单易用，用户只需要提供几个配置文件即可使用，包括特征处理、模型选择、超参调节等等
- 模型众多，几乎包含所有主流 CTR 模型，简单易用，支持灵活组装各种模型
- 端到端的 Ensemble 模型训练
- 线上、线下特征一致，这是因为特征处理逻辑在模型中完成
- 训练数据为原始特征格式，特征处理逻辑在模型中完成，用户通过配置文件进行配置即可
- 导出模型的 serving 接口统一
- 基于 Tensorflow Estimator 接口实现
- 支持多 GPU 训练
- 支持训练时加载 lua 脚本自定义样本 label 和 weight

# 支持的 CTR 模型
- LR
- DNN
- [Wide&Deep](https://arxiv.org/abs/1606.07792)
- [DSSM](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)
- [FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
- [FwFM](https://arxiv.org/abs/1806.03514)
- [AFM](https://arxiv.org/abs/1708.04617)
- [IAFM](https://arxiv.org/abs/1902.09757)
- [KFM](https://arxiv.org/abs/1807.00311)
- [NIFM](https://arxiv.org/abs/1807.00311)
- [NFM](https://arxiv.org/abs/1708.05027)
- [CCPM](https://dl.acm.org/citation.cfm?id=2806603)
- [DeepFM](https://arxiv.org/abs/1703.04247)
- [xDeepFM](https://arxiv.org/abs/1803.05170)
- [DCN](https://arxiv.org/abs/1708.05123)
- [IPNN](https://arxiv.org/abs/1807.00311)
- [KPNN](https://arxiv.org/abs/1807.00311)
- [PIN](https://arxiv.org/abs/1807.00311)
- [FGCNN](https://arxiv.org/abs/1904.04447)
- [FiBiNET](https://arxiv.org/abs/1905.09433)
- [AutoInt](https://arxiv.org/abs/1810.11921)
- [DeepX (model slots 模式)](https://gitlab.vmic.xyz/11118261/easyctr/tree/master/easyctr)

# 待实现的模型
- [FFM](https://arxiv.org/abs/1701.04099)
- [NFFM](https://arxiv.org/abs/1904.12579)
- [BST](https://arxiv.org/abs/1905.06874)
- [DIEN](https://arxiv.org/abs/1809.03672)
- [DSIN](https://arxiv.org/abs/1905.06482)
- [DeepMCP](https://arxiv.org/abs/1906.04365)
- [multi-tower](https://github.com/alibaba/EasyRec/blob/master/docs/source/models/multi_tower.md)

# 用法

## 输入数据格式
输入数据包含 3 种:
 1. 特征描述文件;
 2. 训练样本;
 3. 特征配置文件;


### 特征描述文件
TSV 格式（tab分隔, 需要包含 header），描述特征信息。

格式为：
```
特征id	特征名	特征数据类型	特征类别
```

*特征数据*有如下几类：
 - int
 - float
 - string
 - string_list
 - weighted_string_list

*特征类别*有如下几类：
 - user
 - item
 - ctx
 - extra
 - target

Note: 特征ID小于100的是保留 ID, 用于 label 和 weight 的定义。

示例：
```
id	name	data_type	field_type
101	u_omgid	string	user
102	u_devtype	string	user
103	u_sex	string	user
104	u_age	int	user
105	u_city	string	user
230	i_kd_rowkey	string	item
234	i_kd_cid1	string	item
235	i_kd_cid2	string	item
236	i_kd_cid3	string	item
```

完整版可参考[这里](https://gitlab.vmic.xyz/11118261/easyctr/blob/master/data/data.meta)


### 训练样本

格式(tab 分隔)：
```
特征ID|特征值 特征ID|特征值 ...
```

示例：
```
1|0	2|1.0	335|6045dd66693957ah,0735de71818377bk,1775de1ee54475aw,0375e033f88380bk,1645dce0956051bv,7035dfafb24649aw	337|7625d1f3a7a736bk
```

其中：特征ID 1 表示 label, 2 表示对应的权重。

完整版参考[这里](https://gitlab.vmic.xyz/11118261/easyctr/blob/master/data/train.txt).

## 配置文件
配置文件使用 python 脚本生成，使用时将 [tools/conf_generator/conf_generator.py](https://gitlab.vmic.xyz/11118261/easyctr/blob/master/tools/conf_generator/conf_generator.py) 文件拷贝到工作目录. 参考 [data/conf.py](https://gitlab.vmic.xyz/11118261/easyctr/blob/master/data/conf.py) 文件.

### 特征选择
先创建特征选择类：

```
assembler = Assembler()
```

通过 `Assembler` 的 `add_*` 方法添加特征, 共有 7 种特征类型:
 1. int
 2. float
 3. string
 4. string_list 变长，逗号分隔
 5. float_list  变长，逗号分隔
 6. weighted_string_list  变长，逗号分隔
 7. weighted_video_play_list 变长，逗号分隔，定制化格式，用户视频播放列表

#### 1. int
```
def add_int(self, iname, default_value, force_add=False, alias=None)

# force_add: 与 transform 解耦，为 True 时，即使 transform 没用到，也会添加到 json 配置中
# alias: 特征别名，transform 中使用别名，为 None 则默认为 iname，当需要对同一个
# 特征做两种不同的处理时，可以用这个功能，例如对 ctr 特征不同的归一化方式
```

#### 2. float
```
def add_float(self, iname, default_value, force_add=False, alias=None)
```

#### 3. string
```
def add_string(self, iname, dict_file='', min_count=1, oov_buckets=None,
               top_k=None, default_key=None, force_add=False, alias=None):

# min_count 表示字符串的最小出现次数，小于这个阈值的丢弃或者 hash (当 oov_buckets > 0 时)
# oov_buckets > 0 时，小于 min_count 的字符串 hash 进大小为 oov_buckets 的桶内;
# top_k 为最大词典数, 默认不限制
# default_key 为缺失情况的默认值，类型是 string, 有一个特殊的 key '_extra_', 表示缺失值分配一个特殊 ID
```

#### 4. string_list
例如 tag_list 这种变长特征，逗号分隔
```
def add_string_list(self, iname, dict_file='', min_count=1, width=None,
                    top_k=None, oov_buckets=None, default_key=None,
                    scan_from='head', reverse=False,
                    force_add=False, alias=None):

# width 表示长度, 不足则用 -1, top_k 为最大词典数, 默认不限制
# default_key 为缺失情况的默认值，类型是 string, 有一个特殊的 key '\_extra\_', 表示缺失值分配一个特殊 ID
# scan_from: 'head' 或 'tail', 'head' 表示从前往后扫描，'tail' 表示从后往前添加元素
# reverse: 是否翻转，例如: 1,2,3,4,5,6, width 为 3, 若 scan_from='head', reverse=True,
# 则输出为 3,2,1; 若 scan_from='tail', reverse=False, 则输出为 6,5,4, reverse=True, 输出为 4,5,6
```

#### 5. float_list 类型特征
```
def add_float_list(self, iname, default_value, width, scan_from='head',
                   reverse=False, force_add=False, alias=None)

# width 表示长度, 不足则用 default_value
```


#### 6. weighted_string_list 类型特征
带权重的 string list 特征，例如画像特征，tag 会带有权重
```
def add_weighted_string_list(self, iname, dict_file='', min_count=1,
                             width=None, top_k=None, min_weight=0.0,
                             oov_buckets=None, default_key=None,
                             scan_from='head', reverse=False,
                             force_add=False, alias=None):

# width 表示长度，不足默认 -1，top_k 表示最大词典数，min_weight 为最小权重值
# min_weight 表示最小权重，小于该值的key将会被过滤掉
# default_key 为缺失情况的默认值，类型是 string, 有一个特殊的 key '\_extra\_', 表示缺失值分配一个特殊 ID
```


#### 7. weighted_video_play_list
用户视频播放历史列表，格式为 VideoID:playtime:videotime:algoid:playcnt:timestamp
返回正负播放历史和权重
```
def add_weighted_video_play_list(self, iname, filter_type, params=None,
                                 dict_file='', min_count=1,
                                 width_pos=None, width_neg=None, top_k=None,
                                 oov_buckets=None, default_key=None,
                                 scan_from='tail', reverse=False,
                                 force_add=False, alias=None)
# filter_type: 用户定义的过滤函数
# params: 过滤函数对应的参数，类型为 Python dict
# width_pos: 正浏览历史长度
# width_neg: 负浏览历史长度
```

### 特征转换
先创建特征转换类：

```
transform = Transform()
```

共有 10 种特征转换方式：
 1. numeric: 连续特征处理，支持 log 转换, 归一化;
 2. bucketized: 连续特征分桶;
 3. embedding: 离散特征 embedding;
 4. shared_embedding: 多个特征共享 embedding
 5. pretrained_embedding: 加载预训练 embedding;
 6. indicator: one-hot or multi-hot;
 7. categorical_identity: int 型特征离散处理;
 8. cross: 特征交叉;
 9. weighted_categorical: 带权重的离散特征处理;
 10. attention: attention 结构

通过对应的 `Transform.add_*` 方法添加特征转换.

#### 1. numeric
TODO - 支持更多归一化方法

```
def add_numeric(self, iname, places=[],
                normalizer_fn=None, substract=0, denominator=1.0, exp=1.0,
                oname=None)

# normalizer_fn 可以为 'log-norm', 'norm', 表示归一化方式, substract, denominator 表示归一化参数, exp 为指数项;
# normalizer_fn 为 'log-norm' 时, 转换公式为 ((log(x+1.0) - substract) / denominator) ^ exp;
# normalizer_fn 为 'norm' 时, 转换公式为 ((x - substract) / denominator) ^ exp;
# 输入是原始 int/float 类型的特征名;
# 输出是 numeric_column;
# 可以放在 wide, deep, 不可以作为 cross_column 输入;
# oname 默认为 iname + '.numeric'
```

#### 2. bucketized

```
def add_bucketized(self, iname, places, boundaries, oname=None)

# boundaries 为数值数组, 代表分桶边界, Buckets include the left boundary, and
# exclude the right boundary. Namely, boundaries=[0., 1., 2.] generates buckets
# (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).
# 输入可以是原始 int/float 类型的特征名, 也可以是 numeric_column
# 返回的是 bucketized_column
# 可以放在 wide, deep, 也可以作为 cross_column 的输入

# oname 默认为 iname + '.bucketized'
```

#### 3. embedding

```
def add_embedding(self, iname, places, dimension, combiner='mean', oname=None)

# dimension 为 embedding 维度
# combiner: 'mean', 'sqrtn', 'sum'
# 可以放在 deep
# 输入可以是 categorical_*_column, 也可以是原始 string/string_list 类型, string_list类型时使用 'mean' combiner
# oname 默认为 iname + 'embedding'
```

#### 4. shared_embedding
```
def add_shared_embedding(self, inames, places, dimension, combiner='mean', oname=None)

# 多种特征共享 embedding;
# inames 为数组
# oname 默认为 '#'.join(inames) + '.embedding'
# places 为 list, 其元素个数与 inames 一致，元素类型也为 list，表示每个特征的位置
# combiner: 'mean', 'sqrtn', 'sum'
transform.add_shared_embedding(['i_kd_cid1.identity', 'i_kd_cid2.identity'],
                               [['dssm1', 'deep'], ['dssm2', 'deep']], 20)
```

#### 5. pretrained_embedding

```
def add_pretrained_embedding(self, iname, places, dimension, trainable,
                             pretrained_embedding_file, combiner='mean', oname=None)

# trainable 是否训练 embedding
# pretrained_embedding_file 预训练 embedding 文件, 格式为：
# 第一行包含头部: total, dimension
# 之后每行: item_id <num> <num> ...
# oname 默认为 iname + '.pretrained_embedding'
# combiner: 'mean', 'sqrtn', 'sum'
```

#### 6. indicator
```
def add_indicator(self, iname, places, oname=None)

# 可以直接用于 string_, string__list 类型的特征, int/float 类型的特征需要先用 identity 转换一下
transform.add_indicator('user.city, 'user.city.indicator', ['wide'])

# user.gender 是 int 类型的特征, 用 add_categorical_indentity 转换一下
transform.add_categorical_identity('user.gender', 'user.gender.identity', [], 3)
transform.add_indicator('user.gender.identity', 'user.gender.indicator', ['wide', 'deep'])

# oname 默认为 iname + '.indicator'
```

#### 7. categorical_identity
```
def add_categorical_identity(self, iname, places, num_buckets=-1, oname=None)

# num_buckets 代表离散化的类别数, string/string_list 可以用 -1, 代表使用词典大小, int/float 类型则必须指定大小
# 输入是原始 int/float/string/string_list 的特征
# oname 默认 iname + '.identity'
```

#### 8. cross
```
def add_cross(self, inames, hash_bucket_size, oname=None)

# 此时 inames 为特征名数组, 代表参与交叉的特征, hash_bucket_size 为 hash 空间大小
# oname 默认为 '#'.join(inames) + '.cross'
```


#### 9. weighted_categorical
```
def add_weighted_categorical(self, inames, places, oname=None)

# inames 包含两个元素，第 1 个为 categorical column 特征名, 第 2 个为权重列，为原始字符串
# oname 默认为 inames[0] + '.weighted'
```


# 编译
运行环境：
- python 2.7.x
- tensorflow 1.14.0

因为需要编译[自定义 op](https://www.tensorflow.org/guide/extend/op), 所以需要[源码编译 tensorflow](https://www.tensorflow.org/install/install_sources)。

serving 时需要用到自定义 op，所以需要重新编译 tesorflow serving，将自定义 op 编译进最后的二进制中。

# Demo
编译 tensorflow op:
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

执行 `oh_my_go.sh` 脚本：
    $ ./oh_my_go.sh


# 感谢
- https://github.com/shenweichen/DeepCTR
