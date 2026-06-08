# 02 追问沉淀笔记

这份笔记专门放学习过程中临时追问出来的关键点。它不是主线讲义，而是把“刚刚问明白的卡点”沉淀下来，方便复盘。

## PyTorch 到底接管了什么

上一节神经网络学的是训练逻辑：

```text
feature + target
-> 初始化参数
-> forward 算 prediction
-> loss 比较 prediction 和 target
-> backward 算 grad
-> optimizer.step 更新参数
-> zero_grad 清旧 grad
-> 重复训练
```

PyTorch 不改变这套底层逻辑，它把通用、重复、容易写错的部分工程化：

```text
自动创建参数
自动记录计算图
自动求导
把梯度写到 parameter.grad
让 optimizer 读取 grad 并更新参数
```

## Tensor 是什么

张量（Tensor，白话：深度学习框架使用的多维数字容器）可以理解成模型能吃的数字形态。

```text
一个数 -> 0 维张量
一排数 -> 1 维张量
一张表 -> 2 维张量
多张表叠起来 -> 更高维张量
```

能直接量化的业务字段，比如商品件数、库存、金额，可以直接整理成张量。

不能直接量化的东西，要先编码：

```text
文本 -> token id -> embedding 向量
图片 -> 像素矩阵
类别 -> 编号 / one-hot / embedding
```

Transformer 不是“量化工具”，它是数字化之后处理上下文关系的一种模型架构。

## model 和 parameters

在 PyTorch 里，普通概念要放进对应对象里，框架才能识别。

```text
模型 -> nn.Module
可训练参数 -> nn.Parameter
线性层 -> nn.Linear
前向计算 -> forward()
损失函数 -> nn.MSELoss / CrossEntropyLoss 等
优化器 -> torch.optim.SGD / Adam / AdamW
```

`model.parameters()` 的作用是把模型里已经注册的可训练参数交给 optimizer 管理。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
```

意思是：

```text
SGD 管理 model 里的参数。
step() 时读取这些参数的 .grad。
然后按 参数 -= 学习率 * 梯度 更新。
```

## nn.Linear 自动创建 weight 和 bias

字段很多时，不需要手写 `w1`、`w2`、`b1`。

```python
self.layer1 = nn.Linear(8, 16)
```

PyTorch 内部自动创建：

```text
weight shape = [16, 8]
bias shape   = [16]
```

含义是：

```text
8 个输入字段
-> 16 个隐藏信号
```

每个隐藏信号都有：

```text
8 个输入权重
1 个 bias
```

所以 bias 对齐的是输出维度，也就是 `weight[x, y]` 里的 `x`。

第二层：

```python
self.output_layer = nn.Linear(16, 1)
```

自动创建：

```text
weight shape = [1, 16]
bias shape   = [1]
```

含义是：

```text
16 个隐藏信号
-> 1 个预测结果
```

## 输入层在哪里

输入层通常不是一个需要训练的层。它就是喂给模型的 tensor。

```text
features shape = [batch_size, 8]
```

意思是：

```text
一批样本
每条样本 8 个字段
```

`nn.Linear(8, 16)` 里的 `8` 必须和每条样本的字段数量对上。

## hidden_size 和隐藏信号怎么定

有多少个隐藏信号是人先定的。

```python
nn.Linear(8, 16)
```

这里 `16` 就是隐藏信号数量，也可以叫 hidden size。

怎么定没有绝对公式，一般从小到大试：

```text
8 -> 16 -> 32 -> 64
```

看训练集和验证集：

```text
训练集差，验证集也差 -> 欠拟合，可以加容量
训练集好，验证集差 -> 过拟合，需要减小模型或加正则
训练集和验证集都变好 -> 加容量有用
训练集继续变好，验证集不变好 -> 差不多该停
```

隐藏信号的具体含义不是人写死的，而是模型为了降低 loss 学出来的中间组合。

## forward 是模型的计算路线

`__init__` 里定义有哪些模块：

```text
Linear
Embedding
Dropout
LayerNorm
Attention
```

`forward` 里定义输入怎么一步步变成输出：

```text
取数值特征
类别特征做 embedding
拼接特征
过 Linear
过 ReLU
过 Dropout
输出 prediction / logit
```

训练循环通常比较固定，模型内部的设计复杂度往往集中在 `forward`。

但完整项目里，难点还包括：

```text
数据清洗
feature 选择
target 定义
验证集切分
业务指标
错误分析
部署和监控
```

## Linear + ReLU 为什么需要非线性

如果只有线性层：

```text
Linear -> Linear -> Linear
```

整体仍然等价于一个 Linear，也就是一条直线或一个线性平面。

加上 ReLU 后：

```text
z = Linear(x)
a = ReLU(z) = max(0, z)
```

ReLU 像开关：

```text
z <= 0 -> 输出 0，信号关闭
z > 0  -> 输出 z，信号打开
```

业务理解：

```text
Linear 负责算一个综合信号。
ReLU 判断这个信号是否生效。
多个 Linear + ReLU 可以在输入空间里切分区域，让不同区域有不同斜率。
```

所以非线性不是只处理“突然跳变”，而是让模型能用很多分段直线近似曲线。

## ReLU 在哪个维度上开关

ReLU 不是直接在原始业务字段上开关，而是在 Linear 输出的隐藏维度上逐个开关。

```text
features shape = [batch_size, input_features]
Linear 后 z shape = [batch_size, hidden_units]
ReLU 后 a shape = [batch_size, hidden_units]
```

ReLU 不改变 shape，只是把里面小于等于 0 的值变成 0。

## grad 存在哪里

梯度不是显式传给 optimizer 的，而是存在每个参数自己的 `.grad` 属性里。

```text
loss.backward()
-> 沿计算图反向遍历
-> 把梯度写到 parameter.grad

optimizer.step()
-> 遍历 optimizer 管理的参数
-> 读取 parameter.grad
-> 更新 parameter
```

所以不是：

```python
grads = loss.backward()
optimizer.step(grads)
```

而是：

```python
loss.backward()
optimizer.step()
```

PyTorch 用：

```text
parameter -> parameter.grad
```

这组绑定关系，隐藏了显式传参和 mapping。

## 为什么要 zero_grad

`optimizer.step()` 会更新参数，但不会自动清掉 `.grad`。

下一轮训练时，旧梯度通常已经没有用了，所以要先：

```python
optimizer.zero_grad()
```

它的作用是：

```text
清空上一轮每个 parameter.grad
避免旧梯度累加污染本轮更新
```

## 计算图是不是显式参与

计算图通常不需要手动创建。PyTorch 会在 forward 过程中自动记录。

```python
predictions = model(features)
loss = loss_fn(predictions, targets)
```

这两步会让 PyTorch 记录：

```text
prediction 从哪些参数和输入算来
loss 从 prediction 和 target 算来
```

然后：

```python
loss.backward()
```

会沿这张计算图倒着走，把梯度写进相关参数的 `.grad`。

## forward 分支和动态图

如果 `forward` 里有分支：

```python
if use_branch_a:
    return self.branch_a(x)
else:
    return self.branch_b(x)
```

本轮实际走了哪条路，计算图就记录哪条路。

```text
model.parameters()
是 optimizer 可能管理的全部参数名单。

本轮计算图
是这次 forward 实际用到的参数和运算路线。

backward
只沿本轮计算图给相关参数写 grad。
```

普通模型通常先保持固定 forward 路线。分支属于进阶能力，常用于业务规则明确、缺字段处理、专家模型、MoE 等场景。

## 无关字段和噪音

如果某个字段没有用，模型理想情况下会学到它对应的影响接近 0。

但模型不会天然保证自动删掉噪音字段。噪音字段可能在训练集里碰巧有假相关，导致过拟合。

常见处理：

```text
验证集观察
消融实验
Dropout
weight_decay / L1 正则
异常值清洗
特征重要性分析
错误案例复盘
```

## 训练 loop 外面还有什么

核心训练 loop 是：

```python
for batch in train_loader:
    optimizer.zero_grad()
    prediction = model(batch)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()
```

外面通常还有：

```text
Dataset / DataLoader：读数据、清洗、转 tensor、组 batch。
训练脚本：控制 epoch、日志、验证、保存模型、early stopping。
多任务模型：forward 输出多个结果，loop 里组合多个 loss。
分布式训练：多 GPU 同步参数、分发 batch、合并梯度。
```

这些内容已经补成展示页：

```text
training-pipeline.html
```

学习顺序建议：

```text
先看 Dataset / DataLoader 如何提供 batch
再看 train loop 如何更新参数
再看 validation loop 为什么不更新参数
最后看 state_dict 如何保存和加载模型
```

## 这一段的总总结

```text
PyTorch 内层负责学参数：
forward -> loss -> backward -> step

人负责设计外层：
feature、target、模型结构、loss、optimizer、验证方式

如果要自动试结构和训练策略：
就在 PyTorch 训练 loop 外再包一层超参数搜索。
```
