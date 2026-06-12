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

## 不同数据类型的 Dataset 注释版

Dataset 的核心职责是：

```text
原始数据的一条记录
-> 清洗 / 转换 / 编码
-> feature tensor + target tensor
```

DataLoader 会在 Dataset 外面再加一个 batch 维度。

### 表格数据

适合订单、库存、价格、用户行为这类结构化字段。

```python
class OrderDataset(Dataset):
    def __init__(self, rows):
        # rows 是已经整理好的一行一条样本的数据。
        # 常见来源：list[dict]、pandas DataFrame 转成的 records、CSV 读取结果。
        self.rows = rows

    def __len__(self):
        # 告诉 DataLoader 这个数据集一共有多少条样本。
        # DataLoader 会用它判断能取多少次 index。
        return len(self.rows)

    def __getitem__(self, index):
        # index 是 DataLoader 要取的样本编号。
        # 这里拿出第 index 行业务数据。
        row = self.rows[index]

        # features 是模型能看到的输入字段。
        # dtype=torch.float32 表示这些是浮点数，适合参与神经网络计算。
        features = torch.tensor([
            row["item_count"],       # 商品件数
            row["stock"],            # 当前库存
            row["warehouse_load"],   # 仓库负载
        ], dtype=torch.float32)

        # target 是参考答案。
        # 二分类里可以是 0/1；回归里可以是具体数值。
        target = torch.tensor([row["is_delay"]], dtype=torch.float32)

        # 返回一条样本。
        # DataLoader 会把很多条 features 拼成 [batch_size, 3]。
        # targets 会拼成 [batch_size, 1]。
        return features, target
```

这段代码的意义：

```text
把一行业务字段翻译成一个数字向量。
模型不认识 dict，也不认识字段名，模型只吃 tensor。
```

### 图片数据

适合包裹破损识别、商品图片分类、质检图片判断。

```python
class ImageDataset(Dataset):
    def __init__(self, rows, transform):
        # rows 里通常保存图片路径和标签。
        # transform 负责 resize、转 tensor、标准化等图片预处理。
        self.rows = rows
        self.transform = transform

    def __len__(self):
        # 返回图片样本总数。
        return len(self.rows)

    def __getitem__(self, index):
        # 取出当前图片样本的元信息。
        row = self.rows[index]

        # Image.open 读取图片文件。
        # convert("RGB") 保证图片统一是 3 个颜色通道。
        image = Image.open(row["image_path"]).convert("RGB")

        # transform 把 PIL 图片变成 PyTorch tensor。
        # 常见输出 shape 是 [3, H, W]。
        image_tensor = self.transform(image)

        # 分类任务的 target 通常是类别编号。
        # dtype=torch.long 是 CrossEntropyLoss 常用的标签类型。
        target = torch.tensor(row["label"], dtype=torch.long)

        # DataLoader 后，image_tensor 会变成 [batch_size, 3, H, W]。
        return image_tensor, target
```

这段代码的意义：

```text
把图片文件路径翻译成 RGB 像素张量。
图片进入模型前，必须统一尺寸、通道和数值范围。
```

### 文本数据

适合评论分类、客服意图识别、订单备注风险判断。

```python
class TextDataset(Dataset):
    def __init__(self, rows, tokenizer):
        # rows 保存文本和标签。
        # tokenizer 负责把字符串转成 token id。
        # PyTorch 本身不会自动理解字符串。
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        # 返回文本样本总数。
        return len(self.rows)

    def __getitem__(self, index):
        # 取出当前文本样本。
        row = self.rows[index]

        # tokenizer 把原始文本转成模型能吃的编号。
        # padding="max_length" 表示补齐到固定长度。
        # truncation=True 表示太长就截断。
        # return_tensors="pt" 表示返回 PyTorch tensor。
        encoded = self.tokenizer(
            row["text"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # tokenizer 返回时通常带 batch 维度：[1, seq_len]。
        # squeeze(0) 去掉这个临时维度，变成 [seq_len]。
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # 文本分类任务的 target 通常是类别编号。
        target = torch.tensor(row["label"], dtype=torch.long)

        # 返回 token id、mask 和标签。
        # DataLoader 后 shape 会变成：
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        return input_ids, attention_mask, target
```

这段代码的意义：

```text
把自然语言字符串翻译成 token id。
模型后面会用 embedding 把 token id 再变成向量。
```

### 音频数据

适合语音分类、异常声音检测、客服录音情绪识别。

```python
class AudioDataset(Dataset):
    def __init__(self, rows):
        # rows 保存音频文件路径和标签。
        self.rows = rows

    def __len__(self):
        # 返回音频样本总数。
        return len(self.rows)

    def __getitem__(self, index):
        # 取出当前音频样本。
        row = self.rows[index]

        # torchaudio.load 读取音频文件。
        # waveform 是波形张量，常见 shape 是 [channels, time]。
        # sample_rate 是采样率，比如 16000，表示每秒多少个采样点。
        waveform, sample_rate = torchaudio.load(row["audio_path"])

        # 分类标签。
        target = torch.tensor(row["label"], dtype=torch.long)

        # DataLoader 后 waveform 会多一个 batch 维度。
        # 如果音频长度不一致，还需要额外做裁剪、补齐或自定义 collate_fn。
        return waveform, target
```

这段代码的意义：

```text
把音频文件翻译成波形张量。
真实项目里经常还会把 waveform 转成 spectrogram 频谱图。
```

### 视频数据

适合动作识别、质检视频判断、监控事件检测。

```python
class VideoDataset(Dataset):
    def __init__(self, rows, read_frames):
        # rows 保存视频路径和标签。
        # read_frames 是你自己准备的函数，负责抽帧、resize、转 tensor。
        self.rows = rows
        self.read_frames = read_frames

    def __len__(self):
        # 返回视频样本总数。
        return len(self.rows)

    def __getitem__(self, index):
        # 取出当前视频样本。
        row = self.rows[index]

        # read_frames 把视频变成多帧图片张量。
        # 常见 shape 是 [T, 3, H, W]。
        # T 是帧数，3 是 RGB 通道。
        frames = self.read_frames(row["video_path"])

        # 分类标签。
        target = torch.tensor(row["label"], dtype=torch.long)

        # DataLoader 后 shape 通常变成 [batch_size, T, 3, H, W]。
        return frames, target
```

这段代码的意义：

```text
把视频文件翻译成带时间维度的图片序列。
视频 = 多张图片按时间排列。
```

### 视频 + 音频

适合直播内容判断、视频质检、带声音的行为识别。

```python
class VideoAudioDataset(Dataset):
    def __init__(self, rows, read_frames):
        # rows 保存视频路径、音频路径和标签。
        # read_frames 负责视频抽帧。
        self.rows = rows
        self.read_frames = read_frames

    def __len__(self):
        # 返回多模态样本总数。
        return len(self.rows)

    def __getitem__(self, index):
        # 取出当前样本。
        row = self.rows[index]

        # 视频分支输入：多帧图片张量。
        frames = self.read_frames(row["video_path"])

        # 音频分支输入：波形张量。
        waveform, sample_rate = torchaudio.load(row["audio_path"])

        # 共同 target：比如视频类别、风险标签、是否违规。
        target = torch.tensor(row["label"], dtype=torch.long)

        # 多模态数据常用 dict 返回。
        # forward 里可以按 key 分别取 frames 和 audio。
        return {
            "frames": frames,
            "audio": waveform,
            "target": target,
        }
```

这段代码的意义：

```text
一个样本可以返回多个 tensor。
模型里通常会有视频分支和音频分支，分别提取特征后再融合。
```

统一记法：

```text
表格 -> 数字向量 tensor
图片 -> [3, H, W]
文本 -> [seq_len] token id
音频 -> [channels, time]
视频 -> [T, 3, H, W]
视频 + 音频 -> dict，里面放多个 tensor
```

## loss 是可设计的标量目标

一句话结论：

```text
loss 是把模型在当前 batch 上的表现压缩成一个可优化的标量目标；PyTorch 固定负责沿这个标量目标反向算梯度，但这个目标怎么定义，是人可以设计的。
```

参数不是不参与运算。训练每一轮都会用参数做 forward、算 grad、再更新参数。

更准确的关系是：

```text
参数 -> forward -> prediction
prediction + target -> loss 标量
loss.backward() -> 每个参数的 grad
optimizer.step() -> 更新参数
```

为什么通常要合成一个标量 loss？

```text
因为 backward 需要知道：我要最小化哪个总目标。
```

如果有很多条样本、很多个任务、很多个惩罚项，最后通常要合成一个：

```python
total_loss = ...
total_loss.backward()
```

常见设计方式：

```python
# 普通回归：所有错误一视同仁。
total_loss = mse_loss

# 样本加权：某些样本更重要。
total_loss = (loss_each * sample_weights).mean()

# 多任务：同时优化多个目标，但权重不同。
total_loss = cls_loss + 0.5 * regression_loss

# 加正则：既要预测准，也不希望参数太夸张。
total_loss = mse_loss + 0.01 * weight_penalty
```

人话理解：

```text
loss 不是随便算一个数字。
loss 是你告诉模型“什么叫好、什么错更严重、什么目标更重要”的方式。
```

容易误解的地方：

```text
不是 loss 替代了参数运算。
参数是被调整的对象，loss 是调整方向的评价标准，grad 是 loss 对每个参数的变化率。
```

## 今日主线复盘：从数据到优化器

这一段整理今天顺着 Dataset 往后补齐的主线。

### Dataset 是接口，不是自动清洗器

一句话结论：

```text
Dataset 只规定“怎么取一条样本”，具体清洗、merge、标准化、编码、防泄漏，都要人设计和编排。
```

推荐分工：

```text
Dataset 外面：
多表 merge、复杂聚合、train/validation/test 切分、统计 mean/std、构建类别 vocab、保存预处理配置。

Dataset 里面：
按 index 取一条样本，做轻量转换，返回 feature tensor + target tensor。
```

常见数据类型的 Dataset 注释版和 Dataset 阶段检查清单，单独放在：

```text
dataset-notes.md
```

容易误解的地方：

```text
不是“把数据丢给 Dataset，它就会自动处理”。
Dataset 是你写处理逻辑的入口，不是自动特征工程工具。
```

### DataLoader 是批量传送带

一句话结论：

```text
Dataset 管单条样本，DataLoader 管怎么把样本一批一批交给训练 loop。
```

DataLoader 大概做这些事：

```text
1. 调用 len(dataset)，知道样本总数。
2. 生成 index 列表。
3. shuffle=True 时，每个 epoch 打乱 index。
4. 每次取 batch_size 个 index。
5. 多次调用 dataset[index] 拿单条样本。
6. 用 collate_fn 把多条样本拼成 batch。
7. 把 batch 交给训练 loop。
```

shape 变化：

```text
单条 features: [3]
batch_size=32 后: [32, 3]

单条图片: [3, H, W]
batch_size=32 后: [32, 3, H, W]
```

一个 epoch 里的 step 次数：

```text
训练集 3200 条
batch_size = 32
一个 epoch 大约有 100 个 batch
也就是 optimizer.step() 大约执行 100 次
```

容易误解的地方：

```text
DataLoader 不算梯度，不更新参数。
它只是按规则取样本、打乱、组 batch。
```

### train 和 validation 是两种角色

一句话结论：

```text
训练集负责教模型，验证集负责考模型。
```

训练阶段：

```python
model.train()

for features, targets in train_loader:
    optimizer.zero_grad()
    predictions = model(features)
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
```

验证阶段：

```python
model.eval()

with torch.no_grad():
    for features, targets in val_loader:
        predictions = model(features)
        loss = loss_fn(predictions, targets)
```

区别：

```text
训练：
forward -> loss -> backward -> step
会更新参数。

验证：
forward -> loss / metric
不 backward，不 step，不更新参数。
```

`model.train()` 和 `model.eval()` 的作用：

```text
model.train()
告诉模型现在是训练，Dropout、BatchNorm 等层使用训练行为。

model.eval()
告诉模型现在是验证或预测，输出要稳定，Dropout 关闭，BatchNorm 使用稳定统计。
```

`torch.no_grad()` 的作用：

```text
告诉 PyTorch 这段不需要计算图和梯度。
验证/预测只看结果，不需要 backward。
```

### 为什么验证阶段不 backward

一句话结论：

```text
验证集可以指导训练策略，但不应该直接训练参数。
```

关键区分：

```text
loss.backward()
只算梯度，把验证集梯度写进 parameter.grad。
它本身不更新参数。

optimizer.step()
才是真正更新参数。
```

所以验证时如果只 `backward()` 但不 `step()`：

```text
主要是浪费计算和内存，还可能污染 .grad。
```

如果验证时 `backward()` 后又 `step()`：

```text
验证集就变成训练集，评估不再客观。
```

正确逻辑：

```text
训练集：
用 backward + step 修参数。

验证集：
只 forward，记录 loss / metric / 错误案例。

人或外层搜索：
根据验证结果调整 feature、模型结构、loss、学习率、数据采样、早停等策略。
```

如果外面套超参数搜索，也是：

```text
每个候选配置用训练集训练参数。
用验证集打分。
搜索器根据验证分数选下一组配置。
验证集不直接 backward 更新模型参数。
```

### mini-batch 的参数更新逻辑

一句话结论：

```text
一批样本不会各自生成一份新参数；它们先合成一个 batch loss，再得到一份综合梯度，最后更新一次参数。
```

默认常见逻辑：

```text
batch 里每条样本各自有 loss
-> 求平均 loss
-> batch_loss.backward()
-> 得到平均梯度
-> optimizer.step()
-> 更新一次参数
```

等价理解：

```text
grad = (grad1 + grad2 + grad3 + ... + gradN) / N
参数 = 参数 - 学习率 * grad
```

loss 合并规则是活的：

```text
mean：平均，最常用。
sum：求和，受 batch_size 影响更明显。
none：保留每条样本 loss，自己加权或组合。
样本权重：重要样本罚重一点。
类别权重：少数类错了罚重一点。
多任务 loss：多个目标按权重合成 total_loss。
鲁棒 loss：降低异常值影响。
```

固定骨架：

```text
最终要合成一个可反向传播的标量 total_loss。
PyTorch 沿 total_loss 算梯度。
```

### 保存最佳模型，而不是最后模型

一句话结论：

```text
真实训练常保存验证集表现最好的一轮，而不是默认保存最后一轮。
```

因为：

```text
训练集 loss 可能继续下降。
验证集 loss 可能开始变差。
这说明模型可能开始背训练集。
```

常见逻辑：

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")
```

`state_dict` 是模型参数字典：

```text
layer1.weight
layer1.bias
output_layer.weight
output_layer.bias
```

预测时：

```python
model = OrderDelayModel()
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    prediction = model(new_features)
```

容易误解的地方：

```text
state_dict 保存的是参数，不是整个模型设计。
加载时要先创建同样结构的 model。
```

### SGD、Adam、AdamW 都是固定规则

一句话结论：

```text
优化器不是理解任务的智能体，而是按固定数学规则读取 parameter.grad，并更新每个 parameter。
```

SGD：

```text
参数 = 参数 - 学习率 * 梯度
```

SGD + momentum：

```text
在当前梯度之外，保留一部分历史方向惯性。
连续方向一致时走得更稳，当前梯度乱跳时不至于完全被带偏。
```

Adam：

```text
给每个参数维护自己的历史梯度统计。
```

Adam 内部会为每个参数维护：

```text
m：一阶动量，近似历史平均方向。
v：二阶动量，近似历史波动大小。
step：更新到了第几步。
```

所以 Adam 可以做到：

```text
每个参数根据自己的梯度历史，自动调整更新幅度。
```

AdamW：

```text
Adam + 更合理的 weight_decay。
```

`weight_decay` 的人话：

```text
别让参数长得太夸张。
```

代码：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
)
```

容易误解的地方：

```text
Adam/AdamW 不是自己理解业务。
它只是根据每个参数的历史梯度，用固定公式动态调整更新策略。
```

### scheduler 是学习率策略，不是参数更新器

一句话结论：

```text
optimizer 负责根据 grad 更新参数，scheduler 负责调整 optimizer 当前用多大的 learning rate。
```

固定骨架：

```text
loss.backward()
-> 参数上有 grad
-> optimizer.step() 用当前 lr 更新参数
-> scheduler.step() 调整后续 lr
```

scheduler 不做这些事：

```text
不算梯度
不直接改参数
不替代 optimizer
```

它只改：

```text
optimizer.param_groups 里的 lr
```

常见现成策略：

```text
StepLR：
每隔几轮把学习率乘一个比例。

ReduceLROnPlateau：
验证集长期不变好，就降低学习率。

CosineAnnealingLR：
学习率像余弦曲线一样平滑下降。

warmup：
训练初期先用小学习率，逐渐升高，再进入正常下降。
```

学习率策略是活的，可以自己写：

```python
def get_lr(epoch, val_loss=None):
    if epoch < 5:
        # warmup：前几轮慢慢升高。
        return 0.0002 * (epoch + 1)

    if val_loss is not None and val_loss > 0.5:
        # 如果验证集表现不好，就保守一点。
        return 0.0005

    # 后期细调。
    return 0.0001


for epoch in range(num_epochs):
    train_one_epoch()
    val_loss = validate()

    lr = get_lr(epoch, val_loss)

    for group in optimizer.param_groups:
        group["lr"] = lr
```

也可以用 PyTorch 的 `LambdaLR` 接入自定义规则：

```python
def lr_lambda(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    return 0.95 ** (epoch - 5)


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda,
)
```

这里 `lr_lambda` 返回的是比例：

```text
当前 lr = 初始 lr * lr_lambda(epoch)
```

更进一步，optimizer 可以有多个参数组：

```text
embedding 层用小学习率
新加的分类头用大学习率
```

人话理解：

```text
optimizer 负责走路。
scheduler 负责调步子大小。

训练前期可以步子大一点，快速靠近。
训练后期步子小一点，避免震荡，细细修。
```

容易误解的地方：

```text
scheduler 不是只能用 PyTorch 内置的。
只要你能写出规则，就能自己控制每个阶段的 lr。
```

### 训练护栏：防过拟合、稳数值、防梯度爆炸

一句话结论：

```text
Dropout、weight_decay、BatchNorm、LayerNorm、gradient clipping 不改变训练主线，它们是在主线周围加护栏。
```

整体位置：

```text
Dataset / DataLoader
-> forward
-> loss
-> backward
-> gradient clipping（可选，step 前）
-> optimizer.step
-> scheduler.step（可选，调学习率）
```

Dropout、BatchNorm、LayerNorm 通常写在模型结构里：

```text
__init__ 里定义层
forward 里调用层
```

#### Dropout

人话：

```text
训练时随机关掉一部分中间信号，逼模型别太依赖某几个特征或隐藏信号。
```

代码：

```python
self.dropout = nn.Dropout(p=0.2)

def forward(self, x):
    hidden = self.layer1(x)
    hidden = torch.relu(hidden)
    hidden = self.dropout(hidden)
    output = self.output_layer(hidden)
    return output
```

关键点：

```text
p=0.2 表示训练时随机丢掉 20% 的中间信号。
model.train() 时 Dropout 生效。
model.eval() 时 Dropout 自动关闭。
```

解决的问题：

```text
防止模型过度依赖某几个信号，降低过拟合风险。
```

#### weight_decay

人话：

```text
别让参数长得太夸张。
```

代码：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
)
```

它的位置：

```text
写在 optimizer 里。
optimizer.step() 更新参数时，会按规则加入权重衰减。
```

解决的问题：

```text
让模型别为了训练集某些细节把参数放得过大。
让模型更平滑，降低过拟合风险。
```

#### BatchNorm

批归一化（BatchNorm，白话：按一个 batch 的统计量稳定中间数值）。

代码：

```python
self.layer1 = nn.Linear(8, 32)
self.bn1 = nn.BatchNorm1d(32)

def forward(self, x):
    hidden = self.layer1(x)
    hidden = self.bn1(hidden)
    hidden = torch.relu(hidden)
    return hidden
```

关键点：

```text
model.train() 时使用当前 batch 的均值和方差。
model.eval() 时使用训练过程中累计的稳定统计。
```

解决的问题：

```text
中间数值尺度更稳定，训练更顺。
常见于 CNN 或传统深层网络。
```

#### LayerNorm

层归一化（LayerNorm，白话：按单条样本内部的多个特征稳定数值）。

代码：

```python
self.norm = nn.LayerNorm(32)

def forward(self, x):
    hidden = self.layer1(x)
    hidden = self.norm(hidden)
    hidden = torch.relu(hidden)
    return hidden
```

关键点：

```text
BatchNorm 看一个 batch 的统计。
LayerNorm 看单条样本内部的统计。
```

解决的问题：

```text
减少对 batch size 的依赖。
Transformer / LLM 里非常常见。
```

#### gradient clipping

梯度裁剪（gradient clipping，白话：梯度太大时先压住，再更新参数）。

位置：

```text
loss.backward() 之后
optimizer.step() 之前
```

代码：

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

optimizer.step()
```

解决的问题：

```text
防止某一轮梯度突然巨大，导致参数一步飞掉。
常见于 RNN、Transformer、大模型训练。
```

#### 放在一起的模型例子

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 32)
        self.norm = nn.LayerNorm(32)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        hidden = self.layer1(x)
        hidden = self.norm(hidden)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        return self.output_layer(hidden)
```

#### 总分工

```text
Dropout：
防止模型太依赖某些中间信号。

weight_decay：
限制参数别长太大。

BatchNorm：
按 batch 稳定中间数值。

LayerNorm：
按单条样本稳定中间数值。

gradient clipping：
防止梯度突然爆炸。
```

容易误解的地方：

```text
这些不是新的训练主线。
主线仍然是 forward -> loss -> backward -> step。
它们只是让这条主线更稳、更不容易学歪。
```

### device：模型和 tensor 必须在同一个计算设备上

一句话结论：

```text
device 是模型和 tensor 所在的计算位置；参与同一次计算的模型参数、输入 tensor、target tensor 必须在同一个 device 上。
```

常见设备：

```text
CPU：
普通处理器，PyTorch 默认设备。

GPU / CUDA：
显卡，适合大量矩阵计算。
```

常见写法：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Model().to(device)
```

训练 loop 里：

```python
for features, targets in train_loader:
    features = features.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    predictions = model(features)
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
```

验证 loop 里也一样：

```python
model.eval()

with torch.no_grad():
    for features, targets in val_loader:
        features = features.to(device)
        targets = targets.to(device)

        predictions = model(features)
        loss = loss_fn(predictions, targets)
```

如果 batch 是 dict：

```python
batch_on_device = {}

for key, value in batch.items():
    if torch.is_tensor(value):
        batch_on_device[key] = value.to(device)
    else:
        batch_on_device[key] = value
```

人话：

```text
tensor 可以搬到 GPU，字符串、文件路径、普通元信息不能 `.to(device)`。
```

保存和加载时也要注意 device：

```python
state_dict = torch.load(
    "best_model.pt",
    map_location="cpu",
)

model.load_state_dict(state_dict)
```

`map_location="cpu"` 的作用：

```text
如果模型原来在 GPU 上保存，但当前机器没有 GPU，也能加载到 CPU。
```

容易误解的地方：

```text
torch.cuda.is_available() 只是检查当前 PyTorch 环境有没有可用 CUDA GPU。
如果装的是 CPU 版 PyTorch，它会是 False。
这不影响小 demo，只是不能用 GPU 加速。
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
