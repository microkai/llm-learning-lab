# Dataset 阶段设计笔记

这份笔记专门记录 Dataset 阶段要做什么、能做什么、哪些规则是 PyTorch 提供的接口，哪些规则必须由人来设计。

## 一句话定位

```text
Dataset 不是自动清洗器，也不是自动特征工程器。
Dataset 是“把一条业务样本翻译成模型输入”的标准接口。
```

PyTorch 只规定了接口：

```python
__init__      # 数据来源和处理规则怎么准备
__len__       # 数据集有多少条样本
__getitem__   # 第 index 条样本怎么返回
```

具体做什么，要人来编排：

```text
样本粒度怎么定
target 怎么定义
多表怎么 merge
字段怎么选
缺失值怎么处理
类别怎么编码
数值怎么标准化
变长数据怎么 padding
怎么避免数据泄漏
```

## Dataset 阶段总流程

推荐把数据工作拆成两层：

```text
Dataset 外面：
做重处理，例如多表 join、聚合、清洗、切分、统计 mean/std、构建类别字典。

Dataset 里面：
按 index 取一条样本，做轻量转换，返回 tensor。
```

完整逻辑：

```text
原始业务数据
-> 定义样本粒度
-> 定义 target
-> 多表 merge / 聚合
-> 切 train / validation / test
-> 只用 train 统计标准化参数和类别字典
-> Dataset 按条返回 feature tensor + target tensor
-> DataLoader 组 batch
-> training loop
```

## 第一步：定义样本粒度

先决定“一条样本”是什么。

延迟发货预测里，通常是：

```text
一条订单 = 一个样本
```

但别的任务可能是：

```text
一张图片 = 一个样本
一句文本 = 一个样本
一段 10 秒音频 = 一个样本
一段 16 帧视频 = 一个样本
一个用户一天的行为序列 = 一个样本
```

这个决定会影响后面所有 shape。

## 第二步：定义 target

target 是模型要学的参考答案。

延迟发货预测可以有不同 target：

```text
二分类：是否延迟
回归：延迟了多少分钟
多分类：不延迟 / 轻微延迟 / 严重延迟
多任务：同时预测是否延迟和预计延迟分钟数
```

target 不同，模型输出和 loss 也不同：

```text
回归 -> 输出 1 个数 -> MSELoss / HuberLoss
二分类 -> 输出 1 个 logit -> BCEWithLogitsLoss
多分类 -> 输出 N 个 logits -> CrossEntropyLoss
```

## 第三步：多表 merge 和聚合

真实业务数据经常不是一张表。

延迟发货预测可能来自：

```text
订单表
商品明细表
库存表
仓库负载表
物流表
发货结果表
```

你通常要先整理成：

```text
一行 = 一个样本
```

例子：

```text
order_id
item_count
sku_count
order_amount
min_stock
warehouse_load_at_order_time
carrier_id
main_category_id
is_presale
target_is_delayed
```

注意：

```text
Dataset 里不建议做重型多表查询。
否则 __getitem__ 每次取样本都会慢，训练会被数据处理拖死。
```

## 第四步：切 train / validation / test

不要把所有数据都拿来训练。

```text
train：用来更新参数。
validation：用来调结构、调超参数、看是否过拟合。
test：最后验收，尽量只用一次。
```

延迟发货这种时间相关业务，推荐按时间切：

```text
过去订单 -> train
较近订单 -> validation
最新订单 -> test
```

这样更接近真实上线：

```text
用过去规律预测未来订单。
```

## 第五步：避免数据泄漏

数据泄漏是 Dataset 阶段最危险的问题之一。

如果要预测“下单时会不会延迟发货”，feature 里不能放下单后才知道的字段：

```text
不能用 shipped_at
不能用最终物流状态
不能用售后结果
不能用实际签收时间
不能用延迟原因标签
```

这些字段会让验证集效果虚高，但上线必崩。

判断标准：

```text
模型预测那一刻，业务上能不能拿到这个字段？
拿不到，就不能放进 feature。
```

## 第六步：缺失值处理

缺失值可以在 Dataset 外面先处理，也可以在 Dataset 里轻量处理。

常见策略：

```text
数值缺失 -> 填 0 / 均值 / 中位数
类别缺失 -> unknown_id
文本缺失 -> 空字符串
图片缺失 -> 跳过样本或用占位图
```

最好额外加一个“是否缺失”的字段：

```text
stock_missing = 1 / 0
```

因为“缺失本身”有时就是一种业务信号。

## 第七步：数值标准化

金额、库存、仓库负载的尺度可能差很多。

常见标准化：

```text
x_scaled = (x - mean) / std
```

关键规则：

```text
mean/std 只能用 train 统计。
validation/test/predict 复用 train 的 mean/std。
```

不要让验证集自己算一套 mean/std，否则评估和真实预测不一致。

示例：

```python
class OrderDataset(Dataset):
    def __init__(self, rows, amount_mean, amount_std):
        self.rows = rows
        self.amount_mean = amount_mean
        self.amount_std = amount_std

    def __getitem__(self, index):
        row = self.rows[index]

        amount = (row["order_amount"] - self.amount_mean) / self.amount_std

        features = torch.tensor([
            row["item_count"],
            amount,
            row["warehouse_load"],
        ], dtype=torch.float32)

        target = torch.tensor([row["is_delay"]], dtype=torch.float32)
        return features, target
```

## 第八步：类别字段编码

类别字段不能直接当连续数字理解。

比如：

```text
warehouse_id = 3
```

不表示“仓库 3 比仓库 1 大两倍”。

常见做法：

```text
类别 -> 连续 id -> Embedding
```

必须处理未知类别：

```text
没见过的新仓库 -> unknown_id
```

示例：

```python
class OrderDataset(Dataset):
    def __init__(self, rows, carrier_vocab, unknown_id=0):
        self.rows = rows
        self.carrier_vocab = carrier_vocab
        self.unknown_id = unknown_id

    def __getitem__(self, index):
        row = self.rows[index]

        carrier_id = self.carrier_vocab.get(
            row["carrier"],
            self.unknown_id,
        )

        numeric_features = torch.tensor([
            row["item_count"],
            row["warehouse_load"],
        ], dtype=torch.float32)

        carrier_tensor = torch.tensor(carrier_id, dtype=torch.long)
        target = torch.tensor([row["is_delay"]], dtype=torch.float32)

        return {
            "numeric": numeric_features,
            "carrier_id": carrier_tensor,
            "target": target,
        }
```

## 第九步：固定长度和变长数据

表格和图片通常比较容易固定 shape：

```text
表格 -> [feature_dim]
图片 -> resize 成 [3, H, W]
```

文本、音频、视频经常长度不一样：

```text
文本句子长短不同
音频时间长短不同
视频帧数不同
```

常见处理：

```text
截断：太长就切掉
补齐：太短就 padding
mask：告诉模型哪些是真实内容，哪些是补出来的
```

## 第十步：collate_fn 负责组装不规则 batch

DataLoader 默认会把多条样本直接 stack。

如果每条样本 shape 一样，默认没问题。

如果每条样本长度不同，默认 stack 会失败，这时要写 `collate_fn`。

文本 padding 例子：

```python
def collate_text_batch(batch):
    # batch 是一个列表，里面每个元素来自 Dataset.__getitem__。
    # 每个元素是 (input_ids, target)。
    input_ids_list, targets = zip(*batch)

    # 找出这一批里最长的文本长度。
    max_len = max(len(input_ids) for input_ids in input_ids_list)

    padded_ids = []
    attention_masks = []

    for input_ids in input_ids_list:
        pad_len = max_len - len(input_ids)

        # 0 假设是 padding token id。
        padded = torch.cat([
            input_ids,
            torch.zeros(pad_len, dtype=torch.long),
        ])

        # mask 里 1 表示真实 token，0 表示补齐 token。
        mask = torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])

        padded_ids.append(padded)
        attention_masks.append(mask)

    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(attention_masks),
        "target": torch.tensor(targets, dtype=torch.long),
    }
```

使用：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_text_batch,
)
```

人话：

```text
Dataset 负责拿一条样本。
collate_fn 负责把一批不等长样本拼成一个可训练 batch。
```

## 常见数据格式和 shape

| 数据类型 | 单条样本常见输入 | batch 后常见 shape | 需要注意 |
| --- | --- | --- | --- |
| 表格 | `[feature_dim]` | `[batch_size, feature_dim]` | 数值标准化、类别编码、缺失值 |
| 图片 | `[3, H, W]` | `[batch_size, 3, H, W]` | resize、normalize、通道顺序 |
| 文本 | `[seq_len]` token id | `[batch_size, seq_len]` | padding、attention_mask、tokenizer |
| 音频 | `[channels, time]` | `[batch_size, channels, time]` | 采样率、裁剪、补齐、频谱 |
| 视频 | `[T, 3, H, W]` | `[batch_size, T, 3, H, W]` | 抽帧、统一帧数、resize |
| 视频+音频 | dict 多路 tensor | dict 多路 batch tensor | 时间对齐、mask、分支融合 |

## 各类 Dataset 代码注释版

下面这些代码重点看：

```text
__init__ 准备数据来源和处理规则。
__len__ 告诉 DataLoader 数据集长度。
__getitem__ 把第 index 条原始样本转成 tensor。
return 的内容就是训练 loop 每次拿到的一条样本。
```

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

## Dataset 里面适合做什么

适合放轻量、稳定、按样本发生的操作：

```text
按 index 取一行
字段取值
轻量缺失值处理
转 tensor
图片读取和 transform
文本 tokenizer
音频文件读取
类别 id 映射
```

## Dataset 外面更适合做什么

更适合提前处理：

```text
多表 merge
复杂聚合
异常值清洗
train/validation/test 切分
统计 mean/std
构建类别 vocab
保存预处理配置
重型特征工程
大文件预处理缓存
```

## 预处理配置要保存

训练时用到的预处理配置，预测时必须复用。

需要保存：

```text
数值字段 mean/std
类别字段 vocab
unknown_id / padding_id
字段顺序
tokenizer 名称或词表
图片 resize 尺寸
音频采样率
视频抽帧规则
```

否则会出现：

```text
训练时一种处理方式
预测时另一种处理方式
模型输入分布对不上
```

## Dataset 阶段检查清单

- [ ] 我定义清楚了一条样本是什么。
- [ ] 我定义清楚 target 是什么。
- [ ] 我确认 feature 在预测时刻真实可用。
- [ ] 我把多表数据整理成“一条样本一行”或等价结构。
- [ ] 我按合理方式切了 train / validation / test。
- [ ] 我只用 train 统计 mean/std、vocab 等预处理参数。
- [ ] 我处理了缺失值和未知类别。
- [ ] 我确认每个字段的 dtype 合理。
- [ ] 我知道单条样本返回的 shape。
- [ ] 我知道 DataLoader 后 batch 的 shape。
- [ ] 如果样本长度不一致，我写了 padding / mask / collate_fn。
- [ ] 我保存了预测时要复用的预处理配置。

## 总结

```text
Dataset 阶段不是“把数据丢给 PyTorch”。
它是把业务数据变成可训练样本的边界层。

PyTorch 给你接口：
Dataset、DataLoader、collate_fn。

人负责设计：
样本粒度、target、字段、清洗、编码、补齐、防泄漏、预处理复用。
```
