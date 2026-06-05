# 从零训练异或小网络

这个 demo 用纯 Python 标准库训练一个 2-4-1 神经网络，让它学会异或规律。

## 先看目的和流程

目的：

```text
给模型 4 条异或样本，让它自己调整权重，最后能判断两个输入是否不同。
```

训练流程：

```text
1. 创建网络
   TinyNetwork() 随机生成权重和偏置。

2. 输入样本
   比如 [0.0, 1.0]，正确答案是 1.0。

3. 前向传播
   forward() 用当前参数算出 prediction。

4. 计算损失
   error = prediction - target
   loss = 0.5 * error * error

5. 反向传播
   算 output_delta 和 hidden_delta，得到每层参数的梯度方向。

6. 梯度下降
   用 learning_rate * gradient 修改权重和偏置。

7. 重复训练
   train() 把 4 条样本反复训练 8000 轮。
```

一句话记忆：

```text
前向传播负责答题，反向传播负责算责任，梯度下降负责改参数。
```

## 为什么用纯 Python

| 技术 | 适合原因 |
| --- | --- |
| Python 标准库 | 不需要安装依赖，能把每一步训练逻辑摊开看。 |
| 手写权重更新 | 能直接理解反向传播和学习率，不被框架细节挡住。 |

等你理解这版以后，再上 PyTorch 会更舒服，因为你知道框架自动帮你省掉了什么。

## 运行

```powershell
python mlp_from_scratch.py
```

如果你在项目根目录 `d:\VScodeProject`，可以运行：

```powershell
python .\llm-learning-lab\topics\01-neural-network-basics\demo-project\mlp_from_scratch.py
```

## 关键行号

| 代码位置 | 学什么 |
| --- | --- |
| `mlp_from_scratch.py:28` | 一条训练样本长什么样。 |
| `mlp_from_scratch.py:40` | 激活函数如何把数字压到 0 到 1。 |
| `mlp_from_scratch.py:55` | 点积如何完成加权求和。 |
| `mlp_from_scratch.py:73` | 初始化权重和偏置。 |
| `mlp_from_scratch.py:132` | 前向传播如何算预测。 |
| `mlp_from_scratch.py:175` | 单条样本如何训练。 |
| `mlp_from_scratch.py:190` | 损失如何表示错误大小。 |
| `mlp_from_scratch.py:198` | 输出层梯度如何计算。 |
| `mlp_from_scratch.py:219` | 隐藏层如何接收传回来的错误。 |
| `mlp_from_scratch.py:245` | 多轮训练如何让损失下降。 |

## 语法和 API 说明

- `from __future__ import annotations`：Python 兼容开关，让类型注解先按文字保存，减少解析类型时的麻烦。
- `math.exp(x)`：来自 Python 标准库 `math`，计算 `e` 的 `x` 次方，是 `sigmoid` 公式的一部分。
- `random.Random(seed)`：来自 Python 标准库 `random`，创建一个独立随机数生成器；种子一样，随机序列也一样。
- `rng.uniform(-1.0, 1.0)`：`uniform` 是“均匀随机”的意思。这里会在 `-1.0` 到 `1.0` 之间随机抽一个小数，作为权重或偏置的初始值。
- `NamedTuple`：来自 `typing`，能快速创建“带字段名的数据容器”，这里用来表示一条训练样本。
- `zip(a, b)`：Python 内置函数，把两个列表按位置配对，适合同时拿一个神经元的权重和偏置。
- `sum(...)`：Python 内置函数，把一串数字加起来，这里用来完成点积求和。
- `range(...)`：Python 内置函数，生成循环用的整数序列。
- `list.append(x)`：列表方法，把 `x` 追加到列表末尾。
- `list.copy()`：列表方法，复制一份新列表，避免旧权重被后续更新影响。
- `enumerate(...)`：Python 内置函数，同时给出序号和值，适合按位置更新权重。
- `float(x)`：Python 内置函数，把值转成小数，这里用来明确预测值是单个小数。
- `len(x)`：Python 内置函数，返回长度，这里用来算平均损失。
- `print(...)`：Python 内置函数，把结果输出到命令行。
- `if __name__ == "__main__"`：Python 文件入口判断；直接运行文件时才开始训练，被别的文件导入时不会自动跑。
- 为什么要随机初始化：如果所有权重一开始都一样，多个神经元容易学成同一种东西；随机起点能让它们分头尝试不同方向。

## 可以试着改

- `hidden_size = 4`：隐藏神经元数量。
- `epochs=8000`：训练轮数。
- `learning_rate=0.7`：每次改权重的步子大小。
- `seed=7`：初始权重的随机种子。
