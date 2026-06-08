# Demo：自动求导和 PyTorch 思维

## 先看目的和流程

目的：

```text
看清楚深度学习框架到底帮我们省掉了什么。
```

上一节我们手写：

```text
前向传播 -> 计算损失 -> 手算梯度 -> 手动更新权重
```

这一节换成框架思维：

```text
1. 用 Value 或 Tensor 表示数字
2. 前向计算时自动记录计算图
3. 调用 backward() 自动算梯度
4. 调用 optimizer.step() 更新参数
5. 下一轮先 zero_grad() 清掉旧梯度
```

一句话：

```text
框架不是替你决定怎么学，而是替你可靠地记计算图、算梯度、更新参数。
```

## 文件

- `mini_autograd.py`：纯 Python 可运行版本，模拟自动求导。
- `pytorch_xor_optional.py`：可选 PyTorch 版本，需要先安装 PyTorch。
- `requirements.txt`：说明依赖。

## 技术栈

| 技术 | 为什么适合 |
| --- | --- |
| Python 标准库 | 不依赖安装，先把自动求导的骨架看懂。 |
| PyTorch | 真实深度学习框架，后续 Transformer 和 LLM 都会用到类似思想。 |

## 运行纯 Python 版本

```powershell
python .\topics\02-pytorch-autograd\demo-project\mini_autograd.py
```

它会训练一个线性模型，学习：

```text
y = 2x + 1
```

你应该看到 `loss` 下降，`w` 接近 `2`，`b` 接近 `1`。

## 运行 PyTorch 版本

当前电脑已经安装 CPU 版 PyTorch，可以直接运行下面的脚本。后续如果要换 GPU/CUDA 版本，安装命令请以官方页面为准：

```text
https://pytorch.org/get-started/locally/
```

安装后运行：

```powershell
python .\topics\02-pytorch-autograd\demo-project\pytorch_xor_optional.py
```

## 关键行号

| 代码位置 | 学什么 |
| --- | --- |
| `mini_autograd.py:19` | `Value` 如何保存数值、梯度和计算图。 |
| `mini_autograd.py:61` | 加法节点如何把梯度传回前置节点。 |
| `mini_autograd.py:92` | 乘法节点如何根据另一边的数值传梯度。 |
| `mini_autograd.py:118` | `backward()` 如何倒序走计算图。 |
| `mini_autograd.py:165` | 线性模型 `w*x+b` 的前向计算。 |
| `mini_autograd.py:186` | `zero_grad()` 为什么要清梯度。 |
| `mini_autograd.py:192` | `step()` 如何按梯度更新参数。 |
| `mini_autograd.py:235` | 训练循环如何串起 forward、backward、step。 |

## 可以试着改

- `learning_rate=0.05`：改大或改小，看 loss 是否稳定下降。
- `range(1, 101)`：改训练轮数，看 w、b 是否接近目标。
- 初始 `self.w`、`self.b`：改起点，看训练是否仍能收敛。
