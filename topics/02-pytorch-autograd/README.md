# 02 PyTorch 和自动求导

这一节从“手写反向传播”过渡到“框架自动求导”。

上一节你已经理解了：

```text
前向传播 -> 计算损失 -> 反向传播算梯度 -> 梯度下降改参数
```

这一节要看清楚 PyTorch 这类框架帮你做了什么。

## 先看目的和流程

目的：

```text
理解深度学习框架不是魔法，它主要是自动记录计算图、自动算梯度、自动更新参数。
```

流程：

```text
1. 用张量或 Value 保存数字
   每个数字除了 data，还能保存 grad。

2. 前向计算
   比如 prediction = w * x + b。

3. 自动记录计算图
   每一步运算都记录“我是从谁算来的”。

4. 调用 backward()
   从 loss 开始，沿计算图倒着把梯度传回参数。

5. 调用 optimizer.step()
   用 参数 = 参数 - 学习率 * 梯度 更新参数。

6. 下一轮先 zero_grad()
   清掉旧梯度，避免上一轮梯度累加进来。
```

一句话：

```text
前向传播负责生成答案和计算图；backward 负责自动算梯度；optimizer 负责改参数。
```

## 先打开前端讲解

有前端讲解，只是入口之前写得太像文件清单了，不够像“学习路线”。

建议按这个顺序看：

1. [scenario.html](scenario.html)
   预设一个项目需求：根据订单商品件数预测打包耗时。先把 Tensor、target、prediction、loss、grad、backward、optimizer 这些名词放进真实业务里理解。

2. [slides.html](slides.html)
   PPT 式讲解，先用图把“PyTorch 到底替我们做了什么”讲顺。页面里的虚线术语可以悬停查看人话解释。

3. [demo-web/index.html](demo-web/index.html)
   互动演示，拖动权重、偏置和学习率，看 loss、梯度、下一步参数怎么变；公式里的英文专有名词也能悬停解释。

4. [logic-map.html](logic-map.html)
   代码逻辑图，把“前向计算、记录计算图、反向传播、参数更新”对应到 `mini_autograd.py` 的关键行号，并给术语做即时解释。

这些都是静态 HTML，直接在浏览器打开就行，不需要启动服务。

## 这一节要学会什么

- 张量（tensor，白话：能被框架批量计算的多维数字表）。
- 自动求导（autograd，白话：框架沿计算图自动帮你算梯度）。
- 计算图（computation graph，白话：记录每个结果是从哪些数算出来的关系网）。
- 梯度字段（grad，白话：某个参数对 loss 的责任值）。
- 清梯度（zero_grad，白话：下一轮训练前把旧责任清空）。
- 优化器（optimizer，白话：根据梯度真正改参数的工具）。

## 技术栈

| 技术 | 为什么适合这一节 |
| --- | --- |
| Python 标准库 | 先做一个可运行的迷你自动求导，避免被安装环境卡住。 |
| PyTorch | 真实深度学习框架，后面 Transformer、LLM 训练都会用到类似流程。 |
| HTML + CSS + JavaScript | 做交互图，拖动参数看 loss 和梯度怎么变。 |

## 文件入口

- [slides.html](slides.html)：PPT 式图文讲义。
- [scenario.html](scenario.html)：项目需求场景，用订单打包耗时解释术语。
- [demo-web/index.html](demo-web/index.html)：互动网页，看计算图、梯度和一步更新。
- [logic-map.html](logic-map.html)：从框架流程跳到代码行号。
- [demo-project/mini_autograd.py](demo-project/mini_autograd.py)：纯 Python 自动求导 demo。
- [demo-project/pytorch_xor_optional.py](demo-project/pytorch_xor_optional.py)：可选 PyTorch 版本。

## 运行 demo

在 `d:\VScodeProject\llm-learning-lab` 下运行：

```powershell
python .\topics\02-pytorch-autograd\demo-project\mini_autograd.py
```

你应该能看到：

```text
loss 逐渐下降
w 接近 2
b 接近 1
```

## 官方参考

- PyTorch 自动求导教程：https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- PyTorch 优化器教程：https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch 安装页面：https://pytorch.org/get-started/locally/

## 可以试着改

- 把 `learning_rate=0.05` 改大：看是否会震荡。
- 把训练轮数 `100` 改小：看 w、b 是否还没学到位。
- 把 `samples` 改成别的线性规律：比如 `y = 3x - 2`。
