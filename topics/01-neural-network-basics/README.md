# 01 神经网络入门

这一节用一个很小的网络学习“异或”规律：两个输入不一样时输出 1，一样时输出 0。

这听起来像小玩具，但它刚好能说明神经网络最核心的动作：权重怎么影响预测，损失怎么衡量错误，梯度怎么把错误变成修改方向。

## 先看目的和流程

这一节的目的不是先背术语，而是看清楚一件事：

```text
模型怎样从“预测错了”变成“下次少错一点”。
```

完整训练流程是：

```text
1. 随机初始化参数
   先给权重和偏置一个起点，相当于模型随机站到 loss 地形上的某个位置。

2. 前向传播
   用当前参数把输入算成预测结果。这里是“答题”，不是学习。

3. 计算损失
   把预测结果和参考答案比较，得到错得多不多。

4. 反向传播
   从 loss 往回算，找出每个参数对错误的责任，也就是梯度。

5. 梯度下降
   按 “参数 = 参数 - 学习率 * 梯度” 修改参数，让 loss 往更低处走。

6. 重复很多轮
   每轮都重新答题、算错、算梯度、改参数，直到结果足够好。
```

所以这几个词的分工是：

```text
前向传播：用当前参数算答案
损失函数：给这次答案打分
反向传播：算每个参数该承担多少责任
梯度下降：真的修改参数
```

## 这一节要学会什么

- 神经元：像一个小计算器，把输入乘权重、加偏置，再过激活函数。
- 权重：模型可以学习的“重视程度”。
- 偏置：给神经元一个基础倾向。
- 激活函数（activation function，白话：把一串数字压成更适合传下去的信号）。
- 前向传播（forward pass，白话：从输入一路算到答案）。
- 损失函数（loss function，白话：把错得多不多变成一个数字）。
- 反向传播（backpropagation，白话：把错误从后往前分摊给每个权重）。

## 技术栈

| 技术 | 为什么适合这一节 |
| --- | --- |
| Python 标准库 | 不需要安装框架，能直接看到每一步数学和代码。 |
| HTML + CSS + JavaScript | 浏览器直接打开，适合做滑块、图形和即时反馈。 |

这一节暂时不使用 PyTorch。原因是 PyTorch 会把自动求导包装得很好，但入门阶段我们想先看清楚“错了以后权重到底怎么改”。

## 文件入口

- [slides.html](slides.html)：PPT 式图文讲义。
- [quiz.html](quiz.html)：神经网络算法互动练习题，含出题原则和来源参考。
- [demo-web/index.html](demo-web/index.html)：互动网页，拖动输入和权重看输出变化。
- [demo-web/directional-derivative.html](demo-web/directional-derivative.html)：方向导数可视化，看 `v`、梯度、负梯度和 loss 变化。
- [code-flow.html](code-flow.html)：代码业务流程图，区分训练阶段和预测阶段的前向传播。
- [logic-map.html](logic-map.html)：从概念流程跳到代码行号。
- [demo-project/mlp_from_scratch.py](demo-project/mlp_from_scratch.py)：从零训练小神经网络。

## 运行 demo 项目

在 `d:\VScodeProject` 下运行：

```powershell
python .\llm-learning-lab\topics\01-neural-network-basics\demo-project\mlp_from_scratch.py
```

你应该能看到损失逐渐下降，最后四条异或样本都判断正确。

## 可以试着改

- 把 `epochs=8000` 改小：看模型没学够时会怎样。
- 把 `learning_rate=0.7` 改成 `0.05`：看学习太慢时会怎样。
- 把 `hidden_size = 4` 改成 `2`：看模型容量不稳时会怎样。
- 把随机种子 `seed=7` 改成别的数字：看初始权重会不会影响学习。
