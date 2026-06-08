# 02 PyTorch 和自动求导清单

## 概念

- [ ] 我能解释张量为什么像“批量数字表”。
- [ ] 我能解释计算图记录了什么。
- [ ] 我能解释 `backward()` 不是更新参数，而是算梯度。
- [ ] 我能解释 `optimizer.step()` 才是真正更新参数。
- [ ] 我能解释为什么每轮训练前要 `zero_grad()`。
- [ ] 我能用“直线和开关”的类比解释 Linear + ReLU 为什么带来非线性。
- [ ] 我能说出 Dataset、DataLoader、train loop、validation loop 分别在训练流水线里的位置。
- [ ] 我能说出 Dataset 阶段哪些是 PyTorch 接口，哪些需要人设计和编排。
- [ ] 我能解释 padding、mask、collate_fn 为什么用于变长样本。
- [ ] 我能解释为什么 mean/std、类别 vocab 要只从训练集统计并在验证/预测复用。
- [ ] 我能解释 `model.train()`、`model.eval()` 和 `torch.no_grad()` 为什么不属于参数更新公式，但真实项目必须用。
- [ ] 我能说出 PyTorch 帮我们省掉了上一节哪部分手写代码。

## 代码

- [ ] 我能指出 `Value` 里 `data` 和 `grad` 的作用。
- [ ] 我能指出加法和乘法如何把梯度传回去。
- [ ] 我能指出 `backward()` 如何倒序遍历计算图。
- [ ] 我能指出 `SGD.step()` 如何更新参数。
- [ ] 我能指出训练循环里的四个动作：清梯度、前向、反向、更新。

## 实践

- [ ] 我运行过 `mini_autograd.py`。
- [ ] 我改过学习率。
- [ ] 我改过训练轮数。
- [ ] 我打开过 `demo-web/index.html`。
- [ ] 我打开过 `nonlinearity.html`，并切换过 Linear、Linear + ReLU。
- [ ] 我打开过 `training-pipeline.html`，并切换过数据入口、训练循环、验证循环、保存预测。
- [ ] 我打开过 `logic-map.html` 并顺着行号看过代码。
