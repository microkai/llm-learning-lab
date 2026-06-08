"""
可选 PyTorch 版本：用框架训练一个小模型。

运行前先安装 PyTorch。官方安装命令以 PyTorch 官网为准：
    https://pytorch.org/get-started/locally/

运行方式：
    python pytorch_xor_optional.py
"""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境没有安装 PyTorch。先去 https://pytorch.org/get-started/locally/ 选择适合本机的安装命令。"
    ) from exc


def main() -> None:
    """用 PyTorch 学习异或，让你对照上一节的手写版本。"""

    # torch.manual_seed 的目的：固定 PyTorch 随机数，方便每次运行结果接近。
    torch.manual_seed(7)

    # API 解释：torch.tensor 会创建张量。
    # 张量（tensor，白话：能被框架批量计算的多维数字表）。
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # torch.nn.Sequential 的目的：把多层网络按顺序串起来。
    # torch.nn.Linear 的目的：创建线性层，内部自带权重和偏置。
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Sigmoid(),
        torch.nn.Linear(4, 1),
        torch.nn.Sigmoid(),
    )

    # BCELoss 是二元交叉熵损失，适合输出 0 到 1 的二分类概率。
    loss_fn = torch.nn.BCELoss()

    # torch.optim.SGD 是随机梯度下降优化器，负责按梯度更新参数。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.7)

    for epoch in range(1, 8001):
        # 前向传播：model(x) 用当前参数算预测。
        prediction = model(x)

        # 计算损失：把预测和参考答案比较。
        loss = loss_fn(prediction, y)

        # zero_grad 的目的：清掉上一轮留下的梯度。
        optimizer.zero_grad()

        # backward 的目的：自动反向传播，计算所有参数的梯度。
        loss.backward()

        # step 的目的：执行梯度下降，真正更新参数。
        optimizer.step()

        if epoch in {1, 10, 100, 1000, 8000}:
            print(f"第 {epoch:>4} 轮，loss={loss.item():.6f}")

    print("\n训练后的判断：")
    with torch.no_grad():
        scores = model(x)
        answers = (scores >= 0.5).float()
        for input_value, score, answer in zip(x, scores, answers):
            print(f"输入 {input_value.tolist()} -> 分数 {score.item():.4f} -> 判断 {int(answer.item())}")


if __name__ == "__main__":
    main()
