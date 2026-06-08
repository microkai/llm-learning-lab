"""
迷你自动求导：用纯 Python 模拟深度学习框架的核心训练流程。

运行方式：
    python mini_autograd.py

这一节的目的：
上一节我们手写了反向传播公式；这一节看框架怎么把“算梯度”这件事自动化。

总流程：
    前向计算 -> 生成计算图 -> loss.backward() 自动算梯度 -> optimizer.step() 更新参数
"""

from __future__ import annotations

from typing import Callable


class Value:
    """一个带梯度的小数字，像极简版 PyTorch Tensor。"""

    def __init__(
        self,
        data: float,
        children: tuple["Value", ...] = (),
        operation: str = "",
        label: str = "",
    ) -> None:
        # data 的目的：保存这个节点的数值，比如权重、预测值、loss。
        self.data = data

        # grad 的目的：保存最终 loss 对这个节点的梯度，也就是这个节点对错误的责任。
        self.grad = 0.0

        # _prev 的目的：记录这个节点是从哪些前置节点算出来的，用来形成计算图。
        # API 解释：set(...) 会创建集合，集合适合保存不重复的对象。
        self._prev = set(children)

        # _operation 的目的：记录这个节点由什么运算产生，方便打印和理解。
        self._operation = operation

        # label 的目的：给节点起名字，方便看懂计算图，比如 w、b、loss。
        self.label = label

        # _backward 的目的：保存“这个节点如何把梯度传回前置节点”的小函数。
        # Callable[[], None] 表示这是一个不用参数、也不返回值的函数。
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        # API 解释：__repr__ 是 Python 打印对象时调用的方法。
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, label={self.label!r})"

    def __add__(self, other: float | "Value") -> "Value":
        # API 解释：__add__ 让 Value 支持 a + b 这种语法。
        # ensure_value 的目的：如果右边是普通数字，就包装成 Value，方便统一处理。
        other = ensure_value(other)

        # out 的目的：保存加法结果节点，同时把 self 和 other 记录成它的前置节点。
        out = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            # 加法的梯度规则：
            # out = self + other
            # loss 对 self 的影响 = loss 对 out 的影响 * 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        # API 解释：__radd__ 支持 2 + Value 这种左边是普通数字的写法。
        return self + other

    def __neg__(self) -> "Value":
        # 目的：支持 -x。这里复用乘法，因为 -x 等价于 x * -1。
        return self * -1.0

    def __sub__(self, other: float | "Value") -> "Value":
        # API 解释：__sub__ 支持 a - b。这里复用加法和取负。
        return self + (-ensure_value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        # API 解释：__rsub__ 支持 2 - Value。
        return ensure_value(other) + (-self)

    def __mul__(self, other: float | "Value") -> "Value":
        # API 解释：__mul__ 让 Value 支持 a * b 这种语法。
        other = ensure_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            # 乘法的梯度规则：
            # out = self * other
            # self 的梯度要乘以 other 的当前数值。
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        # API 解释：__rmul__ 支持 2 * Value。
        return self * other

    def __pow__(self, exponent: float) -> "Value":
        # API 解释：__pow__ 让 Value 支持 x ** 2 这种幂运算。
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def backward() -> None:
            # 幂函数求导：
            # d(x^n)/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1.0)) * out.grad

        out._backward = backward
        return out

    def backward(self) -> None:
        """从当前节点开始，把梯度沿计算图反向传回所有前置节点。"""

        # topo 的目的：保存按依赖关系排好的节点列表。
        topo: list[Value] = []

        # visited 的目的：避免同一个节点被重复加入 topo。
        visited: set[Value] = set()

        def build_topology(node: Value) -> None:
            # 目的：先递归访问前置节点，再加入当前节点。
            # 这样 backward 时倒序遍历，就能从 loss 一路传回参数。
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build_topology(child)
            topo.append(node)

        build_topology(self)

        # 当前节点通常是 loss；loss 对自己的梯度是 1。
        self.grad = 1.0

        # API 解释：reversed(topo) 会倒序遍历列表。
        for node in reversed(topo):
            node._backward()


def ensure_value(item: float | Value) -> Value:
    """把普通数字包装成 Value，方便所有运算都能构建计算图。"""

    if isinstance(item, Value):
        return item
    return Value(float(item))


class LinearModel:
    """一个最小线性模型：prediction = w * x + b。"""

    def __init__(self) -> None:
        # w 的目的：输入 x 前面的系数，训练会自动调整它。
        self.w = Value(-0.4, label="w")

        # b 的目的：常数项，也叫偏置，训练会自动调整它。
        self.b = Value(0.2, label="b")

    def __call__(self, x: float) -> Value:
        # API 解释：__call__ 让对象能像函数一样使用，比如 model(2.0)。
        # 这里就是模型的前向计算。
        return self.w * x + self.b

    def parameters(self) -> list[Value]:
        """返回模型里所有需要训练的参数。"""

        return [self.w, self.b]


class SGD:
    """随机梯度下降优化器：负责按梯度更新参数。"""

    def __init__(self, parameters: list[Value], learning_rate: float) -> None:
        # parameters 的目的：告诉优化器哪些 Value 是可训练参数。
        self.parameters = parameters

        # learning_rate 的目的：控制每次沿负梯度方向走多大一步。
        self.learning_rate = learning_rate

    def zero_grad(self) -> None:
        """把旧梯度清零，避免上一轮梯度累加到下一轮。"""

        for parameter in self.parameters:
            parameter.grad = 0.0

    def step(self) -> None:
        """执行一次参数更新。"""

        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad


def mean_squared_error(model: LinearModel, samples: list[tuple[float, float]]) -> Value:
    """计算一批样本的平均平方误差。"""

    # total_loss 的目的：累加所有样本的 loss。
    total_loss = Value(0.0, label="total_loss")

    for x_value, target_value in samples:
        # prediction 的目的：模型当前对 x_value 的预测。
        prediction = model(x_value)

        # error 的目的：预测值和参考答案之间的差距。
        error = prediction - target_value

        # sample_loss 的目的：把差距变成非负惩罚。
        sample_loss = error**2
        total_loss = total_loss + sample_loss

    # API 解释：len(samples) 返回样本数量。
    return total_loss * (1.0 / len(samples))


def main() -> None:
    """程序入口：训练一个线性模型，让它学会 y = 2x + 1。"""

    # samples 的目的：训练集。每一项是 (输入 x, 参考答案 y)。
    samples = [
        (-2.0, -3.0),
        (-1.0, -1.0),
        (0.0, 1.0),
        (1.0, 3.0),
        (2.0, 5.0),
    ]

    model = LinearModel()
    optimizer = SGD(model.parameters(), learning_rate=0.05)

    for epoch in range(1, 101):
        # zero_grad 的目的：清掉上一轮的梯度。
        optimizer.zero_grad()

        # 前向传播：用当前 w 和 b 计算整批样本的 loss。
        loss = mean_squared_error(model, samples)

        # 反向传播：自动沿计算图算出 w.grad 和 b.grad。
        loss.backward()

        # 梯度下降：用 w.grad 和 b.grad 更新 w.data 和 b.data。
        optimizer.step()

        if epoch in {1, 2, 5, 10, 20, 50, 100}:
            print(
                f"第 {epoch:>3} 轮 | "
                f"loss={loss.data:.6f} | "
                f"w={model.w.data:.4f}, b={model.b.data:.4f} | "
                f"w.grad={model.w.grad:.4f}, b.grad={model.b.grad:.4f}"
            )

    print("\n训练后预测：")
    for x_value, target_value in samples:
        prediction = model(x_value)
        print(f"x={x_value:>4.1f} | 预测={prediction.data:>7.3f} | 参考答案={target_value:>5.1f}")


if __name__ == "__main__":
    main()
