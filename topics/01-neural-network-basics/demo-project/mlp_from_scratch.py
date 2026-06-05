"""
迷你神经网络：从零学习“异或”规律。

运行方式：
    python mlp_from_scratch.py

这个文件故意不使用深度学习框架，目的是把学习过程拆开给你看：
输入 -> 加权求和 -> 激活 -> 预测 -> 算损失 -> 反向改权重。
"""

# __future__.annotations 是 Python 的兼容开关。
# 目的：让类型注解先按“文字”保存，减少老版本 Python 在解析类型时的麻烦。
from __future__ import annotations

# math 是 Python 标准库里的数学工具箱；这里用 math.exp 计算指数函数。
import math

# random 是 Python 标准库里的随机数工具箱；这里用它给权重和偏置随机起点。
import random

# NamedTuple 是 typing 提供的“带字段名的元组”工具。
# 目的：让 TrainingSample 既像简单数据容器，又能标注 inputs 和 target 的类型。
from typing import NamedTuple


# API 解释：NamedTuple 会自动帮这个类生成初始化能力。
# 所以 TrainingSample([0.0, 1.0], 1.0) 可以直接创建一条样本。
class TrainingSample(NamedTuple):
    """一条训练样本：inputs 是输入，target 是希望模型学到的答案。"""

    # inputs 的目的：存一条样本的两个输入值，比如 [0.0, 1.0]。
    # 类型注解解释：list[float] 表示“这个变量应该是一个列表，列表里装小数”。
    inputs: list[float]

    # target 的目的：存这条样本的正确答案，训练时用它衡量预测错了多少。
    # 类型注解解释：float 表示小数类型，比如 0.0、1.0、0.37。
    target: float


def sigmoid(value: float) -> float:
    """激活函数：把任意数字压到 0 到 1 之间，像一个“信号开关”。"""

    # 函数签名解释：value: float 表示传入的小数；-> float 表示函数会返回一个小数。
    # API 解释：math.exp(x) 会计算 e 的 x 次方。
    # 这里的 math.exp(-value) 是 sigmoid 公式的一部分，用来把输入压到 0 到 1。
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_derivative(activated_value: float) -> float:
    """激活函数的斜率：告诉模型“这里还能改动多少”。"""

    return activated_value * (1.0 - activated_value)


def dot(left: list[float], right: list[float]) -> float:
    """点积：成对相乘再相加，是神经元做加权求和的核心语法。"""

    # API 解释：zip(left, right) 会把两个列表按位置配对。
    # 例子：zip([1, 2], [3, 4]) 会依次给出 (1, 3) 和 (2, 4)。
    # API 解释：sum(...) 会把括号里产生的一串数字加起来。
    # 这里用了生成器表达式，意思是“边算边交给 sum 加起来”。
    # 等价展开：
    # total = 0.0
    # for a, b in zip(left, right):
    #     total += a * b
    # return total
    return sum(a * b for a, b in zip(left, right))


class TinyNetwork:
    """一个 2-4-1 小网络：2 个输入、4 个隐藏神经元、1 个输出。"""

    def __init__(self, seed: int = 7) -> None:
        # API 解释：__init__ 是 Python 类的初始化方法。
        # 创建 TinyNetwork() 时，它会自动运行，用来准备初始权重和偏置。

        # random.Random(seed) 的目的：固定随机数，方便每次运行得到相近结果。
        # API 解释：random.Random(seed) 会创建一个随机数生成器。
        # seed 是“随机种子”；种子一样，后面抽到的随机数顺序也一样。
        # rng 的含义：一个独立的随机数生成器；rng.uniform(-1.0, 1.0) 会在 -1 到 1 之间均匀随机抽一个小数，用来初始化权重和偏置。
        rng = random.Random(seed)

        # hidden_size 的含义：隐藏层里有几个神经元；越多，模型能表达的形状越复杂。
        hidden_size = 4

        # input_size 的含义：每条样本有几个输入；异或问题固定是 2 个输入。
        input_size = 2

        # hidden_weights[第几个隐藏神经元][第几个输入]。
        # hidden_weights 的目的：保存“输入层 -> 隐藏层”的所有连接权重。
        # 每个权重都是一个可学习参数，先用 rng.uniform(-1.0, 1.0) 随机给起点，训练再不断改它。
        # API 解释：range(input_size) 会生成 0 到 input_size-1 的整数序列。
        # 这里不用具体序号，所以变量名写成 _，表示“这个值我不关心，只是循环这么多次”。
        # 语法解释：列表推导式是在批量生成二维列表。
        # 这段等价展开：
        # self.hidden_weights = []
        # for neuron_index in range(hidden_size):
        #     neuron_weights = []
        #     for input_index in range(input_size):
        #         weight = rng.uniform(-1.0, 1.0)  # uniform 表示“均匀随机”，这里是在 -1 到 1 之间抽小数。
        #         neuron_weights.append(weight)
        #     self.hidden_weights.append(neuron_weights)
        self.hidden_weights = [
            [rng.uniform(-1.0, 1.0) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]

        # hidden_biases 的目的：保存每个隐藏神经元自己的偏置。
        # 偏置像神经元的基础倾向，先用 rng.uniform(-1.0, 1.0) 随机给起点，不依赖输入也会影响输出。
        # API 解释：rng.uniform(a, b) 会在 a 到 b 之间均匀随机抽一个小数。
        # “均匀”表示范围里的各个位置被抽到的机会差不多。
        # 这段等价展开：
        # self.hidden_biases = []
        # for neuron_index in range(hidden_size):
        #     bias = rng.uniform(-1.0, 1.0)  # bias 也先在 -1 到 1 之间均匀随机抽一个初始值。
        #     self.hidden_biases.append(bias)
        self.hidden_biases = [rng.uniform(-1.0, 1.0) for _ in range(hidden_size)]

        # 输出层只有 1 个神经元，所以每个隐藏神经元都连过来。
        # output_weights 的目的：保存“隐藏层 -> 输出层”的连接权重。
        # 它决定每个隐藏神经元的结果对最终答案有多大影响。
        # 这段等价展开：
        # self.output_weights = []
        # for neuron_index in range(hidden_size):
        #     weight = rng.uniform(-1.0, 1.0)  # 输出层权重也先均匀随机初始化。
        #     self.output_weights.append(weight)
        self.output_weights = [rng.uniform(-1.0, 1.0) for _ in range(hidden_size)]

        # output_bias 的目的：保存输出神经元的偏置，先用 uniform 均匀随机给起点，给最终判断一个基础倾向。
        self.output_bias = rng.uniform(-1.0, 1.0)

    def forward(self, inputs: list[float]) -> dict[str, list[float] | float]:
        """前向传播：只负责从输入算出预测值，不修改权重。"""

        # 类型注解解释：list[float] 表示“装着小数的列表”。
        # 类型注解解释：dict[str, list[float] | float] 表示“键是字符串，值可能是小数列表，也可能是单个小数”。
        # 这里的 | 是“或者”的意思。

        # hidden_raw 的目的：保存隐藏神经元激活前的原始分数，方便观察中间过程。
        hidden_raw: list[float] = []

        # hidden_outputs 的目的：保存隐藏神经元激活后的输出，会作为输出层的输入。
        hidden_outputs: list[float] = []

        # API 解释：zip(self.hidden_weights, self.hidden_biases) 会把每个隐藏神经元的权重和偏置配成一对。
        # 这样循环里每次处理一个隐藏神经元。
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            # weights 的含义：当前隐藏神经元连到两个输入上的权重。
            # bias 的含义：当前隐藏神经元自己的偏置。
            # raw_score 的目的：算出当前隐藏神经元还没过激活函数前的分数。
            raw_score = dot(inputs, weights) + bias

            # API 解释：list.append(x) 会把 x 追加到列表末尾。
            # 这里把每个隐藏神经元的原始分数收集起来。
            hidden_raw.append(raw_score)

            # sigmoid(raw_score) 的结果是当前隐藏神经元传给下一层的信号。
            hidden_outputs.append(sigmoid(raw_score))

        # output_raw 的目的：输出神经元激活前的原始分数。
        output_raw = dot(hidden_outputs, self.output_weights) + self.output_bias

        # prediction 的目的：模型最终预测分数；越接近 1，越倾向判断为 1。
        prediction = sigmoid(output_raw)

        # API 解释：return {...} 返回一个字典。
        # 字典像“带名字的结果包”，后面可以用 cache["prediction"] 按名字取值。
        return {
            "hidden_raw": hidden_raw,
            "hidden_outputs": hidden_outputs,
            "output_raw": output_raw,
            "prediction": prediction,
        }

    def train_one(self, sample: TrainingSample, learning_rate: float) -> float:
        """训练一条样本：算错多少，就按梯度方向轻轻改一次权重。"""

        # cache 的目的：保存前向传播的中间结果，后面反向传播要用。
        # API 解释：self.forward(...) 是调用当前对象自己的 forward 方法。
        cache = self.forward(sample.inputs)

        # hidden_outputs 的目的：拿到隐藏层输出，用来计算输出层权重该怎么改。
        hidden_outputs = cache["hidden_outputs"]

        # prediction 的目的：拿到这条样本当前的预测值，和正确答案比较。
        # API 解释：float(x) 会把 x 转成小数。
        # 这里是为了明确告诉 Python 和读代码的人：prediction 是一个单个小数。
        prediction = float(cache["prediction"])

        # 均方误差的一半：0.5 * 差距的平方。
        # 目的：平方会让“大错”更显眼，0.5 只是为了求导后更整洁。
        # error 的含义：预测值 - 正确答案；后面的梯度都从这个差距开始。
        error = prediction - sample.target

        # loss 的目的：把“错得多不多”变成一个非负数字，方便观察训练有没有变好。
        loss = 0.5 * error * error

        # 输出层梯度：预测错多少 * 输出神经元当前还能改多少。
        # output_delta 的目的：表示输出神经元这一次要承担多少错误责任。
        output_delta = error * sigmoid_derivative(prediction)

        # 先保存旧权重，因为隐藏层要根据“旧输出权重”分摊责任。
        # old_output_weights 的目的：避免更新输出层后，隐藏层分摊错误时用到被改过的权重。
        # API 解释：list.copy() 会复制一份新列表。
        # 如果直接 old_output_weights = self.output_weights，只是多了一个名字指向同一份列表。
        old_output_weights = self.output_weights.copy()

        # API 解释：enumerate(hidden_outputs) 会同时给出“序号”和“值”。
        # 例子：第 0 个隐藏输出是多少、第 1 个隐藏输出是多少。
        for index, hidden_value in enumerate(hidden_outputs):
            # enumerate 会同时给出序号和值，适合更新列表中的对应位置。
            # index 的含义：当前正在更新第几个输出层权重。
            # hidden_value 的含义：对应隐藏神经元传给输出层的信号。
            # gradient 的目的：告诉这个输出层权重应该往哪个方向、改多少。
            gradient = output_delta * float(hidden_value)
            self.output_weights[index] -= learning_rate * gradient
        self.output_bias -= learning_rate * output_delta

        # hidden_deltas 的目的：保存每个隐藏神经元分到的错误责任。
        hidden_deltas: list[float] = []

        # API 解释：这里再次用 enumerate，是因为要同时拿隐藏神经元序号和它的输出值。
        for index, hidden_value in enumerate(hidden_outputs):
            # 隐藏层没有直接答案，所以要通过输出层把错误“传回来”。
            # hidden_delta 的目的：表示当前隐藏神经元这一次要承担多少错误责任。
            hidden_delta = (
                output_delta
                * old_output_weights[index]
                * sigmoid_derivative(float(hidden_value))
            )
            hidden_deltas.append(hidden_delta)

        for neuron_index, hidden_delta in enumerate(hidden_deltas):
            for input_index, input_value in enumerate(sample.inputs):
                # neuron_index 的含义：当前正在更新第几个隐藏神经元。
                # input_index 的含义：当前正在更新这个神经元连到第几个输入的权重。
                # input_value 的含义：这条样本在这个输入位置上的值。
                # gradient 的目的：告诉隐藏层权重应该往哪个方向、改多少。
                gradient = hidden_delta * input_value
                self.hidden_weights[neuron_index][input_index] -= learning_rate * gradient
            self.hidden_biases[neuron_index] -= learning_rate * hidden_delta

        return loss

    def train(self, samples: list[TrainingSample], epochs: int, learning_rate: float) -> None:
        """训练很多轮：同一批样本反复看，权重就会慢慢变得更会判断。"""

        # API 解释：range(1, epochs + 1) 会生成从 1 到 epochs 的轮次编号。
        # Python 的 range 右边界不包含自己，所以要写 epochs + 1。
        for epoch in range(1, epochs + 1):
            # total_loss 的目的：累计这一轮所有样本的损失，方便算平均表现。
            total_loss = 0.0
            for sample in samples:
                total_loss += self.train_one(sample, learning_rate)

            # 语法解释：{1, 10, 1000} 是集合，in 用来判断 epoch 是否在集合里。
            # 目的：只在少数关键轮次打印日志，避免 8000 轮都刷屏。
            if epoch in {1, 10, 100, 1000, epochs}:
                # average_loss 的目的：看这一轮平均错得多不多，比单条样本更稳定。
                # API 解释：len(samples) 会返回样本数量，这里是 4。
                average_loss = total_loss / len(samples)

                # API 解释：print(...) 会把内容输出到命令行。
                # f"..." 是格式化字符串，可以把变量嵌进文字里。
                # {epoch:>4} 表示数字靠右，占 4 个字符宽度；{average_loss:.6f} 表示保留 6 位小数。
                print(f"第 {epoch:>4} 轮，平均损失：{average_loss:.6f}")

    def predict(self, inputs: list[float]) -> float:
        """预测：训练完以后，只做前向传播，拿到输出答案。"""

        # API 解释：这里复用 forward 方法，只取返回字典里的 prediction。
        return float(self.forward(inputs)["prediction"])


def main() -> None:
    """程序入口：准备数据、训练网络、打印结果。"""

    # xor_samples 的目的：准备异或问题的全部训练数据。
    # 规律：两个输入一样时答案是 0，不一样时答案是 1。
    xor_samples = [
        TrainingSample([0.0, 0.0], 0.0),
        TrainingSample([0.0, 1.0], 1.0),
        TrainingSample([1.0, 0.0], 1.0),
        TrainingSample([1.0, 1.0], 0.0),
    ]

    # network 的目的：创建一个还没训练的小网络，里面的权重一开始是随机的。
    network = TinyNetwork()

    # API 解释：network.train(...) 是调用 TinyNetwork 里的 train 方法。
    # epochs=8000 和 learning_rate=0.7 是关键字参数，写名字能让含义更清楚。
    network.train(xor_samples, epochs=8000, learning_rate=0.7)

    # API 解释：\n 是换行符，让输出前先空一行。
    print("\n训练后的判断：")
    for sample in xor_samples:
        # score 的目的：保存模型对当前输入给出的 0 到 1 分数。
        # API 解释：network.predict(...) 是调用 TinyNetwork 里的 predict 方法。
        score = network.predict(sample.inputs)

        # answer 的目的：把连续分数转成最终分类结果；0.5 是这里的人为分界线。
        # 语法解释：A if 条件 else B 是条件表达式；条件成立取 A，否则取 B。
        answer = 1 if score >= 0.5 else 0
        print(f"输入 {sample.inputs} -> 分数 {score:.4f} -> 判断 {answer}")


# API 解释：__name__ 是 Python 自动提供的当前文件名字。
# 当你直接运行这个文件时，__name__ 会等于 "__main__"；这时才调用 main()。
# 如果这个文件被别的文件 import，引入时不会自动开始训练。
if __name__ == "__main__":
    main()
