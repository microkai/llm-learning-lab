(function () {
  const TERMS = [
    {
      title: "PyTorch",
      aliases: ["PyTorch"],
      explain: "一个深度学习框架。你写模型和训练流程，它负责高效算数字、自动算梯度、帮参数往更低 loss 的方向改。"
    },
    {
      title: "张量 / Tensor",
      aliases: ["张量", "tensor", "Tensor"],
      explain: "能被框架批量计算的多维数字表。一个数、一串数、矩阵、很多层矩阵，都可以看成张量。"
    },
    {
      title: "Value",
      aliases: ["Value"],
      explain: "这一节纯 Python demo 里的小数字对象。它不只存 data，还存 grad 和自己从哪些节点算来。"
    },
    {
      title: "data",
      aliases: ["data"],
      explain: "当前节点真正的数值。比如 Value(3) 里面的 3，就是 data。"
    },
    {
      title: "grad / 梯度",
      aliases: ["grad", "gradient", "梯度", "梯度字段", "w.grad", "b.grad"],
      explain: "loss 对某个参数的变化率。人话说，就是这个参数对错误有多大责任，以及该往哪个方向改。"
    },
    {
      title: "自动求导 / autograd",
      aliases: ["自动求导", "autograd"],
      explain: "框架自动沿着计算图算梯度。你不用手推每个参数的导数，只要从 loss 调 backward。"
    },
    {
      title: "计算图 / Computation Graph",
      aliases: ["计算图", "computation graph"],
      explain: "记录每个结果是从哪些数字、哪些运算一步步算出来的关系图。反向传播要靠它找回去。"
    },
    {
      title: "前向传播 / forward",
      aliases: ["前向传播", "前向计算", "forward"],
      explain: "用当前参数从输入算到输出。训练时它还会顺手留下计算图，方便后面反向算梯度。"
    },
    {
      title: "反向传播 / backward",
      aliases: ["反向传播", "backward", "backward()", "loss.backward()"],
      explain: "从 loss 开始倒着走计算图，把每个参数该承担的梯度算出来。它不是改参数，只是算责任。"
    },
    {
      title: "优化器 / optimizer",
      aliases: ["优化器", "optimizer"],
      explain: "真正改参数的工具。它读取 grad，然后按学习率执行类似 参数 = 参数 - 学习率 * 梯度。"
    },
    {
      title: "zero_grad",
      aliases: ["zero_grad", "zero_grad()", "optimizer.zero_grad()", "清梯度"],
      explain: "把上一轮留下的梯度清空。因为很多框架默认会累加梯度，不清空就会把旧账混进新一轮。"
    },
    {
      title: "step",
      aliases: ["step", "step()", "optimizer.step", "optimizer.step()"],
      explain: "优化器走一步。也就是根据当前梯度和学习率，真正修改 w、b 这类可学习参数。"
    },
    {
      title: "loss / 损失",
      aliases: ["loss", "损失"],
      explain: "模型这次错得有多严重的数字。训练的目标就是让 loss 尽量变小。"
    },
    {
      title: "prediction / 预测值",
      aliases: ["prediction", "预测", "预测值"],
      explain: "模型根据当前参数算出来的答案。训练时会拿它和参考答案 target 比较。"
    },
    {
      title: "target / 参考答案",
      aliases: ["target", "参考答案", "目标值"],
      explain: "训练数据里给定的正确结果。loss 就是用 prediction 和 target 的差算出来的。"
    },
    {
      title: "参数 / parameter",
      aliases: ["参数", "parameter", "parameters"],
      explain: "模型训练时会被改动的数字。神经网络学习，本质上就是不断调整这些参数。"
    },
    {
      title: "Linear / 线性层",
      aliases: ["Linear", "线性层", "线性"],
      explain: "把输入按权重加起来，再加偏置。人话说，它负责加权汇总信息，但单独使用时只能表达直线式关系。"
    },
    {
      title: "ReLU / 激活函数",
      aliases: ["ReLU", "relu", "激活函数"],
      explain: "常用激活函数：小于 0 的值变成 0，大于 0 的值保留。它像一个开关，让模型能表达非线性规则。"
    },
    {
      title: "非线性",
      aliases: ["非线性", "nonlinear", "non-linearity"],
      explain: "输出不是按固定比例直线变化，而是允许拐弯、分段、触发。真实业务多数都有这种规律。"
    },
    {
      title: "模型 / model",
      aliases: ["模型", "model"],
      explain: "把输入变成输出的一套计算规则。训练前它只是一个会算的结构，训练后参数里才存下经验。"
    },
    {
      title: "特征 / feature",
      aliases: ["特征", "feature", "features"],
      explain: "模型能看到的输入信息。比如预测打包耗时时，订单商品件数就是一个特征。"
    },
    {
      title: "批量 / batch",
      aliases: ["批量", "batch"],
      explain: "一次拿多条样本一起算。框架喜欢批量计算，因为更快，也能让梯度更稳定。"
    },
    {
      title: "回归 / regression",
      aliases: ["回归", "regression"],
      explain: "预测一个连续数值的任务，比如打包耗时、房价、温度。它不是判断类别，而是估一个数。"
    },
    {
      title: "权重 / weight",
      aliases: ["权重", "weight"],
      explain: "输入前面的系数。它决定某个输入对结果的影响有多大。"
    },
    {
      title: "偏置 / bias",
      aliases: ["偏置", "bias"],
      explain: "公式里的常数项。就算输入是 0，偏置也能把输出整体往上或往下挪。"
    },
    {
      title: "学习率 / learning rate / lr",
      aliases: ["学习率", "learning rate", "lr"],
      explain: "每次沿梯度反方向走多远。太大容易跳过头，太小会学得很慢。"
    },
    {
      title: "训练循环",
      aliases: ["训练循环", "训练轮数", "epoch"],
      explain: "把清梯度、前向、算 loss、反向、更新参数重复很多遍。模型就是在重复里慢慢学会的。"
    },
    {
      title: "样本 / sample",
      aliases: ["样本", "sample", "samples", "训练数据"],
      explain: "一条训练参考。通常包含输入和对应答案，比如 x=2、target=5。"
    },
    {
      title: "均方误差 / MSE",
      aliases: ["均方误差", "mean squared error", "MSE", "error²"],
      explain: "常见 loss 算法：先算预测和答案的差，再平方。错得越远，惩罚越大。"
    },
    {
      title: "节点 / node",
      aliases: ["节点", "node", "前置节点"],
      explain: "计算图里的一个点。它可能是原始参数，也可能是某一步运算得到的中间结果。"
    },
    {
      title: "梯度累加",
      aliases: ["梯度累加", "累加"],
      explain: "新的梯度不是覆盖旧梯度，而是加到旧梯度上。多轮训练前通常要 zero_grad 清掉。"
    },
    {
      title: "线性模型",
      aliases: ["线性模型", "w*x+b"],
      explain: "形如 y = w*x + b 的简单模型。w 是斜率，b 是截距，很适合用来入门看训练流程。"
    }
  ];

  const aliasToTerm = new Map();
  const aliases = [];

  for (const term of TERMS) {
    for (const alias of term.aliases) {
      aliasToTerm.set(alias.toLowerCase(), term);
      aliases.push(alias);
    }
  }

  aliases.sort((left, right) => right.length - left.length);

  const pattern = new RegExp(aliases.map(escapeRegExp).join("|"), "gi");
  const skipSelector = [
    "script",
    "style",
    "pre",
    "code",
    "a",
    "button",
    "input",
    "textarea",
    "select",
    ".glossary-term",
    ".no-glossary"
  ].join(",");

  let tooltip = null;
  let activeTerm = null;

  function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function shouldSkip(textNode) {
    const parent = textNode.parentElement;
    return !parent || !textNode.nodeValue.trim() || Boolean(parent.closest(skipSelector));
  }

  function wrapTextNode(textNode) {
    const text = textNode.nodeValue;
    pattern.lastIndex = 0;

    if (!pattern.test(text)) return;
    pattern.lastIndex = 0;

    const fragment = document.createDocumentFragment();
    let lastIndex = 0;
    let match = pattern.exec(text);

    while (match) {
      const matchedText = match[0];
      const term = aliasToTerm.get(matchedText.toLowerCase());

      if (match.index > lastIndex) {
        fragment.append(document.createTextNode(text.slice(lastIndex, match.index)));
      }

      if (term) {
        const span = document.createElement("span");
        span.className = "glossary-term";
        span.tabIndex = 0;
        span.dataset.title = term.title;
        span.dataset.explain = term.explain;
        span.setAttribute("aria-label", `${term.title}：${term.explain}`);
        span.textContent = matchedText;
        fragment.append(span);
      } else {
        fragment.append(document.createTextNode(matchedText));
      }

      lastIndex = match.index + matchedText.length;
      match = pattern.exec(text);
    }

    if (lastIndex < text.length) {
      fragment.append(document.createTextNode(text.slice(lastIndex)));
    }

    textNode.parentNode.replaceChild(fragment, textNode);
  }

  function applyGlossary(root) {
    const start = root || document.body;
    if (!start) return;

    const nodes = [];
    const walker = document.createTreeWalker(start, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        return shouldSkip(node) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      }
    });

    while (walker.nextNode()) {
      nodes.push(walker.currentNode);
    }

    for (const node of nodes) {
      wrapTextNode(node);
    }
  }

  function ensureTooltip() {
    if (tooltip) return tooltip;

    tooltip = document.createElement("div");
    tooltip.className = "glossary-tooltip";
    tooltip.setAttribute("role", "tooltip");
    document.body.append(tooltip);
    return tooltip;
  }

  function positionTooltip(termElement) {
    const bubble = ensureTooltip();
    const rect = termElement.getBoundingClientRect();
    const bubbleRect = bubble.getBoundingClientRect();
    const gap = 10;
    const left = Math.min(
      Math.max(14, rect.left + rect.width / 2 - bubbleRect.width / 2),
      window.innerWidth - bubbleRect.width - 14
    );
    const topCandidate = rect.bottom + gap;
    const top = topCandidate + bubbleRect.height > window.innerHeight - 14
      ? Math.max(14, rect.top - bubbleRect.height - gap)
      : topCandidate;

    bubble.style.left = `${left}px`;
    bubble.style.top = `${top}px`;
  }

  function showTooltip(termElement) {
    activeTerm = termElement;
    const bubble = ensureTooltip();
    bubble.innerHTML = `<strong>${escapeHtml(termElement.dataset.title)}</strong><span>${escapeHtml(termElement.dataset.explain)}</span>`;
    bubble.classList.add("visible");
    positionTooltip(termElement);
  }

  function hideTooltip() {
    activeTerm = null;
    if (tooltip) tooltip.classList.remove("visible");
  }

  function setupEvents() {
    function findTerm(event) {
      return event.target instanceof Element
        ? event.target.closest(".glossary-term")
        : null;
    }

    document.addEventListener("mouseover", (event) => {
      const term = findTerm(event);
      if (term) showTooltip(term);
    });

    document.addEventListener("mouseout", (event) => {
      if (findTerm(event)) hideTooltip();
    });

    document.addEventListener("focusin", (event) => {
      const term = findTerm(event);
      if (term) showTooltip(term);
    });

    document.addEventListener("focusout", (event) => {
      if (findTerm(event)) hideTooltip();
    });

    document.addEventListener("click", (event) => {
      const term = findTerm(event);
      if (term) {
        showTooltip(term);
      } else {
        hideTooltip();
      }
    });

    window.addEventListener("scroll", () => {
      if (activeTerm) positionTooltip(activeTerm);
    }, { passive: true });

    window.addEventListener("resize", () => {
      if (activeTerm) positionTooltip(activeTerm);
    });
  }

  window.LLMLabGlossary = {
    apply: applyGlossary,
    terms: TERMS
  };

  document.addEventListener("DOMContentLoaded", () => {
    applyGlossary(document.body);
    setupEvents();
  });
}());
