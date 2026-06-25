# -*- coding: utf-8 -*-
"""
阶段 4：三模型训练曲线对比图（ppl / loss）

用法：
  1) 把三个模型每个 epoch 的 ppl 填进下面的 PPL 字典（从训练日志里抄）；
  2) python compare.py  ->  生成 compare_ppl.png
"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 把你训练日志里每个 epoch 的 ppl 填进来（列表长度可以不同）
# 阶段1 已是真实数据；阶段2 是你日志里的前 4 个 epoch（跑完后补全）；阶段3 训练后填。
# ---------------------------------------------------------------------------
PPL = {
    "Seq2Seq (LSTM)":        [296.3, 194.9, 147.4, 113.5, 86.6, 68.7, 53.6, 43.1, 34.6, 29.1, 24.1, 20.4],
    "Attention-Seq2Seq":     [281.2, 153.2, 92.9, 59.3],          # TODO: 跑完补全
    "Transformer":           [],                                   # TODO: 训练后填
}


def main():
    plt.figure(figsize=(8, 5))
    for name, ppl in PPL.items():
        if not ppl:
            continue
        epochs = range(1, len(ppl) + 1)
        plt.plot(epochs, ppl, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("困惑度 Perplexity（越低越好）")
    plt.title("三模型训练困惑度对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare_ppl.png", dpi=150)
    print("已保存 compare_ppl.png")


if __name__ == "__main__":
    main()
