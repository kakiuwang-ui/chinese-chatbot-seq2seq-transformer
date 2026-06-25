# -*- coding: utf-8 -*-
"""
阶段4 优化：回复多样性评测（Distinct-1 / Distinct-2 + 平均长度）

在同一组测试问句上，对三代模型（及 Transformer 的不同解码策略）统一计算指标，
用数据支撑"安全回复/单调"问题的改善。Distinct 越高 = 回复用词越丰富。

Distinct-n = 不重复的 n-gram 数 / 总 n-gram 数（字级别，无需参考答案）

用法：python eval.py
"""

import os
import pickle
import torch

import seq2seq as S1
import attn_seq2seq as S2
import transformer as S3

# 测试问句（覆盖问候/情感/提问/长句）
TEST_Q = [
    "你好", "你叫什么名字", "在吗", "真的吗", "你今天心情怎么样",
    "我有点难过", "你会做什么", "讲个笑话", "你喜欢我吗",
    "我今天工作上遇到好多烦心事你能安慰我一下吗",
]


def distinct_n(replies, n):
    total, uniq = 0, set()
    for r in replies:
        chars = list(r)
        grams = [tuple(chars[i:i + n]) for i in range(len(chars) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


def avg_len(replies):
    return sum(len(r) for r in replies) / len(replies) if replies else 0.0


def _load(module, ckpt, vocab_pkl):
    if not (os.path.exists(ckpt) and os.path.exists(vocab_pkl)):
        return None
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    model = module.build_model(len(vocab)).to(module.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=module.DEVICE))
    model.eval()
    return model, vocab


def evaluate(name, gen_fn):
    replies = [gen_fn(q) for q in TEST_Q]
    d1, d2, al = distinct_n(replies, 1), distinct_n(replies, 2), avg_len(replies)
    print(f"{name:<34} | Distinct-1 {d1:.3f} | Distinct-2 {d2:.3f} | 平均长度 {al:.1f}")
    return replies


def main():
    rows = []
    m1 = _load(S1, S1.CKPT, S1.VOCAB_PKL)
    m2 = _load(S2, S2.CKPT, S2.VOCAB_PKL)
    m3 = _load(S3, S3.CKPT, S3.VOCAB_PKL)

    print("=" * 78)
    if m1:
        evaluate("① Seq2Seq (greedy+优化)", lambda q: S1.reply(*m1, q))
    if m2:
        evaluate("② Attn-Seq2Seq (greedy+优化)", lambda q: S2.reply(*m2, q))
    if m3:
        # Transformer 三种解码策略对比，凸显解码优化的作用
        evaluate("③ Transformer (greedy)", lambda q: S3.reply(*m3, q, mode="greedy"))
        evaluate("③ Transformer (beam)  ", lambda q: S3.reply(*m3, q, mode="beam"))
        evaluate("③ Transformer (topk)  ", lambda q: S3.reply(*m3, q, mode="topk"))
    print("=" * 78)
    print("提示：Distinct 越高=回复越多样；对比可见解码策略对多样性的影响。")


if __name__ == "__main__":
    main()
