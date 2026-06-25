# -*- coding: utf-8 -*-
"""
阶段 4：终端版三模型对比聊天（不依赖 gradio，兜底用）

输入一句话，依次打印三代模型的回复。缺哪个模型自动跳过。
用法：python compare_chat.py
"""

import os
import pickle
import torch

import seq2seq as S1
import attn_seq2seq as S2
import transformer as S3


def _load(module, ckpt, vocab_pkl):
    if not (os.path.exists(ckpt) and os.path.exists(vocab_pkl)):
        return None
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    model = module.build_model(len(vocab)).to(module.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=module.DEVICE))
    model.eval()
    return model, vocab


MODELS = [
    ("① Seq2Seq        ", _load(S1, S1.CKPT, S1.VOCAB_PKL), S1.reply),
    ("② Attn-Seq2Seq   ", _load(S2, S2.CKPT, S2.VOCAB_PKL), S2.reply),
    ("③ Transformer    ", _load(S3, S3.CKPT, S3.VOCAB_PKL), S3.reply),
]


def main():
    avail = [n.strip() for n, m, _ in MODELS if m is not None]
    print("已加载：", avail or "（无，请先训练）")
    print("输入一句话对比三模型（q 退出）")
    while True:
        s = input("\n你: ").strip()
        if s in ("q", "quit", "exit"):
            break
        if not s:
            continue
        for name, loaded, reply_fn in MODELS:
            if loaded is None:
                print(f"{name}: （未训练，跳过）")
            else:
                model, vocab = loaded
                print(f"{name}: {reply_fn(model, vocab, s)}")


if __name__ == "__main__":
    main()
