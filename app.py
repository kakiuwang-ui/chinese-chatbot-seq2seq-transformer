# -*- coding: utf-8 -*-
"""
阶段 4：Gradio 三模型对比聊天界面

加载阶段 1/2/3 三个已训练模型，输入一句话，**并排显示三者的回复**，
直观展示技术演进带来的改善。缺哪个模型(没训练)就自动跳过。

依赖：pip install gradio
用法：python app.py  ->  浏览器打开提示的地址
"""

import os
import pickle
import torch

import seq2seq as S1
import attn_seq2seq as S2
import transformer as S3


# ---------------------------------------------------------------------------
# 加载三个模型（缺失的 checkpoint 自动跳过）
# ---------------------------------------------------------------------------
def _load(module, ckpt, vocab_pkl):
    if not (os.path.exists(ckpt) and os.path.exists(vocab_pkl)):
        return None
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    model = module.build_model(len(vocab)).to(module.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=module.DEVICE))
    model.eval()
    return model, vocab


MODELS = {
    "① Seq2Seq (LSTM)":    (_load(S1, S1.CKPT, S1.VOCAB_PKL), S1.reply),
    "② Attention-Seq2Seq": (_load(S2, S2.CKPT, S2.VOCAB_PKL), S2.reply),
    "③ Transformer":       (_load(S3, S3.CKPT, S3.VOCAB_PKL), S3.reply),
}


def chat_all(message):
    outs = []
    for name, (loaded, reply_fn) in MODELS.items():
        if loaded is None:
            outs.append(f"**{name}**：（未训练，跳过）")
        else:
            model, vocab = loaded
            outs.append(f"**{name}**：{reply_fn(model, vocab, message)}")
    return "\n\n".join(outs)


def main():
    import gradio as gr
    available = [n for n, (m, _) in MODELS.items() if m is not None]
    print("已加载模型：", available or "（无，请先训练）")

    demo = gr.Interface(
        fn=chat_all,
        inputs=gr.Textbox(label="你说", placeholder="输入一句话，对比三代模型的回复…"),
        outputs=gr.Markdown(label="三模型回复"),
        title="中文聊天机器人 · 三代模型对比",
        description="Seq2Seq → Attention-Seq2Seq → Transformer，观察技术演进带来的改善。",
        examples=["你好", "你叫什么名字", "我今天心情不太好你能安慰我一下吗"],
    )
    demo.launch()


if __name__ == "__main__":
    main()
