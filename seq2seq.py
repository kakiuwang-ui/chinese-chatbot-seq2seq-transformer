# -*- coding: utf-8 -*-
"""
阶段 1：Seq2Seq (LSTM) 基线，**不带 Attention**。

目的：训练出第一个能对话的模型，并亲手撞上"固定 context 向量瓶颈"——
Encoder 把整句问句压成一个固定长度的向量，句子越长信息丢得越多，
回复就越不相关。这正是 阶段2 引入 Attention 的动机。

核心概念：
  - Teacher forcing：训练时解码器输入喂"真值上一个字"，而非自己上一步的预测
  - 自回归解码：推理时只能喂自己上一步的预测，从 <bos> 开始直到生成 <eos>
  - context 向量：Encoder 最后的 (hidden, cell)，是问句的全部"记忆"

用法：
  python seq2seq.py train     # 训练并保存到 seq2seq.pt
  python seq2seq.py chat      # 加载模型，命令行聊天
"""

import sys
import math
import pickle
import jieba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline import (
    build_dataloader, Vocab, ChatDataset, collate_fn,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID, clean_text,
)

# ---------------------------------------------------------------------------
# 配置（想跑快就把 SAMPLE 调小、EPOCHS 调小）
# ---------------------------------------------------------------------------
CONV_PATH   = "xiaohuangji50w_nofenci.conv"
SAMPLE      = 50000      # 抽样多少对话对（先小后大）
BATCH_SIZE  = 64
EMB_DIM     = 128
HID_DIM     = 256
NUM_LAYERS  = 1
DROPOUT     = 0.1
EPOCHS      = 12
LR          = 1e-3
CLIP        = 1.0        # 梯度裁剪，防 LSTM 梯度爆炸
TF_RATIO    = 0.5        # teacher forcing 概率
MAX_DEC_LEN = 30         # 推理时最多生成多少字
CKPT        = "seq2seq.pt"
VOCAB_PKL   = "vocab.pkl"

# Mac M 系列优先用 mps；若报错（pack_padded 在 mps 偶有 bug）改成 "cpu"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ===========================================================================
# Encoder：把问句压成固定的 (hidden, cell) —— 这就是"瓶颈"所在
# ===========================================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src, src_lens):
        # src: [batch, src_len]
        embedded = self.embedding(src)                      # [batch, src_len, emb]
        # pack：让 LSTM 跳过 pad 部分，得到"真实最后一步"的 hidden，避免被 pad 污染
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed)
        return hidden, cell        # 每个 [num_layers, batch, hid_dim]


# ===========================================================================
# Decoder：每次只走一步，吃"上一个字 + 上一步状态"，吐下一个字的分布
# ===========================================================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input_tok, hidden, cell):
        # input_tok: [batch]  -> 变成 [batch, 1]
        input_tok = input_tok.unsqueeze(1)
        embedded = self.embedding(input_tok)                # [batch, 1, emb]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        pred = self.fc_out(output.squeeze(1))               # [batch, vocab]
        return pred, hidden, cell


# ===========================================================================
# Seq2Seq：把 Encoder 的 context 交给 Decoder，逐字生成
# ===========================================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt, tf_ratio=0.5):
        batch, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch, tgt_len, vocab_size, device=src.device)

        hidden, cell = self.encoder(src, src_lens)
        input_tok = tgt[:, 0]                               # 第一个字是 <bos>
        for t in range(1, tgt_len):
            pred, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = pred
            teacher = torch.rand(1).item() < tf_ratio
            input_tok = tgt[:, t] if teacher else pred.argmax(1)
        return outputs


# ===========================================================================
# 训练
# ===========================================================================
def train():
    loader, vocab, _ = build_dataloader(
        CONV_PATH, batch_size=BATCH_SIZE, sample=SAMPLE)
    with open(VOCAB_PKL, "wb") as f:                        # 保存词表给 chat 用
        pickle.dump(vocab, f)
    print(f"设备: {DEVICE} | 词表: {len(vocab)} | batch 数/epoch: {len(loader)}")

    model = build_model(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)   # pad 不计入 loss

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for src, src_lens, tgt, _ in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            output = model(src, src_lens, tgt, TF_RATIO)    # [batch, tgt_len, vocab]
            # 对齐：丢掉第 0 位(<bos>)，预测从第 1 个字算起
            output = output[:, 1:].reshape(-1, output.size(-1))
            gold = tgt[:, 1:].reshape(-1)
            loss = criterion(output, gold)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        print(f"Epoch {epoch:2d} | loss {avg:.4f} | ppl {math.exp(avg):.1f}")
        torch.save(model.state_dict(), CKPT)
    print(f"已保存模型到 {CKPT}")


def build_model(vocab_size):
    enc = Encoder(vocab_size, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    dec = Decoder(vocab_size, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    return Seq2Seq(enc, dec)


# ===========================================================================
# 推理：greedy 自回归解码
# ===========================================================================
@torch.no_grad()
def reply(model, vocab, sentence):
    model.eval()
    tokens = list(jieba.cut(clean_text(sentence)))
    if not tokens:                                              # 空输入保护
        return "（请输入有效内容）"
    src = torch.tensor([vocab.encode(tokens)], device=DEVICE)   # [1, len]
    src_lens = torch.tensor([src.size(1)])
    hidden, cell = model.encoder(src, src_lens)

    input_tok = torch.tensor([BOS_ID], device=DEVICE)
    out_ids = []
    for _ in range(MAX_DEC_LEN):
        pred, hidden, cell = model.decoder(input_tok, hidden, cell)
        logits = pred.squeeze(0)
        logits[UNK_ID] = -1e9                               # 禁止 <unk>
        for tok in set(out_ids):                            # 重复惩罚
            logits[tok] = logits[tok] / 1.3 if logits[tok] > 0 else logits[tok] * 1.3
        nxt = int(logits.argmax(-1))
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
        input_tok = torch.tensor([nxt], device=DEVICE)
    return "".join(vocab.decode(out_ids))


def chat():
    with open(VOCAB_PKL, "rb") as f:
        vocab = pickle.load(f)
    model = build_model(len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print("模型已加载，开始聊天（输入 q 退出）。试试长句，感受'固定 context 瓶颈'。")
    while True:
        s = input("你: ").strip()
        if s in ("q", "quit", "exit"):
            break
        if not s:
            continue
        print("Bot:", reply(model, vocab, s))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
    elif mode == "chat":
        chat()
    else:
        print("用法: python seq2seq.py [train|chat]")
