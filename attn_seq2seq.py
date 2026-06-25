# -*- coding: utf-8 -*-
"""
阶段 2：Attention-Seq2Seq（Bahdanau 加性注意力）

对比阶段 1：Encoder 不再只交一个固定向量，而是保留**每个时刻**的输出；
Decoder 生成每个字时，对 Encoder 所有时刻算一组注意力权重 → 加权求和得到
**动态 context** → 长句信息不再被压丢。这就是对"固定 context 瓶颈"的解药。

手写要点（不调 nn.MultiheadAttention）：
  energy = tanh(W·[decoder_hidden ; encoder_output])
  score  = v·energy                  -> 每个位置一个分数
  weights= softmax(score)            -> 归一化注意力权重（pad 位置屏蔽掉）
  context= Σ weights · encoder_output

用法：
  python attn_seq2seq.py train                 # 训练，存 attn_seq2seq.pt
  python attn_seq2seq.py chat                  # 命令行聊天
  python attn_seq2seq.py heatmap "你今天开心吗"  # 生成回复并画注意力热力图
"""

import sys
import math
import pickle
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import (
    build_dataloader, PAD_ID, BOS_ID, EOS_ID, UNK_ID, clean_text,
)

# ---------------------------------------------------------------------------
# 配置（与阶段1保持一致，便于公平对比）
# ---------------------------------------------------------------------------
CONV_PATH   = "xiaohuangji50w_nofenci.conv"
SAMPLE      = 50000
BATCH_SIZE  = 64
EMB_DIM     = 128
HID_DIM     = 256
NUM_LAYERS  = 1
DROPOUT     = 0.1
EPOCHS      = 12
LR          = 1e-3
CLIP        = 1.0
TF_RATIO    = 0.5
MAX_DEC_LEN = 30
CKPT        = "attn_seq2seq.pt"
VOCAB_PKL   = "attn_vocab.pkl"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ===========================================================================
# Encoder：保留每个时刻的输出（不再只给最后一个向量）
# ===========================================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        # 解包回 [batch, src_len, hid]，total_length 保证和 mask 对齐
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1))
        return outputs, hidden, cell


# ===========================================================================
# Attention：手写加性注意力，返回 context 和 weights（weights 用来画热力图）
# ===========================================================================
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Linear(hid_dim * 2, hid_dim)   # 把[解码状态;编码输出]映射
        self.v = nn.Linear(hid_dim, 1, bias=False) # 打成一个分数

    def forward(self, dec_hidden, enc_outputs, mask):
        # dec_hidden: [batch, hid]  enc_outputs: [batch, src_len, hid]
        src_len = enc_outputs.size(1)
        dec = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)   # [batch, src_len, hid]
        energy = torch.tanh(self.W(torch.cat([dec, enc_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)                    # [batch, src_len]
        scores = scores.masked_fill(mask == 0, -1e9)          # pad 位置不参与
        weights = F.softmax(scores, dim=1)                    # [batch, src_len]
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # [batch, hid]
        return context, weights


# ===========================================================================
# Decoder：每步先算 attention，再把 [字向量 ; 动态context] 喂给 LSTM
# ===========================================================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.attention = Attention(hid_dim)
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid_dim * 2, vocab_size)      # [lstm输出;context]

    def forward(self, input_tok, hidden, cell, enc_outputs, mask):
        embedded = self.embedding(input_tok)                  # [batch, emb]
        context, weights = self.attention(hidden[-1], enc_outputs, mask)  # 用顶层hidden
        lstm_in = torch.cat([embedded, context], dim=1).unsqueeze(1)      # [batch,1,emb+hid]
        output, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        output = output.squeeze(1)                            # [batch, hid]
        pred = self.fc_out(torch.cat([output, context], dim=1))           # [batch, vocab]
        return pred, hidden, cell, weights


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt, tf_ratio=0.5):
        batch, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch, tgt_len, vocab_size, device=src.device)
        mask = (src != PAD_ID)

        enc_outputs, hidden, cell = self.encoder(src, src_lens)
        input_tok = tgt[:, 0]                                 # <bos>
        for t in range(1, tgt_len):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, enc_outputs, mask)
            outputs[:, t] = pred
            teacher = torch.rand(1).item() < tf_ratio
            input_tok = tgt[:, t] if teacher else pred.argmax(1)
        return outputs


def build_model(vocab_size):
    enc = Encoder(vocab_size, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    dec = Decoder(vocab_size, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    return Seq2SeqAttn(enc, dec)


# ===========================================================================
# 训练
# ===========================================================================
def train():
    loader, vocab, _ = build_dataloader(CONV_PATH, batch_size=BATCH_SIZE, sample=SAMPLE)
    with open(VOCAB_PKL, "wb") as f:
        pickle.dump(vocab, f)
    print(f"设备: {DEVICE} | 词表: {len(vocab)} | batch 数/epoch: {len(loader)}")

    model = build_model(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for src, src_lens, tgt, _ in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            output = model(src, src_lens, tgt, TF_RATIO)
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


# ===========================================================================
# 推理：greedy 解码，同时收集每步的注意力权重
# ===========================================================================
@torch.no_grad()
def reply(model, vocab, sentence, return_attn=False):
    model.eval()
    src_tokens = list(jieba.cut(clean_text(sentence)))
    if not src_tokens:                                          # 空输入保护
        return ("（请输入有效内容）", [], [], None) if return_attn else "（请输入有效内容）"
    src = torch.tensor([vocab.encode(src_tokens)], device=DEVICE)
    src_lens = torch.tensor([src.size(1)])
    mask = (src != PAD_ID)
    enc_outputs, hidden, cell = model.encoder(src, src_lens)

    input_tok = torch.tensor([BOS_ID], device=DEVICE)
    out_ids, attns = [], []
    for _ in range(MAX_DEC_LEN):
        pred, hidden, cell, w = model.decoder(input_tok, hidden, cell, enc_outputs, mask)
        logits = pred.squeeze(0)
        logits[UNK_ID] = -1e9                     # 禁止 <unk>
        for tok in set(out_ids):                  # 重复惩罚
            logits[tok] = logits[tok] / 1.3 if logits[tok] > 0 else logits[tok] * 1.3
        nxt = int(logits.argmax(-1))
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
        attns.append(w.squeeze(0).cpu())          # [src_len]
        input_tok = torch.tensor([nxt], device=DEVICE)
    text = "".join(vocab.decode(out_ids))
    if return_attn:
        out_tokens = vocab.decode(out_ids)
        attn_mat = torch.stack(attns).numpy() if attns else None  # [out_len, src_len]
        return text, src_tokens, out_tokens, attn_mat
    return text


def load_for_infer():
    with open(VOCAB_PKL, "rb") as f:
        vocab = pickle.load(f)
    model = build_model(len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model, vocab


def chat():
    model, vocab = load_for_infer()
    print("Attention 版已加载，开始聊天（q 退出）。试试长句，对比阶段1的改善。")
    while True:
        s = input("你: ").strip()
        if s in ("q", "quit", "exit"):
            break
        if not s:
            continue
        print("Bot:", reply(model, vocab, s))


def heatmap(sentence):
    import matplotlib
    import matplotlib.pyplot as plt
    # mac 中文字体，避免方块
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    model, vocab = load_for_infer()
    text, src_tokens, out_tokens, attn = reply(model, vocab, sentence, return_attn=True)
    print("回复:", text)
    if attn is None:
        print("（没有生成内容，换一句试试）")
        return
    fig, ax = plt.subplots(figsize=(max(4, len(src_tokens)), max(3, len(out_tokens) * 0.5)))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(out_tokens)))
    ax.set_yticklabels(out_tokens)
    ax.set_xlabel("输入(问句)")
    ax.set_ylabel("输出(回复)")
    ax.set_title("注意力热力图")
    fig.colorbar(im)
    fig.tight_layout()
    out_png = "attn_heatmap.png"
    fig.savefig(out_png, dpi=150)
    print(f"热力图已保存到 {out_png}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
    elif mode == "chat":
        chat()
    elif mode == "heatmap":
        sent = sys.argv[2] if len(sys.argv) > 2 else "你今天开心吗"
        heatmap(sent)
    else:
        print("用法: python attn_seq2seq.py [train|chat|heatmap <句子>]")
