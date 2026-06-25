# -*- coding: utf-8 -*-
"""
阶段 3：从零手写 Transformer（对照《Attention Is All You Need》）

与阶段 1/2 最大的不同：
  - 扔掉 RNN，纯注意力。Encoder/Decoder 都是注意力 + 前馈的堆叠。
  - 训练**并行**：不再像 LSTM 那样一个时间步一个时间步地循环，
    而是把整句一次喂进去（靠 mask 防止解码器偷看未来）。这就是为什么
    Transformer 训练快、能扩展到大模型。

逐模块对应论文：
  Scaled Dot-Product Attention  -> 3.2.1
  Multi-Head Attention          -> 3.2.2
  Positional Encoding           -> 3.5
  Masked self-attention(解码器) -> 3.2.3（上三角 look-ahead mask）
  Cross-attention               -> 3.2.3（Q 来自解码器，K/V 来自编码器）
  FFN / 残差 / LayerNorm         -> 3.3

三种 mask（最容易混淆，搞清就通了 80%）：
  1) src padding mask：编码器自注意力 + 交叉注意力里，屏蔽输入的 <pad>
  2) look-ahead mask：解码器自注意力里，屏蔽"未来的字"（上三角）
  3) tgt padding mask：解码器里同时屏蔽目标的 <pad>，与 2) 取交集

用法：
  python transformer.py train
  python transformer.py chat
  python transformer.py heatmap "你今天心情怎么样"
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
# 配置
# ---------------------------------------------------------------------------
CONV_PATH   = "xiaohuangji50w_nofenci.conv"
SAMPLE      = 50000
BATCH_SIZE  = 64
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 3
D_FF        = 512
DROPOUT     = 0.1
EPOCHS      = 12
LR          = 5e-4
CLIP        = 1.0
MAX_DEC_LEN = 30
CKPT        = "transformer.pt"
VOCAB_PKL   = "tf_vocab.pkl"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ===========================================================================
# Multi-Head Attention（含 Scaled Dot-Product，全手写）
# ===========================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None                         # 存最后一次注意力权重，画热力图用

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        # 线性映射后拆成多头：[B, len, d_model] -> [B, h, len, d_k]
        Q = self.wq(query).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.wk(key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.wv(value).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # Scaled Dot-Product：QKᵀ/√d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:                     # mask: [B,1,1,len] 或 [B,1,len,len]
            scores = scores.masked_fill(mask == 0, -1e9)   # 屏蔽位置→softmax后≈0
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.attn = attn.detach()
        out = torch.matmul(attn, V)              # [B, h, len, d_k]
        # 合并多头：[B, h, len, d_k] -> [B, len, d_model]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.wo(out)


# ===========================================================================
# 位置前馈网络 FFN
# ===========================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ===========================================================================
# 位置编码（正弦版，论文 3.5）：给词向量注入"位置"信息
# ===========================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ===========================================================================
# Encoder / Decoder 层（pre-LN：sublayer 前先 LayerNorm，小数据更稳）
# ===========================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, src_mask))   # 自注意力 + 残差
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))                          # 前馈 + 残差
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 带 look-ahead
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 看编码器
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))          # 1) 掩码自注意力
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, enc_out, enc_out, src_mask))  # 2) 交叉注意力
        h = self.norm3(x)
        x = x + self.dropout(self.ff(h))                                # 3) 前馈
        return x


# ===========================================================================
# 完整 Transformer
# ===========================================================================
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, dropout)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    # --- 三种 mask ---
    def make_src_mask(self, src):
        # padding mask: [B, 1, 1, src_len]
        return (src != PAD_ID).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # padding mask ∩ look-ahead(上三角) mask -> [B, 1, tgt_len, tgt_len]
        pad = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)             # [B,1,1,L]
        L = tgt.size(1)
        sub = torch.tril(torch.ones(L, L, device=tgt.device)).bool()  # [L,L]
        return pad & sub

    def encode(self, src, src_mask):
        x = self.pos(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos(self.embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.dec_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm_dec(x)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.generator(dec_out)           # [B, tgt_len, vocab]


def build_model(vocab_size):
    return Transformer(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)


# ===========================================================================
# 训练（注意：解码器输入是 tgt[:, :-1]，目标是 tgt[:, 1:]，整句并行算！）
# ===========================================================================
def train():
    loader, vocab, _ = build_dataloader(CONV_PATH, batch_size=BATCH_SIZE, sample=SAMPLE)
    with open(VOCAB_PKL, "wb") as f:
        pickle.dump(vocab, f)
    print(f"设备: {DEVICE} | 词表: {len(vocab)} | batch 数/epoch: {len(loader)}")

    model = build_model(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for src, _, tgt, _ in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]   # 右移一位
            opt.zero_grad()
            logits = model(src, tgt_in)                 # 一次算完整句，无循环！
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        print(f"Epoch {epoch:2d} | loss {avg:.4f} | ppl {math.exp(avg):.1f}")
        torch.save(model.state_dict(), CKPT)
    print(f"已保存模型到 {CKPT}")


# ===========================================================================
# 推理 —— 优化后的解码（禁 unk + 重复惩罚 + greedy/beam/topk）
# ===========================================================================
def _adjust_logits(logits, generated, ban_unk, rep_penalty):
    """对单步 logits 做两项优化：禁止 <unk>、对已生成词施加重复惩罚。"""
    if ban_unk:
        logits[UNK_ID] = -1e9                            # 禁止生成 <unk>（解决 P1）
    if rep_penalty and rep_penalty != 1.0:
        for tok in set(generated):                       # 已出现过的词降权（解决 P2 复读）
            logits[tok] = logits[tok] / rep_penalty if logits[tok] > 0 else logits[tok] * rep_penalty
    return logits


@torch.no_grad()
def _encode_src(model, vocab, sentence):
    src_tokens = list(jieba.cut(clean_text(sentence)))
    if not src_tokens:
        return None, None, None
    src = torch.tensor([vocab.encode(src_tokens)], device=DEVICE)
    src_mask = model.make_src_mask(src)
    enc_out = model.encode(src, src_mask)
    return src_tokens, src_mask, enc_out


@torch.no_grad()
def _step_logits(model, ys, enc_out, src_mask):
    tgt_mask = model.make_tgt_mask(ys)
    dec_out = model.decode(ys, enc_out, src_mask, tgt_mask)
    return model.generator(dec_out[:, -1]).squeeze(0)   # [vocab]


@torch.no_grad()
def greedy_decode(model, enc_out, src_mask, ban_unk=True, rep_penalty=1.3):
    ys = torch.tensor([[BOS_ID]], device=DEVICE)
    out_ids = []
    for _ in range(MAX_DEC_LEN):
        logits = _step_logits(model, ys, enc_out, src_mask)
        logits = _adjust_logits(logits, out_ids, ban_unk, rep_penalty)
        nxt = int(logits.argmax(-1))
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
        ys = torch.cat([ys, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return out_ids


@torch.no_grad()
def topk_decode(model, enc_out, src_mask, k=10, temperature=1.0, ban_unk=True, rep_penalty=1.3):
    ys = torch.tensor([[BOS_ID]], device=DEVICE)
    out_ids = []
    for _ in range(MAX_DEC_LEN):
        logits = _step_logits(model, ys, enc_out, src_mask)
        logits = _adjust_logits(logits, out_ids, ban_unk, rep_penalty) / temperature
        topv, topi = logits.topk(k)                      # 只在前 k 个里采样（解决 P3 单调）
        probs = F.softmax(topv, dim=-1)
        nxt = int(topi[torch.multinomial(probs, 1)])
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
        ys = torch.cat([ys, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return out_ids


@torch.no_grad()
def beam_decode(model, enc_out, src_mask, beam=3, ban_unk=True, rep_penalty=1.3):
    # 每个 beam: (累计log概率, [已生成token])
    beams = [(0.0, [])]
    for _ in range(MAX_DEC_LEN):
        cands = []
        for score, seq in beams:
            if seq and seq[-1] == EOS_ID:                # 已结束的 beam 原样保留
                cands.append((score, seq)); continue
            ys = torch.tensor([[BOS_ID] + seq], device=DEVICE)
            logits = _step_logits(model, ys, enc_out, src_mask)
            logits = _adjust_logits(logits, seq, ban_unk, rep_penalty)
            logp = F.log_softmax(logits, dim=-1)
            topv, topi = logp.topk(beam)                 # 每个 beam 扩展 top-beam 个
            for v, i in zip(topv.tolist(), topi.tolist()):
                cands.append((score + v, seq + [i]))
        # 选总分最高的 beam 条（长度归一，避免偏好短句）
        beams = sorted(cands, key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)[:beam]
        if all(s and s[-1] == EOS_ID for _, s in beams):
            break
    best = beams[0][1]
    return [t for t in best if t not in (EOS_ID, BOS_ID)]


@torch.no_grad()
def reply(model, vocab, sentence, mode="beam", beam=3, top_k=10,
          ban_unk=True, rep_penalty=1.3, return_attn=False):
    """mode: greedy | beam | topk。return_attn 仅在 greedy 下提供热力图。"""
    model.eval()
    src_tokens, src_mask, enc_out = _encode_src(model, vocab, sentence)
    if src_tokens is None:
        return ("（请输入有效内容）", [], [], None) if return_attn else "（请输入有效内容）"

    if return_attn:                                      # 热力图走 greedy（注意力清晰）
        out_ids = greedy_decode(model, enc_out, src_mask, ban_unk, rep_penalty)
    elif mode == "greedy":
        out_ids = greedy_decode(model, enc_out, src_mask, ban_unk, rep_penalty)
    elif mode == "topk":
        out_ids = topk_decode(model, enc_out, src_mask, top_k, 1.0, ban_unk, rep_penalty)
    else:
        out_ids = beam_decode(model, enc_out, src_mask, beam, ban_unk, rep_penalty)

    text = "".join(vocab.decode(out_ids))
    if return_attn:
        attn = model.dec_layers[-1].cross_attn.attn      # [1, h, tgt_len, src_len]
        attn = attn.mean(1).squeeze(0).cpu().numpy()[1:1 + len(out_ids)]
        return text, src_tokens, vocab.decode(out_ids), attn
    return text


def load_for_infer():
    with open(VOCAB_PKL, "rb") as f:
        vocab = pickle.load(f)
    model = build_model(len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model, vocab


def chat():
    model, vocab = load_for_infer()
    print("Transformer 已加载，开始聊天（q 退出）。对比阶段1/2的连贯度。")
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
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    model, vocab = load_for_infer()
    text, src_tokens, out_tokens, attn = reply(model, vocab, sentence, return_attn=True)
    print("回复:", text)
    if attn is None or len(out_tokens) == 0:
        print("（没有生成内容，换一句试试）")
        return
    fig, ax = plt.subplots(figsize=(max(4, len(src_tokens)), max(3, len(out_tokens) * 0.5)))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_tokens))); ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(out_tokens))); ax.set_yticklabels(out_tokens)
    ax.set_xlabel("输入(问句)"); ax.set_ylabel("输出(回复)")
    ax.set_title("Transformer 交叉注意力热力图（末层·多头平均）")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig("tf_heatmap.png", dpi=150)
    print("热力图已保存到 tf_heatmap.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
    elif mode == "chat":
        chat()
    elif mode == "heatmap":
        sent = sys.argv[2] if len(sys.argv) > 2 else "你今天心情怎么样"
        heatmap(sent)
    else:
        print("用法: python transformer.py [train|chat|heatmap <句子>]")
