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
import jieba                    # 中文分词：把句子切成词，作为词表的基本单位
import torch
import torch.nn as nn
import torch.nn.functional as F

# 数据管线在同目录的 data_pipeline.py 里：
#   build_dataloader —— 读 .conv 语料、建词表、切分词、pad 成等长张量、返回 DataLoader
#   PAD/BOS/EOS/UNK_ID —— 四个特殊 token 的固定 id（填充 / 句首 / 句尾 / 未登录词）
#   clean_text —— 推理时对用户输入做同样的清洗，保证和训练分布一致
from data_pipeline import (
    build_dataloader, PAD_ID, BOS_ID, EOS_ID, UNK_ID, clean_text,
)

# ---------------------------------------------------------------------------
# 配置（超参数集中放这里，改一处即可）
# ---------------------------------------------------------------------------
CONV_PATH   = "xiaohuangji50w_nofenci.conv"   # 小黄鸡对话语料文件
SAMPLE      = 50000        # 只采样 5 万条对话（全量 50w 太慢，教学够用）
BATCH_SIZE  = 64           # 每个 batch 的对话对数
D_MODEL     = 256          # 模型主维度：词向量维度 / 各层输入输出维度（论文用 512）
N_HEADS     = 8            # 多头注意力的头数（256 / 8 = 每头 32 维）
N_LAYERS    = 3            # 编码器、解码器各堆几层（论文用 6）
D_FF        = 512          # FFN 中间隐藏层维度（一般是 d_model 的 2~4 倍）
DROPOUT     = 0.1          # dropout 比例，防过拟合
EPOCHS      = 20           # 训练轮数
WARMUP      = 2000         # Noam warmup 步数：前期线性升温，之后按 step^-0.5 衰减
CLIP        = 1.0          # 梯度裁剪阈值，防梯度爆炸
MAX_DEC_LEN = 30           # 推理时最多生成多少个词（防止无限生成）
CKPT        = "transformer.pt"     # 模型权重保存路径
VOCAB_PKL   = "tf_vocab.pkl"       # 词表保存路径（推理时要用同一份）

# 优先用 Apple Silicon 的 MPS（GPU）加速，没有就退回 CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ===========================================================================
# Multi-Head Attention（含 Scaled Dot-Product，全手写）
# ===========================================================================
class MultiHeadAttention(nn.Module):
    """一句话直觉：注意力就是「带权检索」——每个词拿着自己的问题(Q)去和所有词的
    标签(K)比对，谁对得上就多抄谁的内容(V)。所谓「多头」，是把这件事在 8 个不同的
    子空间里各做一遍：有的头盯语法、有的头盯指代、有的头盯远距离呼应，最后拼起来，
    比单头只能学一种关系更丰富。深层看：这是全网络里唯一让任意两个位置「直接对话」
    的模块（距离为 1），信息不必像 RNN 那样一步步传递——这正是它能并行、能建长依赖的根。"""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0            # 必须整除，才能把维度均分给每个头
        self.d_k = d_model // n_heads            # 每个头负责的子空间维度
        self.h = n_heads                         # 头数
        # 四个线性层：Q/K/V 各一个投影，输出再一个投影 wo（论文式子里的 W^Q/W^K/W^V/W^O）
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None                         # 存最后一次注意力权重，画热力图用

    def forward(self, query, key, value, mask=None):
        B = query.size(0)                        # batch 大小
        # 线性映射后拆成多头：[B, len, d_model] -> [B, h, len, d_k]
        # view 把 d_model 拆成 (h, d_k)，transpose 把「头」维提到前面，方便每个头独立算注意力
        Q = self.wq(query).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.wk(key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.wv(value).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # Scaled Dot-Product：QKᵀ/√d_k
        # QKᵀ 得到 [B, h, len_q, len_k]，即每个 query 位置对每个 key 位置的相似度
        # 除以 √d_k 是为了防止点积太大、softmax 饱和导致梯度消失
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:                     # mask: [B,1,1,len] 或 [B,1,len,len]
            # 被 mask 的位置填成 -1e9（≈负无穷），softmax 后概率≈0，等于「看不见」
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)         # 在 key 维度上归一化成概率分布
        attn = self.dropout(attn)                # 对注意力权重做 dropout（论文的正则手段）
        self.attn = attn.detach()                # detach 存下来，仅供可视化，不参与反传
        out = torch.matmul(attn, V)              # 用注意力权重对 V 加权求和 -> [B, h, len, d_k]
        # 合并多头：先 transpose 回 [B, len, h, d_k]，再 view 拼回 [B, len, d_model]
        # contiguous 是因为 transpose 后内存不连续，view 前必须整理
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.wo(out)                      # 过输出投影，混合各头信息


# ===========================================================================
# 位置前馈网络 FFN（论文 3.3）：对每个位置独立地做一次「升维-激活-降维」
# ===========================================================================
class PositionwiseFeedForward(nn.Module):
    """一句话直觉：注意力负责「跨位置搬运信息」，FFN 负责「就地深加工」。它对每个位置
    单独套同一个小型两层网络，谁也不看谁——先升维到更宽的空间里把特征摊开、用 ReLU
    非线性地筛选，再压回原维度。可以理解成注意力管「你该和谁说话」，FFN 管「听完之后
    自己想清楚」。参数量其实大头在这儿，是模型真正「记住知识」的地方。"""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)      # 升维：d_model -> d_ff
        self.fc2 = nn.Linear(d_ff, d_model)      # 降维：d_ff -> d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 先升维 + ReLU 引入非线性 + dropout，再降维回原维度
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ===========================================================================
# 位置编码（正弦版，论文 3.5）：给词向量注入"位置"信息
# 注意力本身对顺序无感（打乱输入结果一样），所以要显式告诉模型每个词的位置
# ===========================================================================
class PositionalEncoding(nn.Module):
    """一句话直觉：注意力是「一袋词」——把句子里的词打乱，算出来的结果一模一样，它天生
    不知道谁先谁后。所以要在词向量上「盖一个位置邮戳」。这里用不同频率的 sin/cos 组合当
    邮戳：低频维度像时针（管全局第几个词），高频维度像秒针（管相邻细节），合起来每个位置
    都有独一无二的波形指纹。妙处在于相对位置可由三角恒等式线性表示，模型容易学到「隔几个词」
    这种相对关系，还能外推到训练时没见过的更长句子。"""

    def __init__(self, d_model, dropout, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)                       # [max_len, d_model] 预分配
        pos = torch.arange(0, max_len).unsqueeze(1).float()      # 位置列向量 [max_len, 1]
        # div 是不同维度对应的频率（几何级数衰减），实现论文里的 10000^(2i/d_model)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)                       # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)                       # 奇数维用 cos
        # register_buffer：随模型保存/搬设备，但不是可训练参数（位置编码是固定的）
        self.register_buffer("pe", pe.unsqueeze(0))              # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]           # 只取当前句长的那一段位置编码，加到词向量上
        return self.dropout(x)


# ===========================================================================
# Encoder / Decoder 层（pre-LN：sublayer 前先 LayerNorm，小数据更稳）
# 结构统一是：x = x + Dropout(SubLayer(LayerNorm(x)))，即「残差 + 前置归一化」
# ===========================================================================
class EncoderLayer(nn.Module):
    """一句话直觉：一个编码器层 = 「互相看一眼(自注意力) + 各自消化(FFN)」。整句里每个词
    先环顾全场、根据上下文更新自己的含义（"苹果"看到"吃"就偏向水果、看到"股价"就偏向公司），
    再各自过 FFN 深加工。堆 N 层，就是把这个「看—想」循环做 N 遍，理解逐层加深。
    残差(x + ...)是关键：它保证每层只需学「在原表示上补一点修正」，梯度能直通到底层，
    深网络才训得动；pre-LN（子层前先归一化）则让这个过程在小数据上更稳、不易发散。"""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 自注意力子层
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)        # 前馈子层
        self.norm1 = nn.LayerNorm(d_model)       # 自注意力前的归一化
        self.norm2 = nn.LayerNorm(d_model)       # 前馈前的归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, src_mask))   # 自注意力 + 残差（Q=K=V=h）
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))                          # 前馈 + 残差
        return x


class DecoderLayer(nn.Module):
    """一句话直觉：解码器层比编码器多一次「抬头看原文」。它有三步——先「回顾已经写出的字」
    (掩码自注意力，只能往左看，因为未来还没生成)，再「回去查问句」(交叉注意力，Q 来自
    自己、K/V 来自编码器输出，决定这一步该聚焦输入的哪几个词)，最后 FFN 消化。这正是
    seq2seq 的精髓：一边参考原文、一边顺着自己已写的内容，逐字接龙。那个 look-ahead 掩码
    是训练能整句并行的命门——它让「预测第 i 个字」时严格看不到第 i 个及之后的答案，
    于是一次前向就能同时算出所有位置的预测，而不必真的一个字一个字地跑。"""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 带 look-ahead 的自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 交叉注意力：看编码器
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)       # 掩码自注意力前
        self.norm2 = nn.LayerNorm(d_model)       # 交叉注意力前
        self.norm3 = nn.LayerNorm(d_model)       # 前馈前
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        h = self.norm1(x)
        # 1) 掩码自注意力：Q=K=V 都来自解码器自身，tgt_mask 保证只能看前面的词
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))
        h = self.norm2(x)
        # 2) 交叉注意力：Q 来自解码器(h)，K/V 来自编码器输出(enc_out)，src_mask 屏蔽输入 <pad>
        x = x + self.dropout(self.cross_attn(h, enc_out, enc_out, src_mask))
        h = self.norm3(x)
        # 3) 前馈 + 残差
        x = x + self.dropout(self.ff(h))
        return x


# ===========================================================================
# 完整 Transformer（编码器 + 解码器 + 输出层，串成一个 seq2seq）
# ===========================================================================
class Transformer(nn.Module):
    """一句话直觉：整体就是「读懂问句 → 逐字写回答」。编码器把整句问话反复咀嚼成一组
    富含上下文的向量（一份「读后理解」），解码器再拿着这份理解、顺着自己已写的字往下接龙，
    每步在全词表上打分选下一个词。全流程只靠 embedding + 注意力 + FFN，没有一个循环结构——
    这就是它相比 RNN 的根本优势：训练时整句并行、天然建长距离依赖、堆层数就能变强，
    从而一路撑起了后来的大模型。"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        # 词嵌入：把 token id 映射成向量；padding_idx 让 <pad> 的向量恒为 0 且不更新
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, dropout)          # 位置编码
        # N 层编码器
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        # N 层解码器
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm_enc = nn.LayerNorm(d_model)    # pre-LN 架构需要在最后再补一次归一化
        self.norm_dec = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)          # 输出层：映射到词表大小，得到每个词的分数
        self.d_model = d_model

    # --- 三种 mask ---
    def make_src_mask(self, src):
        # padding mask: 标出哪些位置不是 <pad>（True 保留 / False 屏蔽）
        # unsqueeze 两次是为了广播到注意力 scores 的 [B, h, len_q, len_k] 形状 -> [B, 1, 1, src_len]
        return (src != PAD_ID).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # 解码器需要「padding mask ∩ look-ahead(上三角) mask」两个约束同时成立
        pad = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)             # [B,1,1,L] 屏蔽目标 <pad>
        L = tgt.size(1)
        # torch.tril 取下三角（含对角线）为 1：第 i 行只有前 i 列是 1，即第 i 个位置只能看 0..i
        sub = torch.tril(torch.ones(L, L, device=tgt.device)).bool()  # [L,L]
        return pad & sub                                           # 广播取交集 -> [B,1,L,L]

    def encode(self, src, src_mask):
        # 词嵌入乘 √d_model 是论文的缩放约定（让嵌入和位置编码量级匹配），再加位置编码
        x = self.pos(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.enc_layers:            # 逐层过编码器
            x = layer(x, src_mask)
        return self.norm_enc(x)                  # 末尾归一化，得到「输入的语义表示」

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos(self.embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.dec_layers:            # 逐层过解码器（每层都会看一眼 enc_out）
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm_dec(x)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)       # 构造输入侧 mask
        tgt_mask = self.make_tgt_mask(tgt)       # 构造目标侧 mask
        enc_out = self.encode(src, src_mask)     # 编码问句
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)   # 解码生成
        return self.generator(dec_out)           # [B, tgt_len, vocab]，每个位置对全词表的打分


def build_model(vocab_size):
    # 用全局超参数造一个模型实例，训练和推理共用，保证结构一致
    return Transformer(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)


# ===========================================================================
# 训练（注意：解码器输入是 tgt[:, :-1]，目标是 tgt[:, 1:]，整句并行算！）
# ===========================================================================
def noam_lr(step):
    """论文 5.3 的学习率调度：lr = d_model^-0.5 · min(step^-0.5, step·warmup^-1.5)。
    前 WARMUP 步线性升温（避免 Adam 初期大步长把大模型带崩），之后按 step^-0.5 衰减。
    这是 Transformer 收敛的关键，缺它小数据上 ppl 会明显偏高。"""
    step = max(step, 1)                          # 防 step=0 时出现 0 的负次幂
    # min(...) 的两项：step^-0.5 是衰减段，step·warmup^-1.5 是升温段，二者在 step=warmup 处相交
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP ** -1.5))


def train():
    # 建 DataLoader 和词表；vocab 之后要保存，推理时必须用同一份
    loader, vocab, _ = build_dataloader(CONV_PATH, batch_size=BATCH_SIZE, sample=SAMPLE)
    with open(VOCAB_PKL, "wb") as f:             # 把词表序列化到磁盘
        pickle.dump(vocab, f)
    print(f"设备: {DEVICE} | 词表: {len(vocab)} | batch 数/epoch: {len(loader)}")

    model = build_model(len(vocab)).to(DEVICE)   # 造模型并搬到 MPS/CPU
    # 论文配置：Adam betas=(0.9,0.98), eps=1e-9；lr 由 noam_lr 每步覆盖，初值随意
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    # 损失用交叉熵；ignore_index=PAD_ID 让填充位置不计入 loss（否则会被大量 <pad> 带偏）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    step = 0                                     # 全局步数计数器，喂给 noam_lr
    for epoch in range(1, EPOCHS + 1):
        model.train()                            # 训练模式（启用 dropout）
        total = 0.0                              # 累计本 epoch 的 loss
        for src, _, tgt, _ in loader:            # loader 返回 (src, src_len, tgt, tgt_len)，长度这里用不到
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            # Teacher forcing + 右移一位：
            #   tgt_in  = 去掉最后一个词（喂给解码器的输入，通常以 <bos> 开头）
            #   tgt_out = 去掉第一个词（要预测的目标，即每个位置的「下一个词」）
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            step += 1
            for g in opt.param_groups:                  # Noam：每步按公式覆盖学习率
                g["lr"] = noam_lr(step)
            opt.zero_grad()                             # 清空上一步的梯度
            logits = model(src, tgt_in)                 # 一次算完整句，无循环！[B, L, vocab]
            # reshape 成 [B*L, vocab] 和 [B*L]，逐 token 算交叉熵
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()                             # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)   # 梯度裁剪，防爆炸
            opt.step()                                  # 更新参数
            total += loss.item()
        avg = total / len(loader)                       # 本 epoch 平均 loss
        # ppl（困惑度）= exp(loss)，越低越好；越接近 1 说明模型越「确定」
        print(f"Epoch {epoch:2d} | loss {avg:.4f} | ppl {math.exp(avg):.1f} | lr {noam_lr(step):.2e}")
        torch.save(model.state_dict(), CKPT)            # 每个 epoch 存一次权重
    print(f"已保存模型到 {CKPT}")


# ===========================================================================
# 推理 —— 优化后的解码（禁 unk + 重复惩罚 + greedy/beam/topk）
# ===========================================================================
def _adjust_logits(logits, generated, ban_unk, rep_penalty):
    """对单步 logits 做两项优化：禁止 <unk>、对已生成词施加重复惩罚。"""
    if ban_unk:
        logits[UNK_ID] = -1e9                            # 禁止生成 <unk>（解决 P1：老是蹦未登录词）
    if rep_penalty and rep_penalty != 1.0:
        for tok in set(generated):                       # 已出现过的词降权（解决 P2：复读机）
            # 正 logit 除以惩罚系数、负 logit 乘惩罚系数，都是让它的概率变小
            logits[tok] = logits[tok] / rep_penalty if logits[tok] > 0 else logits[tok] * rep_penalty
    return logits


@torch.no_grad()                                         # 推理不需要梯度，省显存加速
def _encode_src(model, vocab, sentence):
    """把用户输入的句子清洗、分词、编码，得到编码器输出（供后续解码复用）。"""
    src_tokens = list(jieba.cut(clean_text(sentence)))   # 清洗 + 分词
    if not src_tokens:                                   # 空输入直接返回 None
        return None, None, None
    src = torch.tensor([vocab.encode(src_tokens)], device=DEVICE)   # 词 -> id，加 batch 维
    src_mask = model.make_src_mask(src)
    enc_out = model.encode(src, src_mask)                # 编码只需算一次，解码每步复用
    return src_tokens, src_mask, enc_out


@torch.no_grad()
def _step_logits(model, ys, enc_out, src_mask):
    """给定已生成序列 ys，算出「下一个词」的 logits（词表上的分数）。"""
    tgt_mask = model.make_tgt_mask(ys)                   # 对当前已生成序列构造 mask
    dec_out = model.decode(ys, enc_out, src_mask, tgt_mask)
    # 只取最后一个位置的输出（即对下一个词的预测），squeeze 掉 batch 维 -> [vocab]
    return model.generator(dec_out[:, -1]).squeeze(0)


@torch.no_grad()
def greedy_decode(model, enc_out, src_mask, ban_unk=True, rep_penalty=1.3):
    """贪心解码：每步都选概率最大的词。最确定，但容易单调/复读。"""
    ys = torch.tensor([[BOS_ID]], device=DEVICE)         # 以 <bos> 开头
    out_ids = []
    for _ in range(MAX_DEC_LEN):
        logits = _step_logits(model, ys, enc_out, src_mask)
        logits = _adjust_logits(logits, out_ids, ban_unk, rep_penalty)
        nxt = int(logits.argmax(-1))                     # 取分数最高的词
        if nxt == EOS_ID:                                # 生成到句尾就停
            break
        out_ids.append(nxt)
        ys = torch.cat([ys, torch.tensor([[nxt]], device=DEVICE)], dim=1)  # 拼到序列末尾
    return out_ids


@torch.no_grad()
def topk_decode(model, enc_out, src_mask, k=10, temperature=1.0, ban_unk=True, rep_penalty=1.3):
    """Top-k 采样：只在概率最高的 k 个词里按概率随机采样，增加多样性（解决 P3：回答太单调）。"""
    ys = torch.tensor([[BOS_ID]], device=DEVICE)
    out_ids = []
    for _ in range(MAX_DEC_LEN):
        logits = _step_logits(model, ys, enc_out, src_mask)
        # temperature 调节随机性：>1 更随机，<1 更保守
        logits = _adjust_logits(logits, out_ids, ban_unk, rep_penalty) / temperature
        topv, topi = logits.topk(k)                      # 取分数最高的 k 个
        probs = F.softmax(topv, dim=-1)                  # 在这 k 个里归一化成概率
        nxt = int(topi[torch.multinomial(probs, 1)])     # 按概率抽一个
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
        ys = torch.cat([ys, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return out_ids


@torch.no_grad()
def beam_decode(model, enc_out, src_mask, beam=3, ban_unk=True, rep_penalty=1.3):
    """束搜索：同时维护 beam 条候选序列，每步扩展并保留总分最高的几条，比贪心更可能找到好句子。"""
    # 每个 beam: (累计log概率, [已生成token])
    beams = [(0.0, [])]
    for _ in range(MAX_DEC_LEN):
        cands = []                                       # 本步所有候选扩展
        for score, seq in beams:
            if seq and seq[-1] == EOS_ID:                # 已结束的 beam 原样保留，不再扩展
                cands.append((score, seq)); continue
            ys = torch.tensor([[BOS_ID] + seq], device=DEVICE)
            logits = _step_logits(model, ys, enc_out, src_mask)
            logits = _adjust_logits(logits, seq, ban_unk, rep_penalty)
            logp = F.log_softmax(logits, dim=-1)         # 用 log 概率，便于累加
            topv, topi = logp.topk(beam)                 # 每个 beam 扩展 top-beam 个后继
            for v, i in zip(topv.tolist(), topi.tolist()):
                cands.append((score + v, seq + [i]))     # 累计分数 = 原分 + 新词 log 概率
        # 选总分最高的 beam 条（除以长度做归一，避免偏好短句）
        beams = sorted(cands, key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)[:beam]
        if all(s and s[-1] == EOS_ID for _, s in beams):  # 所有候选都结束了就停
            break
    best = beams[0][1]                                   # 取分最高的那条
    return [t for t in best if t not in (EOS_ID, BOS_ID)]  # 去掉特殊 token


@torch.no_grad()
def reply(model, vocab, sentence, mode="beam", beam=3, top_k=10,
          ban_unk=True, rep_penalty=1.3, return_attn=False):
    """对外统一入口。mode: greedy | beam | topk。return_attn 仅在 greedy 下提供热力图。"""
    model.eval()                                         # 评估模式（关 dropout）
    src_tokens, src_mask, enc_out = _encode_src(model, vocab, sentence)
    if src_tokens is None:                               # 空输入的兜底
        return ("（请输入有效内容）", [], [], None) if return_attn else "（请输入有效内容）"

    if return_attn:                                      # 热力图走 greedy（注意力最清晰）
        out_ids = greedy_decode(model, enc_out, src_mask, ban_unk, rep_penalty)
    elif mode == "greedy":
        out_ids = greedy_decode(model, enc_out, src_mask, ban_unk, rep_penalty)
    elif mode == "topk":
        out_ids = topk_decode(model, enc_out, src_mask, top_k, 1.0, ban_unk, rep_penalty)
    else:
        out_ids = beam_decode(model, enc_out, src_mask, beam, ban_unk, rep_penalty)

    text = "".join(vocab.decode(out_ids))                # id -> 词 -> 拼成字符串
    if return_attn:
        # 取末层交叉注意力：[1, h, tgt_len, src_len]
        attn = model.dec_layers[-1].cross_attn.attn
        # 多头求平均、去 batch 维，再切掉 <bos> 那一行、只留实际生成的部分
        attn = attn.mean(1).squeeze(0).cpu().numpy()[1:1 + len(out_ids)]
        return text, src_tokens, vocab.decode(out_ids), attn
    return text


def load_for_infer():
    """加载词表 + 模型权重，供 chat / heatmap 复用。"""
    with open(VOCAB_PKL, "rb") as f:
        vocab = pickle.load(f)
    model = build_model(len(vocab)).to(DEVICE)
    # map_location 保证在不同设备（如训练时 mps、推理时 cpu）间也能正确加载
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model, vocab


def chat():
    """命令行交互聊天：循环读输入、生成回复，输入 q/quit/exit 退出。"""
    model, vocab = load_for_infer()
    print("Transformer 已加载，开始聊天（q 退出）。对比阶段1/2的连贯度。")
    while True:
        s = input("你: ").strip()
        if s in ("q", "quit", "exit"):
            break
        if not s:                                        # 空行跳过
            continue
        print("Bot:", reply(model, vocab, s))


def heatmap(sentence):
    """生成回复并把交叉注意力画成热力图，直观看到「输出每个字主要看了输入哪些词」。"""
    import matplotlib
    import matplotlib.pyplot as plt
    # 指定中文字体，否则 matplotlib 画中文会变成方框
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    model, vocab = load_for_infer()
    # return_attn=True 会额外返回注意力矩阵 attn
    text, src_tokens, out_tokens, attn = reply(model, vocab, sentence, return_attn=True)
    print("回复:", text)
    if attn is None or len(out_tokens) == 0:             # 没生成内容就不画
        print("（没有生成内容，换一句试试）")
        return
    # 画布大小随句长自适应
    fig, ax = plt.subplots(figsize=(max(4, len(src_tokens)), max(3, len(out_tokens) * 0.5)))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")  # 用颜色深浅表示注意力权重
    ax.set_xticks(range(len(src_tokens))); ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(out_tokens))); ax.set_yticklabels(out_tokens)
    ax.set_xlabel("输入(问句)"); ax.set_ylabel("输出(回复)")
    ax.set_title("Transformer 交叉注意力热力图（末层·多头平均）")
    fig.colorbar(im); fig.tight_layout()                 # 加颜色条、自动排版
    fig.savefig("tf_heatmap.png", dpi=150)
    print("热力图已保存到 tf_heatmap.png")


if __name__ == "__main__":
    # 命令行第一个参数选模式，默认 train
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
