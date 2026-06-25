# -*- coding: utf-8 -*-
"""
阶段 0：数据管线（小黄鸡 .conv -> 清洗 -> 词表 -> DataLoader）
后续 阶段1 Seq2Seq / 阶段2 Attention / 阶段3 Transformer 都复用本文件。

用法：
    python data_pipeline.py            # 自检：解析、清洗、建词表、打印一个 batch
依赖：
    pip install torch jieba
"""

import re
import jieba
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# 特殊符号：四个一定要有
#   <pad> 填充，<bos> 解码起始，<eos> 解码结束，<unk> 未登录词
# ---------------------------------------------------------------------------
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3   # 与 SPECIALS 顺序一一对应


# ===========================================================================
# 1. 解析 .conv -> (问, 答) 句对
# ===========================================================================
def load_conv(path):
    """读取小黄鸡 .conv 文件，返回 [(问, 答), ...]。"""
    pairs, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "E":                       # 新对话，结算上一组
                if len(cur) >= 2:
                    pairs.append((cur[0], cur[1]))
                cur = []
            elif line.startswith("M "):
                cur.append(line[2:].strip())      # 去掉开头的 "M "
        if len(cur) >= 2:                         # 别漏掉最后一组
            pairs.append((cur[0], cur[1]))
    return pairs


# ===========================================================================
# 2. 清洗：去 emoji/颜文字、过滤过短或纯符号的句子
#    小黄鸡噪声多、表情多，这一步本身就是文档里的"数据清洗"亮点。
# ===========================================================================
# 只保留中文、英文、数字和常见标点；其余（emoji/特殊符号）一律删掉
_KEEP = re.compile(r"[^一-龥a-zA-Z0-9，。！？、,.!?]")
# 判断一句话里是否还有中文/字母/数字（用来过滤纯符号句）
_HAS_CONTENT = re.compile(r"[一-龥a-zA-Z0-9]")


def clean_text(s):
    s = _KEEP.sub("", s)
    return s.strip()


def clean_pairs(pairs, min_len=1, max_len=30):
    """清洗 + 长度过滤。max_len 指词数上限，过长的句子先丢掉，训练更稳。"""
    out = []
    for q, a in pairs:
        q, a = clean_text(q), clean_text(a)
        if not q or not a:
            continue
        if not _HAS_CONTENT.search(q) or not _HAS_CONTENT.search(a):
            continue                              # 纯符号/空，丢弃
        qs, as_ = list(jieba.cut(q)), list(jieba.cut(a))
        if not (min_len <= len(qs) <= max_len):
            continue
        if not (min_len <= len(as_) <= max_len):
            continue
        out.append((qs, as_))                     # 注意：这里存的已经是分好的词列表
    return out


# ===========================================================================
# 3. 词表
# ===========================================================================
class Vocab:
    def __init__(self, tokenized_pairs, min_freq=2, max_size=30000):
        counter = Counter()
        for qs, as_ in tokenized_pairs:
            counter.update(qs)
            counter.update(as_)
        # 特殊符号占前 4 个 id
        self.itos = list(SPECIALS)
        for tok, freq in counter.most_common():
            if freq < min_freq or len(self.itos) >= max_size:
                break
            self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


# ===========================================================================
# 4. Dataset + collate（padding，给 target 加 <bos>/<eos>）
# ===========================================================================
class ChatDataset(Dataset):
    def __init__(self, tokenized_pairs, vocab):
        self.data = tokenized_pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qs, as_ = self.data[idx]
        src = self.vocab.encode(qs)                      # 输入（问）
        tgt = [BOS_ID] + self.vocab.encode(as_) + [EOS_ID]  # 目标（答），包 bos/eos
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    """把一个 batch 内不等长的句子 pad 到等长，并返回长度信息。"""
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(s) for s in srcs])
    tgt_lens = torch.tensor([len(t) for t in tgts])
    src_pad = pad_sequence(srcs)
    tgt_pad = pad_sequence(tgts)
    return src_pad, src_lens, tgt_pad, tgt_lens


def pad_sequence(seqs):
    """用 PAD_ID 把一组 1D tensor 补齐成 [batch, max_len]。"""
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


# ===========================================================================
# 一键构建：原始文件 -> DataLoader + vocab
# ===========================================================================
def build_dataloader(conv_path, batch_size=64, sample=None,
                     min_freq=2, max_len=30, shuffle=True):
    pairs = load_conv(conv_path)
    if sample:                                    # 先抽样快速迭代（建议 5w~10w）
        pairs = pairs[:sample]
    tokenized = clean_pairs(pairs, max_len=max_len)
    vocab = Vocab(tokenized, min_freq=min_freq)
    ds = ChatDataset(tokenized, vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=collate_fn)
    return loader, vocab, tokenized


# ===========================================================================
# 自检
# ===========================================================================
if __name__ == "__main__":
    PATH = "xiaohuangji50w_nofenci.conv"          # 改成你下载的文件路径
    loader, vocab, pairs = build_dataloader(PATH, batch_size=4, sample=50000)

    print(f"清洗后句对数: {len(pairs)}")
    print(f"词表大小: {len(vocab)}")
    print("示例句对(已分词):", pairs[0])

    src, src_len, tgt, tgt_len = next(iter(loader))
    print("\nsrc batch shape:", src.shape, "  tgt batch shape:", tgt.shape)
    print("第 0 条 src ids:", src[0].tolist())
    print("第 0 条 tgt ids:", tgt[0].tolist())
    print("第 0 条 src 还原:", "".join(vocab.decode(src[0].tolist())))
    print("第 0 条 tgt 还原:", "".join(vocab.decode(tgt[0].tolist())))
