# 基于 Seq2Seq 与 Transformer 的中文聊天机器人

《自然语言处理》综合实习 · 题目二

按 NLP 技术演进顺序，**从零手写**并对比三代生成式对话模型：
`Seq2Seq(LSTM) → Attention-Seq2Seq → Transformer`。
每一代都为解决上一代的缺陷而生——理解"动机"，而非调包。

---

## 技术链（每步解决上一步的缺陷）

```
词向量 → LSTM → Seq2Seq ──❗固定context瓶颈──→ Attention ──❗依赖RNN不能并行──→ Transformer
```

| 阶段 | 模型 | 解决的问题 | 关键技术 |
|---|---|---|---|
| 1 | Seq2Seq (LSTM) | —（基线） | Encoder-Decoder、Teacher forcing、自回归解码 |
| 2 | Attention-Seq2Seq | 固定 context 瓶颈 | Bahdanau 加性注意力、动态 context |
| 3 | Transformer | RNN 无法并行、长依赖弱 | 多头自注意力、位置编码、三种 mask、并行训练 |

---

## 文件结构

| 文件 | 说明 |
|---|---|
| [data_pipeline.py](data_pipeline.py) | 阶段0：`.conv` 解析 → 清洗(去emoji/短句) → 词表 → DataLoader（三模型共用） |
| [seq2seq.py](seq2seq.py) | 阶段1：Seq2Seq (LSTM) 基线 |
| [attn_seq2seq.py](attn_seq2seq.py) | 阶段2：Attention-Seq2Seq + 注意力热力图 |
| [transformer.py](transformer.py) | 阶段3：从零手写 Transformer + 交叉注意力热力图 |
| [实习报告.md](实习报告.md) | **完整实习报告**：思考过程、方法论、实现、实验对比、结论 |
| [学习路线.md](学习路线.md) | 分模块学习路线 + 一周阅读顺序 + 资源 |
| [题目二_聊天机器人_实习计划.md](题目二_聊天机器人_实习计划.md) | 实习计划、分工、交付清单、参考资料 |
| `Attention Is All You Need.pdf` | Transformer 原论文 |
| `xiaohuangji50w_nofenci.conv` | 数据集（需自行下载，见下） |

---

## 环境与数据

```bash
pip install torch jieba matplotlib
```

> ⚠️ 本机注意：`python` 曾被 `~/.zshrc` 里 `alias python='python3'` 劫持到 Homebrew Python（无依赖）。
> 统一用 conda 解释器：`/Users/wangjiaqiao/miniconda3/bin/python`，或删掉该 alias 后重开终端。

**数据集**：小黄鸡 50w 对话语料。下载 `xiaohuangji50w_nofenci.conv` 放到本目录：
- https://github.com/candlewill/Dialog_Corpus
- https://github.com/codemayq/chinese-chatbot-corpus

---

## 运行

每个阶段都支持 `train` / `chat` / （阶段2、3 还有 `heatmap`）：

```bash
# 阶段0：自检数据管线
python data_pipeline.py

# 阶段1：Seq2Seq 基线
python seq2seq.py train
python seq2seq.py chat              # 试长句，感受"固定 context 瓶颈"

# 阶段2：Attention-Seq2Seq
python attn_seq2seq.py train
python attn_seq2seq.py chat
python attn_seq2seq.py heatmap "你今天心情怎么样"   # -> attn_heatmap.png

# 阶段3：Transformer
python transformer.py train
python transformer.py chat
python transformer.py heatmap "你今天心情怎么样"    # -> tf_heatmap.png
```

模型权重与词表会存到各自的 `*.pt` / `*_vocab.pkl`（或 `vocab.pkl`）。

---

## 实验结果（统一配置：50k 样本 / 12 epoch / mps）

| 模型 | 最终 ppl | 长句连贯度 | 训练速度 | 可视化 |
|---|---|---|---|---|
| 阶段1 Seq2Seq | 20.4 | 差（安全回复+瓶颈） | 中 | — |
| 阶段2 Attn-Seq2Seq | _填_ | 改善 | 慢（逐步算注意力） | 编码-解码对齐热力图 |
| 阶段3 Transformer | _填_ | 最好 | 快（整句并行） | 交叉注意力热力图 |

> 现象记录：基线存在**安全回复**（老回"对对""你的"）和**固定 context 瓶颈**（长句崩坏），
> 这正是引入 Attention / Transformer 的动机。

---

## 三个核心知识点（答辩重点）

1. **固定 context 瓶颈**：Seq2Seq 把整句压成一个向量，长句信息丢失 → Attention 用动态 context 解决。
2. **三种 mask**：src-padding（屏蔽输入pad）/ look-ahead（解码器防偷看未来，上三角）/ tgt-padding。
3. **为什么 Transformer 能并行**：mask + teacher forcing 让整句一次算完，不依赖上一步输出 → 训练快、可扩展。

---

## 参考资料
见 [学习路线.md](学习路线.md) 与 [题目二_聊天机器人_实习计划.md](题目二_聊天机器人_实习计划.md) 的"参考资料"节
（d2l、李宏毅、Illustrated Transformer、The Annotated Transformer、CS224n、各阶段论文）。
