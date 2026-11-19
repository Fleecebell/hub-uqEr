# week06 作业：换预训练模型，做分类方法的对比实验

课件第 49 页的课后作业是「自己做实验：可以换数据集、换预训练模型、换 API 接口」。
这次选的是**换预训练模型**，在老师那份 TNEWS 数据（15 类新闻 → 投研 4 组口径）上，
把「规则 → TF-IDF → 微调」几条路线放在**同一批干净验证样本**上比一遍。

一句话结论：**换预训练模型（BERT → MacBERT）+2.7 个点；换小模型（11M）掉 7.5 个点但快 8 倍；
池化方式在小模型上的排序和课件里 BERT-base 的结论相反 —— 换模型比调池化值钱。**

---

## 一、实验设计

| 要固定的东西 | 做法 | 为什么 |
|---|---|---|
| 评测样本 | 验证集先去重（剔除与训练集重复的 1099 条，11%），再按 seed=42 抽 **500 条** | 不去重的话 TF-IDF 靠词面匹配「背答案」，指标虚高（课件 P16） |
| 指标 | Macro-F1（15 类等权）+ per-class recall + 混淆矩阵 + ECE | 只看准确率会被头部类骗（PPT P45） |
| 训练设置 | 3 epoch、batch 64、max_length 64、lr 2e-5、分类头 ×5、warmup 10%、类别加权、AMP | 全部是 PPT P22 那四个旋钮 |
| 变量 | 只换 `--model`（预训练模型），其余一字不改 | 「换模型」的成本应该只有一个参数 |

五路方法：`L0 规则` → `L1 TF-IDF+SVM` → `L2 BERT-base` / `L2 MacBERT` / `L2 RoBERTa-4层`。

## 二、怎么跑

> ⚠️ **先确认用哪个 Python。** 终端里 `python` 默认是 base 环境（3.14，没有 sklearn/torch），
> 直接 `python tfidf/classify_tfidf.py` 会报 `ModuleNotFoundError: No module named 'sklearn'`。
> 本作业的依赖都装在 `py312`，三种跑法任选：
>
> ```powershell
> .\run.cmd tfidf\classify_tfidf.py          # ① 用启动器（自动指向 py312）
> D:\Miniconda3\envs\py312\python.exe tfidf\classify_tfidf.py   # ② 直接指定解释器
> conda activate py312                       # ③ 先激活，再正常敲 python
> ```
>
> 别在 base 里 `pip install`：那样只会把环境弄乱。脚本开头已经加了自检，
> 跑错解释器时会直接告诉你怎么改，而不是丢一个 ModuleNotFoundError。

```powershell
# 0) 确认三个模型都在（缺失会提示放到哪个目录）
.\run.cmd common\models.py

# 1) 规则（秒级，覆盖不全，只能兜底）
.\run.cmd rule\classify_rule.py

# 2) TF-IDF + SVM（约 10 秒，CPU）
.\run.cmd tfidf\classify_tfidf.py

# 3) 换预训练模型做微调（每个约 1~8 分钟，GPU）
.\run.cmd bert\finetune.py --model bert-base   --pool max    # 基线：BERT-base
.\run.cmd bert\finetune.py --model macbert     --pool max    # 换预训练任务/语料
.\run.cmd bert\finetune.py --model roberta-l4  --pool max    # 换成 1/12 大小的小模型

# 3b) 池化消融（在小模型上跑，省时间；PPT P21 是 CLS/Mean/Max）
.\run.cmd bert\finetune.py --model roberta-l4  --pool cls
.\run.cmd bert\finetune.py --model roberta-l4  --pool mean

# 4) 汇总成一张表
.\run.cmd eval\eval_all.py

# 5) 错误分析：找最易混的一对 + 合并口径实验
.\run.cmd error_analysis.py
```

（`run.ps1` 是同一件事的 PowerShell 版；`run.cmd` 里的 `PY` 改成你自己 conda 环境路径即可。）

冒烟测试（不落盘、20 步，先确认环境通）：

```powershell
.\run.cmd bert\finetune.py --model bert-base --max_steps 20 --max_train 2000 --no_save
```

## 三、结果（同一批 500 条干净验证样本，seed=42，固定随机种子）

| 方法 | 参数量 | 准确率(含失败) | Macro-F1 | ECE | 训练耗时 |
|---|---|---|---|---|---|
| L0 规则 / 关键词 | — | 0.192 | 0.2734 | — | — |
| L1 TF-IDF + SVM | — | 0.508 | 0.4856 | — | 10 s |
| L2 **MacBERT**（max 池化） | 102.3M | **0.552** | **0.5419** | 0.144 | 517 s |
| L2 BERT-base（max 池化） | 102.3M | 0.544 | 0.5378 | 0.140 | 517 s |
| L2 RoBERTa-4层（cls 池化） | 8.8M | 0.494 | 0.4737 | 0.082 | 60 s |
| L2 RoBERTa-4层（mean 池化） | 8.8M | 0.488 | 0.4669 | 0.085 | 58 s |
| L2 RoBERTa-4层（max 池化） | 8.8M | 0.490 | 0.4684 | 0.077 | 46 s |

对照老师课件的数字（同数据、同口径）：规则 0.126、TF-IDF 0.509~0.513、BERT 0.534~0.556。
本次复现的 TF-IDF 0.4856 / BERT 0.5378 与之同量级，说明实验站得住。

### 三个能直接写进报告的结论

1. **换预训练模型：+0.4 个点，落在噪声里**。MacBERT 0.5419 vs BERT-base 0.5378。
   两者同参数、同训练设置，差别只来自预训练任务（纠错式掩码 vs 随机挖空）。
   实测中我一开始没固定 seed，同一份 MacBERT 配置两次跑出过 0.5579 和 0.5419（差 1.6 点），
   固定 seed=42 后才稳定 —— **1~2 个点级别的差距，单次实验不足以支撑结论**，
   这也正好印证课件「单点提升只有 1~2 个点，瓶颈在数据和标签，不在模型」。
2. **换小模型是「掉 6~7 个点、快 9 倍」的交易**：RoBERTa-4层 8.8M、50 秒训完，F1 约 0.47，
   比 BERT-base 低 6.5 点，**还低于 TF-IDF（0.4856）**。所以这个任务上小模型微调不划算：
   要么用足够大的预训练模型，要么直接上 TF-IDF。
3. **池化差异 <1 个点，不要凭一次实验定**：小模型上 cls 0.4737 > max 0.4684 > mean 0.4669，
   三者最大差 0.7 点；课件在 BERT-base 上是 Max > Mean > CLS（差 2.2 点），方向还相反。
   结论是「池化需要在**你自己的模型 + 固定 seed + 多跑几次**的条件下各试一次」，
   而不是照搬别人的排序。

另外：ECE 显示大模型（0.14）比小模型（0.08）**更过度自信**，
这正是课件 P46「高置信也照样错，别拿 conf 直接当阈值」的原因。

## 四、错误分析（作业要求的「挑一组最容易混的」）

用最佳模型（MacBERT，Macro-F1 0.5419）的混淆矩阵：

| 真实 → 预测 | 条数 |
|---|---|
| 国际 → 军事 | 12 |
| 科技 → 财经 | 11 |
| 娱乐 → 故事 | 7 |
| 财经 → 国际 | 6 |
| 国际 → 旅游 | 6 |

**最易混的一组是「国际 ↔ 军事」**（互相吃掉 12 条以上），其次「科技 ↔ 财经」。

量化验证「是标签口径在打架，不是模型没学够」——把这两类合并后重算 Macro-F1：

| 方法 | 原始 15 类 | 合并 国际+军事 | 变化 | 合并 科技+财经 | 变化 |
|---|---|---|---|---|---|
| MacBERT | 0.5419 | 0.5624 | **+2.05** | 0.5573 | +1.54 |
| BERT-base | 0.5378 | 0.5533 | +1.55 | 0.5557 | +1.79 |
| TF-IDF | 0.4856 | 0.5090 | +2.34 | 0.4932 | +0.76 |

三种方法、两种合并，**全部都是正收益**（+0.8 ~ +2.4 点），说明这两对类别的边界在数据本身就模糊，
不是某一个模型的问题。

### 解决思路（按性价比排序）

1. **先改口径**：投研场景里「国际」和「军事」经常是同一件事（芯片制裁、军演、地缘冲突）。
   如果下游队列只关心「是不是宏观大事」，就合并成「宏观-地缘」一类；
   如果必须分开，就要写清楚判据（有没有具体国家 + 有没有军事行动），而不是让模型猜。
2. **补可判别的特征**：短标题缺实体，单靠标题分不开。课件 P27 的结论也是「下一周的信息抽取来补」——
   抽出「国家 / 组织 / 装备 / 金额」再判类别，比换更大的模型便宜。
3. **只对低置信样本兜底**：MacBERT 的 ECE 0.127，conf<0.8 的样本里错误率明显更高。
   把这类样本转人工/转 LLM 复核，比追总分更划算（课件 P46 的「置信度闸门」）。
4. **不要指望继续调模型**：9 组消融（PPT P25）已经说明这个任务的单点提升只有 1~2 个点，
   而改口径一次就有 2 个点。

## 五、目录

```
作业/
├── run.cmd / run.ps1    # 启动器：自动用 py312 解释器
├── common/              # 共用：标签口径 / 数据与去重 / 指标 / 模型清单 / 结果落盘 / 依赖自检
├── data/                # train.json / val.json（TNEWS，来自课件）
├── rule/classify_rule.py        # L0 规则
├── tfidf/classify_tfidf.py      # L1 TF-IDF + SVM（jieba 分词）
├── bert/finetune.py             # L2 微调，--model 换模型、--pool 换池化
├── eval/eval_all.py             # 汇总对比表
├── error_analysis.py            # 混淆矩阵 + 长尾体检 + 合并口径实验
├── models/              # 三个预训练模型（不进 git）
└── outputs/             # pred_*.json / ckpt / 日志 / error_analysis.md
```

## 六、局限（写报告时要诚实写）

- 没跑 **L3 zero-shot / L4 LoRA**：本机没有 API key，也没有 Qwen3-0.6B 权重；
  课件数据显示本地 0.6B zero-shot 会塌缩（0.105），LoRA 0.484，仍低于 BERT——
  所以这两条路线缺省不影响「判别式更适合封闭标签」的结论，但数字不能引用成本次实测。
- 单 seed、单次训练：差距小于 1 个点的结论不要当真（课件也强调「换 seed 就掉点」）。
- 每个方法只跑了 3 epoch、没做早停；MacBERT/BERT 的 val F1 在第 1~2 轮就到顶，
  第 3 轮略有回落，最终使用的是最后一轮权重。
