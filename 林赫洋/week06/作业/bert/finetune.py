# coding:utf8
"""
方法三：微调预训练模型（L2）——本作业的实验主体

作业要求「换预训练模型做对比」，所以这一个脚本吃 `--model` 参数，
换模型只改一个词（PPT 第 20 页：架构只改池化和损失，BERT 本体不动）：

    python bert/finetune.py --model bert-base   --pool max    # 基线
    python bert/finetune.py --model macbert     --pool max    # 换预训练任务/语料
    python bert/finetune.py --model roberta-l4  --pool max    # 换成 1/10 大小的小模型

四个训练旋钮（都来自 PPT 第 22 页「三个旋钮比换模型更值钱」）：
    ① 分层学习率：BERT 层 2e-5，分类头 ×5（头是随机初始化的，要学得快）
    ② Warmup + 线性衰减：前 10% 步数升温，直接大 lr 会冲掉预训练知识
    ③ 类别加权 balanced：23 倍不均衡，不加权时尾部类 recall 趋近 0
    ④ AMP 混合精度：显存约降 40%，速度更快

以及池化（PPT 第 21 页）：CLS / Mean / Max 三选一，注意 Mean/Max 必须 mask 掉 PAD。

用法（冒烟测试不落盘、只跑 30 步）：
    python bert/finetune.py --model bert-base --max_steps 30 --no_save
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 先检查解释器：base 环境里没有 torch，直接 import 会抛一堆看不懂的错
from common.env import ensure_deps  # noqa: E402

ensure_deps(["torch", "transformers", "sklearn"])

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common import io  # noqa: E402
from common.data import load_clean_val, load_splits  # noqa: E402
from common.labels import LABELS  # noqa: E402
from common.models import model_dir  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"


class TnewsDataset(Dataset):
    def __init__(self, data, tok, max_length=64):
        self.items = []
        texts = [d["sentence"] for d in data]
        # 一次性批量 tokenize：比在 __getitem__ 里逐条 tokenize 快很多
        for i in range(0, len(texts), 1000):
            enc = tok(texts[i:i + 1000], truncation=True, max_length=max_length,
                      padding="max_length", return_tensors="pt")
            for j in range(enc["input_ids"].size(0)):
                self.items.append((enc["input_ids"][j], enc["attention_mask"][j], data[i + j]["label"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, mask, label = self.items[i]
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor(label)}


class PooledClassifier(nn.Module):
    """预训练模型 + 池化 + 线性头。池化前一定要 mask 掉 PAD。"""

    def __init__(self, path, num_labels=15, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(str(path))
        self.pool = pool
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:  # max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds, probs, ys = [], [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1).cpu()
        probs.append(p)
        preds.extend(p.argmax(-1).tolist())
        ys.extend(b["labels"].tolist())
    return preds, torch.cat(probs).numpy(), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base", help="bert-base / macbert / roberta-l4")
    ap.add_argument("--pool", default="max", choices=["cls", "mean", "max"], help="PPT 第 21 页三种池化")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="预训练层 lr（PPT：3e-5 以内）")
    ap.add_argument("--head_mult", type=float, default=5.0, help="分类头 lr = 预训练层 lr × 该倍数")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="每 epoch 最多多少步（冒烟用）")
    ap.add_argument("--max_train", type=int, default=0, help="只用前 N 条训练（做数据量实验用）")
    ap.add_argument("--eval_n", type=int, default=500, help="评测用的干净验证样本数")
    ap.add_argument("--seed", type=int, default=42, help="固定随机种子，否则同一配置每次差 1 个点左右")
    ap.add_argument("--no_save", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    # 固定种子：不固定的话，同一配置重跑一次就会差 1 个点左右，
    # 「谁比谁高 0.8 个点」这种结论根本站不住（PPT 也强调过换 seed 掉点）
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = model_dir(args.model)
    tag = f"{args.model}_{args.pool}"

    train, _ = load_splits()
    if args.max_train:
        # 分层抽样，保证类别比例不变（PPT 第 18 页数据量实验的做法）
        by_label = {}
        for d in train:
            by_label.setdefault(d["label"], []).append(d)
        train = [d for v in by_label.values() for d in v[: max(1, args.max_train // len(by_label))]]

    samples, meta = load_clean_val(n=args.eval_n)

    tok = AutoTokenizer.from_pretrained(str(path))
    train_ds = TnewsDataset(train, tok, args.max_length)
    eval_ds = TnewsDataset(samples, tok, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size * 2)

    model = PooledClassifier(path, num_labels=len(LABELS), pool=args.pool).to(device)
    n_param = sum(p.numel() for p in model.parameters())

    # 分层学习率
    head = list(model.classifier.parameters()) + list(model.dropout.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": body, "lr": args.lr},
        {"params": head, "lr": args.lr * args.head_mult},
    ], weight_decay=0.01)

    # 类别加权
    if args.no_class_weight:
        weight = None
    else:
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight

        cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                                  y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"[cfg] model={args.model} pool={args.pool} params={n_param / 1e6:.1f}M "
          f"train={len(train)} eval={len(samples)} steps/epoch={steps_per_epoch} "
          f"cw={'off' if weight is None else 'on'}", flush=True)

    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for i, b in enumerate(train_dl):
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            y = b["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n += 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n >= args.max_steps:
                break

        preds, probs, _ = run_eval(model, eval_dl, device)
        from sklearn.metrics import f1_score
        f1 = f1_score([s["label"] for s in samples], preds, average="macro")
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n), 4),
                    "val_macro_f1": round(float(f1), 4), "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: val_macro_f1={f1:.4f} 用时 {log[-1]['elapsed_s']}s", flush=True)

    # 最终评测（用最后一轮权重，与「数据量少时 val 一路涨」的实际情况一致）
    preds, probs, _ = run_eval(model, eval_dl, device)
    if args.no_save:
        # 冒烟模式：只报指标，不写 pred_*.json —— 否则几条 step 的垃圾结果
        # 会覆盖掉正式实验的预测文件，汇总表就乱了
        from sklearn.metrics import f1_score as _f1
        print(f"[smoke] macro_f1={_f1([s['label'] for s in samples], preds, average='macro'):.4f} "
              f"（--no_save：不写预测文件、不存 ckpt）")
        return

    io.save_predictions(tag, samples, preds,
                        {**meta, "model": args.model, "pool": args.pool,
                         "params_m": round(n_param / 1e6, 1),
                         "epochs": args.epochs, "lr": args.lr,
                         "class_weight": weight is not None,
                         "max_length": args.max_length,
                         "train_seconds": round(sum(x["elapsed_s"] for x in log), 1)},
                        probs=probs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "pool": args.pool, "model": args.model},
               OUT_DIR / f"ckpt_{tag}.pt")
    (OUT_DIR / f"log_{tag}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"ckpt + log → outputs/ckpt_{tag}.pt")


if __name__ == "__main__":
    main()
