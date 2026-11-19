# coding:utf8
"""
把各路结果汇总成一张对比表（同一批 500 条干净验证样本）

用法：
    python eval/eval_all.py
    python eval/eval_all.py --sort macro_f1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.env import ensure_deps  # noqa: E402

ensure_deps(["numpy"])

from common.io import load_all_predictions  # noqa: E402
from common.labels import LABELS  # noqa: E402

DISPLAY = {
    "rule": "L0 规则 / 关键词",
    "tfidf": "L1 TF-IDF + SVM",
    "bert-base_max": "L2 BERT-base (max池化)",
    "macbert_max": "L2 MacBERT (max池化)",
    "roberta-l4_max": "L2 RoBERTa-4层 (max池化)",
    "roberta-l4_cls": "L2 RoBERTa-4层 (cls池化)",
    "roberta-l4_mean": "L2 RoBERTa-4层 (mean池化)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="macro_f1")
    args = ap.parse_args()

    results = load_all_predictions()
    if not results:
        print("outputs/ 下还没有 pred_*.json，先跑 rule / tfidf / bert 的脚本")
        return

    rows = sorted(results.items(), key=lambda kv: kv[1]["metrics"].get(args.sort, 0) or 0, reverse=True)

    print(f"{'方法':<26}{'准确率(含失败)':>14}{'Macro-F1':>10}{'失败':>6}{'ECE':>8}{'耗时':>10}")
    print("-" * 78)
    for name, payload in rows:
        m = payload["metrics"]
        meta = payload.get("meta", {})
        sec = meta.get("train_seconds")
        sec_s = f"{sec:.0f}s" if isinstance(sec, (int, float)) else "-"
        print(f"{DISPLAY.get(name, name):<26}{m.get('acc_incl_unparsed', '-'):>14}"
              f"{m.get('macro_f1', '-'):>10}{m.get('unparsed', 0):>6}{m.get('ece', '-'):>8}{sec_s:>10}")

    print("\n每个方法在多少类上有非零预测（覆盖度）：")
    for name, payload in rows:
        used = len({p["pred"] for p in payload["preds"] if p["pred"] >= 0})
        print(f"  {DISPLAY.get(name, name):<26} {used}/15")

    best = rows[0]
    print(f"\n最佳：{DISPLAY.get(best[0], best[0])}  Macro-F1={best[1]['metrics'].get('macro_f1')}")
    print("\n按类别 F1（只列表现最差的 5 类）：")
    per = best[1]["metrics"].get("per_class_f1", {})
    for c, v in sorted(per.items(), key=lambda kv: kv[1])[:5]:
        print(f"  {LABELS[int(c)]:<4} {v}")


if __name__ == "__main__":
    main()
