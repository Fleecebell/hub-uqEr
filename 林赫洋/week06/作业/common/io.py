# coding:utf8
"""预测结果落盘（全作业共用）

每一路方法都写同一个格式，后面 eval/eval_all.py 与 error_analysis.py
才能直接读、直接横向比：

{
  "method": "tfidf",
  "meta": {...},                # 样本口径（共多少条、去重多少、seed）
  "metrics": {...},             # common.metrics 算出来的
  "preds": [{"idx","label","pred","conf"}, ...]
}
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"

from common.env import ensure_deps  # noqa: E402

# 只要用到落盘/指标，就一定需要 numpy + sklearn —— 在这里统一检查，
# 免得每个入口脚本各写一遍（base 环境里 numpy 可能有、sklearn 一定没有）。
ensure_deps(["numpy", "sklearn"])

from common import metrics as M  # noqa: E402


def save_predictions(method, samples, preds, meta, probs=None, num_classes=15):
    """samples: 评测样本；preds: 与 samples 等长的类别 id（-1=未命中）；
    probs: 可选 [N,15] 概率，用来算 ECE。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y_true = [s["label"] for s in samples]
    rep = M.classification_report(y_true, preds, num_classes=num_classes)
    if probs is not None:
        rep["ece"] = M.expected_calibration_error(y_true, probs)

    rows = []
    for s, p, in zip(samples, preds):
        row = {"idx": s["idx"], "label": s["label"], "pred": int(p)}
        if probs is not None:
            row["conf"] = round(float(max(probs[len(rows)])), 4)
        rows.append(row)

    payload = {"method": method, "meta": meta, "metrics": rep, "preds": rows}
    path = OUT_DIR / f"pred_{method}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[{method}] acc={rep.get('acc')} macro_f1={rep.get('macro_f1')} "
          f"unparsed={rep.get('unparsed', 0)} → {path.name}")
    return payload


def load_all_predictions():
    """读 outputs 下所有 pred_*.json，返回 {method: payload}"""
    out = {}
    for p in sorted(OUT_DIR.glob("pred_*.json")):
        out[p.stem.replace("pred_", "")] = json.loads(p.read_text(encoding="utf-8"))
    return out
