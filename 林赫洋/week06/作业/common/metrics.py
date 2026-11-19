# coding:utf8
"""
统一评测指标（全作业共用）

第 6 周 PPT 第 45 页：主指标 Macro-F1（15 类等权），辅以 per-class recall、
混淆矩阵、ECE（置信度校准）。

一个关键口径：生成式 / 规则类方法可能「没给出答案」（规则未命中、解析失败，记 -1）。
把它从分母里剔掉会让指标虚高，所以这里统一按**答错**处理：
    - acc_incl_unparsed = 答对条数 / 全部条数（对外张榜用这个）
    - macro_f1          = 把「没答」当作该类的一次漏判，逐类算 P/R/F1 再平均
    - macro_f1_parsed   = 只在答出来的样本上算（内部诊断用，别拿去和别人比）
"""

import numpy as np


def classification_report(y_true, y_pred, num_classes=15):
    """返回统一口径的指标字典；y_pred 里 -1 表示解析失败/规则未命中。"""
    from sklearn.metrics import confusion_matrix, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    valid = y_pred >= 0
    n_unparsed = int((~valid).sum())

    rep = {
        "n": n,
        "unparsed": n_unparsed,
        "unparsed_rate": round(n_unparsed / n, 4) if n else 0.0,
    }
    if valid.sum() == 0:
        return rep

    yt, yp = y_true[valid], y_pred[valid]
    rep["acc"] = round(float((yt == yp).mean()), 4)                 # 只在答出来的样本上
    rep["acc_incl_unparsed"] = round(float((yt == yp).sum() / n), 4)  # 全部样本，答错算错

    # —— 严格口径的 per-class P/R/F1：把「没答」算成该类漏判 ——
    cm_full = np.zeros((num_classes, num_classes + 1), dtype=int)  # 最后一列 = 没答
    for t, p in zip(y_true, y_pred):
        if t < 0 or t >= num_classes:
            continue
        cm_full[t, num_classes if p < 0 else p] += 1

    per_f1, per_recall = {}, {}
    for c in range(num_classes):
        tp = cm_full[c, c]
        support = cm_full[c].sum()
        pred_pos = cm_full[:num_classes, c].sum()
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_f1[c] = round(float(f1), 4)
        per_recall[c] = round(float(recall), 4) if support else None
    rep["macro_f1"] = round(float(np.mean(list(per_f1.values()))), 4)
    rep["per_class_f1"] = per_f1
    rep["per_class_recall"] = per_recall
    rep["macro_f1_parsed"] = round(float(f1_score(yt, yp, average="macro")), 4)
    rep["confusion_matrix"] = confusion_matrix(yt, yp, labels=list(range(num_classes))).tolist()
    return rep


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE：模型说「0.9」时，实际命中率是不是 0.9。越小越可信。"""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / len(conf)
    return round(float(ece), 4)


def to_markdown_table(results):
    """{方法名: 指标字典} → Markdown 表"""
    lines = ["| 方法 | 准确率(含失败) | Macro-F1 | 解析失败 |", "|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r.get('acc_incl_unparsed', '-')} | {r.get('macro_f1', '-')} | {r.get('unparsed', 0)} |")
    return "\n".join(lines)
