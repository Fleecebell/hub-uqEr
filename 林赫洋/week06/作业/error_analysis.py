# coding:utf8
"""
错误分析：找最容易混的一对，并给一个能量化的修法

对应第 6 周 PPT 第 27 页（「这节课的课后作业就是：挑一组最容易混的，写清楚你打算怎么解决」）
和第 46 页（长尾 / 域偏移 / 过度自信三个坑）。

做三件事：
  1. 混淆矩阵：把对角去掉，排出最常互错的类别对
  2. 长尾体检：每类的样本数 + recall，看哪类被「吃掉」
  3. 口径实验：把最易混的两类合并成一类，重新算 14 类的 Macro-F1
     —— PPT 说「难样本来自标签口径重叠」时，这就是验证它的最小实验：
        如果合并后两类各自的 F1 明显回升、总 F1 也涨，说明是口径在打架，
        不该继续调模型，而该改标签定义。

用法：
    python error_analysis.py                       # 用 Macro-F1 最好的那一路
    python error_analysis.py --method macbert_max
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.env import ensure_deps
from common.io import load_all_predictions
from common.labels import LABELS

BASE_DIR = Path(__file__).resolve().parent   # 本脚本在 作业/ 根目录
OUT_DIR = BASE_DIR / "outputs"


def top_confusions(cm, top=6):
    pairs = []
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            if i != j and v > 0:
                pairs.append((v, LABELS[i], LABELS[j]))
    pairs.sort(reverse=True)
    return pairs[:top]


def merge_two_classes(rows, a_name, b_name):
    """把 a、b 两类合并后重新算 Macro-F1（只用于口径实验）"""
    a, b = LABELS.index(a_name), LABELS.index(b_name)
    kept = [c for c in range(len(LABELS)) if c != b]

    def remap(x):
        if x == b:
            return a
        return x

    y_true = [remap(r["label"]) for r in rows]
    y_pred = [remap(r["pred"]) for r in rows]

    from sklearn.metrics import f1_score
    return round(float(f1_score(y_true, y_pred, labels=kept, average="macro")), 4)


def main():
    ensure_deps(["numpy", "sklearn"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="", help="用哪一路的预测做分析（默认取 F1 最高）")
    args = ap.parse_args()

    results = load_all_predictions()
    if not results:
        print("outputs/ 下没有 pred_*.json，先跑各路方法")
        return
    if args.method:
        name = args.method
        payload = results[name]
    else:
        name, payload = max(results.items(), key=lambda kv: kv[1]["metrics"].get("macro_f1", 0))

    m = payload["metrics"]
    cm = m["confusion_matrix"]
    rows = payload["preds"]
    print(f"分析对象：{name}  Macro-F1={m.get('macro_f1')}\n")

    # 1) 最易混
    pairs = top_confusions(cm)
    print("最容易互错的前 6 组（真实 → 预测，样本数）：")
    for v, a, b in pairs:
        print(f"  {a} → {b}：{v} 条")

    # 2) 长尾体检
    counts = Counter(r["label"] for r in rows)
    print("\n长尾体检（评测集里每类的条数 / recall）：")
    rec = m.get("per_class_recall", {})
    for c in sorted(rec, key=lambda k: (counts[int(k)],)):
        print(f"  {LABELS[int(c)]:<4} n={counts[int(c)]:<3} recall={rec[c]}")

    # 3) 口径实验：合并最易混的一对
    if pairs:
        v, a, b = pairs[0]
        merged = merge_two_classes(rows, a, b)
        before = m.get("macro_f1")
        print(f"\n口径实验：把「{a}」和「{b}」合并成一类（15 → 14 类）")
        print(f"  合并前 15 类 Macro-F1 = {before}")
        print(f"  合并后 14 类 Macro-F1 = {merged}")
        delta = round((merged or 0) - (before or 0), 4)
        print(f"  变化 {delta:+.4f}")
        print("  读法：如果合并后明显回升，说明这两类的边界在业务上本来就模糊（PPT 第 27 页："
              "科技新闻里常带财报信息），下一步该改标签口径/补实体特征，而不是继续换模型。")

    # 落盘一份 markdown，方便贴进报告
    lines = [f"# 错误分析：{name}", "", f"- 评测样本：{m.get('n')} 条",
             f"- 准确率：{m.get('acc')}　Macro-F1：{m.get('macro_f1')}", "", "## 最易混的类别对", ""]
    for v, a, b in pairs:
        lines.append(f"- {a} → {b}：{v} 条")
    lines += ["", "## 长尾体检", "", "| 类别 | 条数 | recall |", "|---|---|---|"]
    for c in sorted(rec, key=lambda k: counts[int(k)]):
        lines.append(f"| {LABELS[int(c)]} | {counts[int(c)]} | {rec[c]} |")
    if pairs:
        lines += ["", "## 口径实验（合并最易混的一对）", "",
                  f"合并「{a}」+「{b}」：15 类 Macro-F1 {m.get('macro_f1')} → 14 类 {merged}"]
    (OUT_DIR / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写出 {OUT_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
