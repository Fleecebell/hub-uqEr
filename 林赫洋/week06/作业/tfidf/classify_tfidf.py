# coding:utf8
"""
方法二：TF-IDF + 线性分类器（L1，第 6 周 PPT 第 17 页）

四步就是全部：
    分词（jieba） → 建词表（df 过滤） → 算 TF×IDF → LinearSVC

不加任何预训练模型，秒级训练，CPU 就够。PPT 的结论是：
    它是「数据封闭」场景的强基线（内网 / 无权重 / 可解释 / 要秒级重训），
    但在通用短文本上落后于 BERT，而且数据越少差距越大。

用法：
    python tfidf/classify_tfidf.py
    python tfidf/classify_tfidf.py --n 0            # 全部干净验证集
    python tfidf/classify_tfidf.py --ngram 1        # 关掉 2-gram
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import io
from common.data import load_clean_val, load_splits
from common.env import ensure_deps


def tokenize(text):
    """jieba 分词；中文必须先分词，否则词表全是长串、泛化很差"""
    import jieba

    return [w for w in jieba.cut(text.strip()) if w.strip()]


def main():
    ensure_deps(["sklearn", "jieba"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ngram", type=int, default=2, help="最大 n-gram（1=只用单词）")
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC 的正则强度倒数")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    train, _ = load_splits()
    samples, meta = load_clean_val(n=args.n)

    t0 = time.time()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,        # 用自定义 tokenizer 时必须关掉默认正则
            ngram_range=(1, args.ngram),
            min_df=args.min_df,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(C=args.C, class_weight="balanced")),
    ])
    pipe.fit([d["sentence"] for d in train], [d["label"] for d in train])
    train_s = time.time() - t0

    t1 = time.time()
    preds = pipe.predict([s["sentence"] for s in samples]).tolist()
    infer_s = time.time() - t1

    n_feat = len(pipe.named_steps["tfidf"].vocabulary_)
    print(f"词表 {n_feat:,} 维；训练 {train_s:.1f}s；推理 {len(samples)} 条 {infer_s:.2f}s")
    io.save_predictions("tfidf", samples, preds,
                        {**meta, "n_features": n_feat,
                         "train_seconds": round(train_s, 1),
                         "ngram": args.ngram, "min_df": args.min_df, "C": args.C})


if __name__ == "__main__":
    main()
