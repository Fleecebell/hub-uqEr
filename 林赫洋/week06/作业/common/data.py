# coding:utf8
"""
数据读取与「干净验证集」（全作业共用）

第 6 周 PPT 第 16 页讲的坑：这份 TNEWS 的验证集里有 1099/10000 条
和训练集**完全重复**（11%）。靠词面匹配的 TF-IDF 会「背答案」，
指标被虚高（全量 val F1 0.541 → 去重后 0.509）。

所以作业里所有方法都在同一批「去重后的验证样本」上评测：
  - 先去重（整句完全相同算重复）
  - 再用固定 seed 抽 500 条
同一批样本是「横向比较」的前提，否则差 2 个点可能只是抽样波动。
"""

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _load(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_splits():
    """返回 (train, val)，每条形如 {"idx", "sentence", "label"}"""
    return _load("train.json"), _load("val.json")


def clean_val(train=None, val=None):
    """去掉与训练集整句重复的验证样本，返回 (干净列表, 重复条数)"""
    if train is None or val is None:
        train, val = load_splits()
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    return clean, len(val) - len(clean)


def sample_subset(items, n, seed=42):
    """固定 seed 抽样：各路方法必须用同一批样本"""
    if not n or n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)


def load_clean_val(n=500, seed=42):
    """最常用入口：读数据 → 去重 → 抽样，返回 (samples, meta)"""
    train, val = load_splits()
    clean, n_dup = clean_val(train, val)
    samples = sample_subset(clean, n, seed)
    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "n_dup": n_dup,
        "dup_rate": round(n_dup / len(val), 4) if val else 0.0,
        "n_clean": len(clean),
        "n_eval": len(samples),
        "seed": seed,
    }
    return samples, meta
