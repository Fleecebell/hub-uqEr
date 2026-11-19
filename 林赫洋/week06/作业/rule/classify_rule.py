# coding:utf8
"""
方法一：规则 / 关键词（L0，第 6 周 PPT 第 6 页那一级）

不训练、不加载任何模型：一张「类别 → 关键词」表，按优先级从上往下匹配，
第一个命中的词决定类别，全都没命中就返回 -1（"未知"）。

它的定位是**兜底和冷启动**：秒级、可解释、0 数据，
但 PPT 实测 F1 只有 0.07 —— 因为新闻标题的表述千变万化，规则一定覆盖不全。

用法：
    python rule/classify_rule.py                 # 500 条干净验证样本
    python rule/classify_rule.py --n 0           # 全部干净验证样本
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import io
from common.data import load_clean_val
from common.env import ensure_deps
from common.labels import LABELS, LABEL2ID

# 顺序 = 优先级：越具体的类放越前面（"证券"比"财经"具体，"军事"比"国际"具体）
RULES = [
    ("证券", ["券商", "证券", "基金", "期货", "A股", "港股", "美股", "债券", "指数", "涨停", "科创板"]),
    ("财经", ["股", "营收", "净利", "财报", "业绩", "市值", "融资", "IPO", "业绩预告", "亿元", "同比"]),
    ("军事", ["军事", "军演", "导弹", "航母", "战机", "武器", "军队", "国防", "阅兵"]),
    ("国际", ["美国", "特朗普", "联合国", "总统", "外交", "白宫", "欧盟", "俄罗斯", "日本", "英国"]),
    ("房产", ["房", "楼市", "地产", "房价", "物业", "楼盘", "租房", "土地"]),
    ("科技", ["芯片", "5G", "AI", "人工智能", "手机", "华为", "小米", "苹果", "互联网", "云计算", "大模型", "算法", "半导体"]),
    ("汽车", ["汽车", "车型", "新能源车", "特斯拉", "比亚迪", "发动机", "交付", "车展"]),
    ("教育", ["高考", "学校", "学生", "老师", "教育部", "考研", "招生", "校园"]),
    ("农业", ["农业", "种植", "农民", "水稻", "养殖", "生猪", "粮食", "丰收"]),
    ("旅游", ["旅游", "景区", "酒店", "自驾", "游客", "出行", "度假", "旅行社"]),
    ("电竞", ["电竞", "战队", "选手", "英雄联盟", "王者荣耀", "主播", "赛事", "游戏"]),
    ("体育", ["比赛", "冠军", "奥运", "联赛", "球员", "NBA", "足球", "篮球", "世界杯"]),
    ("娱乐", ["明星", "电影", "综艺", "演唱会", "电视剧", "票房", "导演", "演员"]),
    ("文化", ["文化", "书法", "文物", "非遗", "展览", "博物馆", "诗词", "传统"]),
    ("故事", ["故事", "回忆", "老人", "乡村", "小说", "情感", "父亲", "母亲", "那年"]),
]


def classify_one(text):
    """返回类别 id；没命中返回 -1"""
    for name, kws in RULES:
        for kw in kws:
            if kw in text:
                return LABEL2ID[name]
    return -1


def main():
    ensure_deps(["numpy"])  # 规则本身不用 numpy，但落盘指标要算，用它提示环境

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="评测多少条（0=全部干净验证集）")
    args = ap.parse_args()

    samples, meta = load_clean_val(n=args.n)
    preds = [classify_one(s["sentence"]) for s in samples]

    hit = sum(1 for p in preds if p >= 0)
    print(f"评测 {len(samples)} 条；规则命中 {hit} 条（{hit / len(samples):.1%}）")
    io.save_predictions("rule", samples, preds, {**meta, "method_note": "关键词规则，未命中=-1"})


if __name__ == "__main__":
    main()
