# coding:utf8
"""
标签口径（全作业共用）

第 6 周 PPT 第 12 页：TNEWS 的 15 个新闻类别不是投研关心的东西，
先用一张映射表把它们收拢成 4 组主题，再由主题派生「相关性」——
标签体系是业务定义，写在代码里，换领域只改这张表。

15 类的顺序必须和 TNEWS 官方 label_map 一致，否则所有方法口径都错。
"""

LABELS = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# 15 类 → 4 组投研主题
GROUP_MAP = {
    "财经": "金融", "证券": "金融", "房产": "金融",
    "科技": "产业", "汽车": "产业", "农业": "产业",
    "国际": "宏观", "军事": "宏观",
    "教育": "噪音", "娱乐": "噪音", "体育": "噪音",
    "电竞": "噪音", "文化": "噪音", "故事": "噪音", "旅游": "噪音",
}
GROUPS = ["金融", "产业", "宏观", "噪音"]

# 相关性：金融 + 产业 = 相关；宏观 = 弱相关（算不相关）；噪音 = 不相关
RELEVANT_GROUPS = {"金融", "产业"}


def to_group(label_id):
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id):
    return to_group(label_id) in RELEVANT_GROUPS


def group_of_pred(pred_id):
    """预测值 -1（解析失败 / 规则没命中）时返回 '未知'"""
    if pred_id is None or pred_id < 0:
        return "未知"
    return to_group(pred_id)
