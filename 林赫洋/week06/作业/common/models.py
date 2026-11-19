# coding:utf8
"""
预训练模型清单（本作业的实验变量）

作业要求「换预训练模型做对比」，所以这里把三个模型登记成同一张表：
    名字（命令行用）      本地目录                结构              参数量
    bert-base            bert-base-chinese       12 层 / 768       约 102M   ← 老师的基线
    macbert              macbert-base            12 层 / 768       约 102M   ← 换预训练任务/语料
    roberta-l4           roberta-l4-h256          4 层 / 256        约 11M    ← 故意换成小模型

三者都能被 transformers 的 AutoModel/AutoTokenizer 直接加载，
所以 bert/finetune.py 里一行代码就能换模型 —— 这正是「换模型」这件事该有的成本。
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

REGISTRY = {
    "bert-base": {
        "dir": "bert-base-chinese",
        "desc": "BERT-base 中文（12 层 / 768 维 / 约 102M）",
        "url": "local: week04/作业/bert-base-chinese",
    },
    "macbert": {
        "dir": "macbert-base",
        "desc": "MacBERT-base 中文（12 层 / 768 维 / 约 102M，MLM 换成纠错式掩码）",
        "url": "hf-mirror: hfl/chinese-macbert-base",
    },
    "roberta-l4": {
        "dir": "roberta-l4-h256",
        "desc": "RoBERTa 小模型（4 层 / 256 维 / 约 11M，故意用来对比容量）",
        "url": "hf-mirror: uer/chinese_roberta_L-4_H-256",
    },
}


def _has_weights(p: Path) -> bool:
    if not p.is_dir():
        return False
    if any(p.glob("*.safetensors")):
        return True
    return (p / "pytorch_model.bin").exists()


def model_dir(name: str) -> Path:
    """名字 → 本地目录；缺文件就明确报错，不做静默联网回退。"""
    if name not in REGISTRY:
        raise KeyError(f"未知模型 '{name}'，可选：{', '.join(REGISTRY)}")
    p = MODELS_DIR / REGISTRY[name]["dir"]
    if not _has_weights(p):
        raise FileNotFoundError(
            f"本地缺少模型 {name} → {p}\n"
            f"来源：{REGISTRY[name]['url']}\n"
            f"把它放到 models/{REGISTRY[name]['dir']}/ 下（config.json + 权重 + vocab.txt）"
        )
    return p


def list_models():
    for name, info in REGISTRY.items():
        ok = "OK " if _has_weights(MODELS_DIR / info["dir"]) else "缺失"
        print(f"  {ok} {name:<12} {info['desc']}")


if __name__ == "__main__":
    list_models()
