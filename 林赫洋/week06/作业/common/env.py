# coding:utf8
"""依赖自检：跑错解释器时给出能直接照抄的解决办法

坑：cmd / PowerShell 里敲 `python` 默认是 base 环境（3.14），
而本作业的依赖（sklearn / jieba / torch / transformers）都装在 py312 里。
报错 `ModuleNotFoundError: No module named 'sklearn'` 时，人往往会去 pip install，
结果装进 base，越弄越乱。所以这里在脚本开头先检查，并打印正确的命令。
"""

import os
import sys

PY312 = r"D:\Miniconda3\envs\py312\python.exe"


def ensure_deps(modules):
    """检查当前解释器是否装了这些模块；缺了就打印指引并退出。"""
    missing = []
    for m in modules:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if not missing:
        return

    cur = sys.executable
    lines = [
        "",
        "=" * 66,
        f"缺依赖：{', '.join(missing)}",
        f"当前解释器：{cur}（Python {sys.version.split()[0]}）",
    ]
    if os.path.isfile(PY312) and os.path.normcase(cur) != os.path.normcase(PY312):
        lines += [
            "",
            "这个作业的依赖装在 py312 环境里，用下面任意一种方式跑：",
            f"  1) 直接指定解释器：  {PY312} <脚本> <参数>",
            "  2) 先激活环境：      conda activate py312",
            "                      python <脚本> <参数>",
            "",
            "不要在 base 里 pip install —— base 是 conda 自己的运行时，装进去容易冲突。",
        ]
    else:
        lines += [
            "",
            "当前环境就是 py312，但依赖不全，安装：",
            f"  {PY312} -m pip install scikit-learn jieba torch transformers tqdm",
        ]
    lines.append("=" * 66)
    print("\n".join(lines))
    sys.exit(1)
