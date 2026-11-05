"""cnfont.py —— 给 matplotlib 配中文字体。

Windows 上 matplotlib 默认用 DejaVu Sans，没有中文，标题会变成方框。
微软雅黑（Microsoft YaHei）系统自带，直接用它。

用法：
    from cnfont import setup_cn_font
    setup_cn_font()
"""
import os

import matplotlib


def setup_cn_font(verbose=False):
    """把 matplotlib 的中文字体设为系统自带字体。"""
    import matplotlib.font_manager as fm

    candidates = [
        "Microsoft YaHei",   # 微软雅黑，Windows 自带
        "SimHei",            # 黑体
        "SimSun",            # 宋体
        "DengXian",          # 等线
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen is None:
        if verbose:
            print("警告：没找到中文字体，标题会显示为方框")
        return None

    matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    if verbose:
        print("中文字体: %s" % chosen)
    return chosen


if __name__ == "__main__":
    setup_cn_font(verbose=True)
