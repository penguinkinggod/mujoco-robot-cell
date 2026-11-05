"""find_desktop_assets.py —— 从 2595 个 SimReady 资产里挑出能放在机器人桌上的。

为什么需要这个
--------------
SimReady 里全是仓储/工厂/医疗的大件（叉车、货架、X光机）。
SO-101 只有 60 cm 臂展、600 g 负载，只能用桌面尺度的零件。
手动一个个试会浪费很多时间，直接用 manifest 里的 bounds 过滤。

筛选标准（针对 SO-101 + 车床场景）：
    最长边      < 0.25 m    （能放进工作台）
    最短边      > 0.02 m    （太小了抓不住）
    质量（若有）< 0.6 kg    （SO-101 安全负载）

跑法：
    D:\\robot-env\\Scripts\\python.exe find_desktop_assets.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
MANIFEST = os.path.join(ROOT, "models", "simready", "manifest.jsonl")

rows = []
with open(MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print("=" * 80)
print("SimReady 资产筛选（共 %d 个）" % len(rows))
print("=" * 80)
print()

# 先看尺寸分布，找"桌面尺度"在哪里
dims = []
for r in rows:
    b = r.get("bounds")
    if not b:
        continue
    sz = [b["max"][i] - b["min"][i] for i in range(3)]
    dims.append((max(sz), r["asset"], sz))

dims.sort()
print("尺寸分布（按最长边）：")
print("  最小 : %.3f m  (%s)" % (dims[0][0], dims[0][1]))
print("  1/4  : %.3f m" % dims[len(dims) // 4][0])
print("  中位 : %.3f m  (%s)" % (dims[len(dims) // 2][0], dims[len(dims) // 2][1]))
print("  3/4  : %.3f m" % dims[3 * len(dims) // 4][0])
print("  最大 : %.3f m  (%s)" % (dims[-1][0], dims[-1][1]))
print()

# 桌面尺度
DESK_MAX = 0.25
DESK_MIN = 0.03

cand = []
for L, name, sz in dims:
    if L > DESK_MAX or min(sz) < DESK_MIN:
        continue
    cand.append((L, name, sz))

print("桌面尺度资产（最长边 < %.0f cm 且最短边 > %.0f mm）：%d 个"
      % (DESK_MAX * 100, DESK_MIN * 1000, len(cand)))
print()

# 按"像工件"的程度排序：接近立方体或细长轴，避开片状
def score(item):
    L, name, sz = item
    s = sorted(sz, reverse=True)
    a = s[0] / max(s[1], 1e-6)     # 长细比
    b = s[1] / max(s[2], 1e-6)     # 扁不扁
    # 想要 长细比 1~4（柱状/方块），扁比 < 3
    ok = (1.0 <= a <= 4.0) and (b < 3.0)
    return (0 if ok else 1, abs(a - 2.0))


cand.sort(key=score)

print("最像'可抓取工件'的 30 个：")
print("  %-40s %-24s %s" % ("资产名", "尺寸 (cm)", "物理"))
print("  " + "-" * 78)
for L, name, sz in cand[:30]:
    r = next(x for x in rows if x["asset"] == name)
    ph = r.get("physics", {})
    ninfo = "bodies=%d hulls=%d joints=%d" % (
        ph.get("rigidBodies", 0),
        ph.get("collisionHulls", 0),
        ph.get("joints", 0),
    )
    print("  %-40s %7.1f x %5.1f x %5.1f   %s"
          % (name, sz[0] * 100, sz[1] * 100, sz[2] * 100, ninfo))

print()
print("=" * 80)
print("按类别找（关键词）")
print("=" * 80)
print()

KEYWORDS = {
    "轴/杆类（最像 D 轴）": r"(?i)(shaft|axle|rod|pin|bolt|screw|spindle|cylinder|pipe|tube)",
    "夹具/台钳": r"(?i)(vise|clamp|fixture|chuck|grip|jig)",
    "块/坯料": r"(?i)(block|billet|brick|bar_stock|ingot|plate|slab)",
    "料箱/容器": r"(?i)(bin|crate|box|container|basket|tray|bucket)",
    "齿轮/盘类": r"(?i)(gear|flange|disc|wheel|pulley|sprocket|collar)",
}

import re
for label, pat in KEYWORDS.items():
    hits = [(L, n, sz) for L, n, sz in dims
            if re.search(pat, n) and L <= 0.35]
    hits.sort()
    print("  %s ：%d 个（最长边 < 35 cm）" % (label, len(hits)))
    for L, n, sz in hits[:6]:
        print("      %-42s %.1f x %.1f x %.1f cm"
              % (n, sz[0] * 100, sz[1] * 100, sz[2] * 100))
    print()

print("OK")
