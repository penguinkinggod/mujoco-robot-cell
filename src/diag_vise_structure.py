"""diag_vise_structure.py —— 量清 SimReady 台钳的真实结构。

为什么必须量
-----------
我直接假设"钳口朝 -y"，结果钳口跑到机器人够不着的一侧。
SimReady 资产的局部坐标系是 CAD 导出的，朝向不固定。
必须实测：
    1. 钳口开口方向是哪个轴
    2. 钳口在局部坐标的什么位置
    3. 可动钳口到固定钳口的净空是多少
    4. 钳口底面高度（工件要贴在这里）

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_vise_structure.py
"""
import json
import os

import numpy as np

from simready_loader import load_obj, OBJ_DIR

NAME = "equipment_bench_vise_a01"

print("=" * 76)
print("台钳结构分析：%s" % NAME)
print("=" * 76)
print()

V, F = load_obj(os.path.join(OBJ_DIR, NAME + ".obj"))
print("整体：verts=%d faces=%d" % (len(V), len(F)))
lo, hi = V.min(axis=0), V.max(axis=0)
print("包围盒 X: %.4f ~ %.4f (%.4f)" % (lo[0], hi[0], hi[0] - lo[0]))
print("包围盒 Y: %.4f ~ %.4f (%.4f)" % (lo[1], hi[1], hi[1] - lo[1]))
print("包围盒 Z: %.4f ~ %.4f (%.4f)" % (lo[2], hi[2], hi[2] - lo[2]))
print()

# 按物理子部件拆开：从 phys.json 拿不到顶点分组，
# 改从 OBJ 的 object 分组读
groups = {}
cur = None
vstart = 0
with open(os.path.join(OBJ_DIR, NAME + ".obj"), encoding="utf-8",
          errors="ignore") as f:
    vcount = 0
    for line in f:
        if line.startswith("o "):
            cur = line[2:].strip()
            groups[cur] = [vcount, vcount, 0]   # [start, end, facecount]
        elif line.startswith("v "):
            vcount += 1
        elif line.startswith("f ") and cur:
            groups[cur][1] = vcount
            groups[cur][2] += 1

print("OBJ 里的 object 分组：")
for name, (a, b, nf) in groups.items():
    if b <= a:
        continue
    sub = V[a:b]
    slo, shi = sub.min(axis=0), sub.max(axis=0)
    print("  %-28s verts=%-6d faces=%-6d  size=%.3f x %.3f x %.3f"
          % (name, b - a, nf, *(shi - slo)))
    print("  %-28s   X[%.3f,%.3f] Y[%.3f,%.3f] Z[%.3f,%.3f]"
          % ("", slo[0], shi[0], slo[1], shi[1], slo[2], shi[2]))
print()

# ---- 找两个钳口的净空 ----
# 钳口是两块大致平行的竖板，间距最小的地方就是开口
# 用 Y-Z 平面上的 X 分布来看：台钳的钳口沿丝杠方向开合，
# 通常是沿 Y（长轴方向），开口在 Y 中间
print("=" * 76)
print("判断钳口方向")
print("=" * 76)
print()

size = hi - lo
print("  整体尺寸：X %.3f  Y %.3f  Z %.3f" % (size[0], size[1], size[2]))
long_axis = int(np.argmax(size))
print("  最长轴   ：%s（丝杠方向，钳口沿这个方向开合）" % "XYZ"[long_axis])
print("  竖直轴   ：Z")
up_axis = 2

# 钳口高度范围：取 Z 的上半部分（钳口在台钳上方）
z_thresh = lo[2] + size[2] * 0.55
mask = V[:, 2] > z_thresh
Vtop = V[mask]
if len(Vtop) > 100:
    tlo, thi = Vtop.min(axis=0), Vtop.max(axis=0)
    print()
    print("  钳口区域（Z > %.3f）的包围盒：" % z_thresh)
    print("    X: %.3f ~ %.3f" % (tlo[0], thi[0]))
    print("    Y: %.3f ~ %.3f" % (tlo[1], thi[1]))
    print("    Z: %.3f ~ %.3f" % (tlo[2], thi[2]))

    # 沿最长轴做直方图，钳口会表现为两个峰
    hist, edges = np.histogram(Vtop[:, long_axis], bins=40)
    print()
    print("  沿 %s 轴的顶点分布（钳口会显示为两个峰）：" % "XYZ"[long_axis])
    peak = hist.max()
    for i in range(len(hist)):
        bar = "#" * int(30 * hist[i] / peak) if peak else ""
        mid = (edges[i] + edges[i + 1]) / 2
        print("    %+.3f |%-30s %d" % (mid, bar, hist[i]))

    # 找两个最高峰之间的空隙 = 净空
    thr = peak * 0.25
    above = hist > thr
    runs = []
    i = 0
    while i < len(hist):
        if above[i]:
            j = i
            while j < len(hist) and above[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    print()
    print("  连续密集区: %s" % runs)
    if len(runs) >= 2:
        gap_lo = edges[runs[0][1] + 1]
        gap_hi = edges[runs[1][0]]
        print("  两块密集区之间的空隙: %.4f m（%.1f mm）"
              % (gap_hi - gap_lo, (gap_hi - gap_lo) * 1000))
        print("  -> 这就是钳口净空")
        print("  -> 钳口中心在 %s = %.4f" % ("XYZ"[long_axis],
                                              (gap_lo + gap_hi) / 2))
    else:
        print("  没有找到两个分离的峰，钳口可能是闭合状态或被凸包挡住")

print()
print("OK")
