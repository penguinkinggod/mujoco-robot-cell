"""diag_jaw_face_sweep.py -- 把三个卡爪的接触面改小，能不能治住小直径互顶。

上一轮变径扫描（diag_diameter_sweep.py）的结论是：Φ18 以下三爪先互相顶上，
力全花在爪与爪之间；按该直径重新镗孔也照样顶，所以不是弧面半径的问题。
当时猜的原因是 104 deg 包角 + 5 段 box 拼弧 + 段间 1.02 倍重叠。

本脚本把「接触面」拆成三个可调参数，只测卡盘这一站：

    half_deg  每爪半包角     36 / 42 / 48 / 52 deg（出厂 52）
    overlap   相邻弧片段重叠  1.00 / 1.02（出厂 1.02）
    thick     弧片径向厚度    0.0025 / 0.005 m（出厂 0.005）

弧片是直 box 沿圆弧摆的，box 内表面的角向投影比中心大；料越细、弧片越厚，
放大越狠，三爪内缘就越容易先撞上。所以减薄和缩包角是对症的两条路，一起扫。

阶段一（默认）：只跑 Φ15（问题最重）和 Φ20（主线基准），粗扫互顶与接触。
阶段二（--ratio）：在 diag_chuck_health 的主线场景下细扫三爪力比。这一步不能省：
    包角缩太小，短弧片会把 0.1 mm 的三爪同步差放大成 4 倍力不均，
    互顶是没了，主线体检却挂了。两个场景摆位不同，得各量各的。
阶段三（--full）：拿一组候选参数跑 Φ15~Φ25 × fit/phi20 全十组。

跑法：
    D:\\robot-env\\Scripts\\python.exe src/diag_jaw_face_sweep.py
    D:\\robot-env\\Scripts\\python.exe src/diag_jaw_face_sweep.py --ratio
    D:\\robot-env\\Scripts\\python.exe src/diag_jaw_face_sweep.py --full \
        --half 44 --overlap 1.02 --thick 0.005

输出：_build/jaw_face_sweep.csv / .png，--ratio 加 _ratio、--full 加 _full 后缀
"""
import argparse
import contextlib
import csv
import io
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diag_diameter_sweep as W
import scene_machining as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "_build")
os.makedirs(OUT, exist_ok=True)

GRID_HALVES = (36.0, 42.0, 48.0, 52.0)
RATIO_HALVES = (36.0, 40.0, 42.0, 44.0, 45.0, 48.0, 52.0)
OVERLAPS = (1.00, 1.02)
THICKS = (0.0025, 0.005)
PROBE_D = (0.015, 0.020)
FULL_D = (0.015, 0.018, 0.020, 0.022, 0.025)
_DEF = (S.JAW_ARC_HALF_DEG, S.JAW_ARC_OVERLAP, S.JAW_ARC_THICK)


def one(D, mode, half, overlap, thick):
    r = W.run_case(D, mode, half_deg=half, overlap=overlap, thick=thick)
    r.update(half_deg=half, wrap_deg=2 * half, overlap=overlap,
             thick_mm=thick * 1000)
    return r


def grid():
    rows = []
    print("=" * 106)
    print("接触面扫描：半包角 x 段间重叠 x 弧片厚度，只测卡盘这一站")
    print("=" * 106)
    print("  %6s %6s %8s %6s %7s %7s %9s %9s %8s %8s %9s"
          % ("半包角", "直径", "互顶力N", "互顶对", "背板N", "接触",
             "爪力N", "同轴mm", "倾斜°", "拉拔mm", "判定"))
    print("  " + "-" * 100)
    for thick in THICKS:
        for overlap in OVERLAPS:
            for half in GRID_HALVES:
                for D in PROBE_D:
                    r = one(D, "fit", half, overlap, thick)
                    rows.append(r)
                    print("  %6.0f Φ%-5.0f %8.0f %6d %7.0f %7d %9.1f %9.2f %8.2f %8.2f %9s"
                          % (half, r["D_mm"], r["jaw_jam_N"], r["jaw_jam_pairs"],
                             r["back_jam_N"], r["contacts"], r["force_N"],
                             r["coax_mm"], r["tilt_deg"], r["pull20_mm"],
                             "PASS" if r["ok"] else "FAIL"))
    return rows


def ratio_scan():
    """主线体检场景下的三爪力比。这个场景从镗孔位斜坡闭合，摆位和
    diag_diameter_sweep 的悬空摆位不同，力比单独量才有意义。
    """
    import diag_chuck_health as H

    rows = []
    print("=" * 96)
    print("三爪力比扫描（diag_chuck_health 主线场景，判据 力比 <= 2.0）")
    print("=" * 96)
    print("  半包角  重叠  厚度mm   力比   位移差mm  接触  三爪力 N            体检")
    print("  " + "-" * 88)
    for thick in THICKS:
        for overlap in OVERLAPS:
            for half in RATIO_HALVES:
                S.JAW_ARC_HALF_DEG = half
                S.JAW_ARC_OVERLAP = overlap
                S.JAW_ARC_THICK = thick
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = H.main()
                out = buf.getvalue()
                m = re.search(
                    r"各爪法向力 \(N\)\s+:\s+\[([^\]]+)\]\s+力比\s+([\d.]+)", out)
                s = re.search(r"三爪位移 \(mm\)\s+:\s+\[[^\]]+\]\s+差\s+([\d.]+)", out)
                if not m:
                    continue
                r = dict(half_deg=half, wrap_deg=2 * half, overlap=overlap,
                         thick_mm=thick * 1000, force_ratio=float(m.group(2)),
                         spread_mm=float(s.group(1)) if s else -1.0,
                         forces_N=m.group(1).strip(), health_pass=(rc == 0))
                rows.append(r)
                print("  %6.0f  %.2f  %6.2f  %6.2f  %9.3f  %4s  %-20s %s"
                      % (half, overlap, thick * 1000, r["force_ratio"],
                         r["spread_mm"], "30", r["forces_N"],
                         "PASS" if r["health_pass"] else "FAIL"))
    # 恢复出厂值，免得同一进程后续误用
    S.JAW_ARC_HALF_DEG, S.JAW_ARC_OVERLAP, S.JAW_ARC_THICK = _DEF
    return rows


def full(half, overlap, thick):
    rows = []
    print("=" * 106)
    print("全直径验证：half=%.0f deg  overlap=%.2f  thick=%.2f mm"
          % (half, overlap, thick * 1000))
    print("=" * 106)
    for mode in ("fit", "phi20"):
        for D in FULL_D:
            r = one(D, mode, half, overlap, thick)
            rows.append(r)
            print("  %-6s Φ%-5.0f 互顶 %7.0f N / %2d 对   背板 %5.0f N   "
                  "接触 %2d   爪力 %7.1f N   同轴 %.2f mm  倾斜 %.2f°  "
                  "拉拔 %.2f mm  %s"
                  % (mode, r["D_mm"], r["jaw_jam_N"], r["jaw_jam_pairs"],
                     r["back_jam_N"], r["contacts"], r["force_N"],
                     r["coax_mm"], r["tilt_deg"], r["pull20_mm"],
                     "PASS" if r["ok"] else "FAIL"))
    return rows


def save(rows, tag):
    csv_path = os.path.join(OUT, "jaw_face_sweep%s.csv" % tag)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n已写出 %s" % csv_path)
    return csv_path


def plot_grid(rows):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    titles = ("jaw-jam force (N)", "jaw-shaft contacts", "jaw normal force (N)")
    keys = ("jaw_jam_N", "contacts", "force_N")
    styles = (("1.00 / 2.5mm", 1.00, 0.0025, "#c0392b", "o"),
              ("1.00 / 5.0mm", 1.00, 0.005, "#e67e22", "s"),
              ("1.02 / 2.5mm", 1.02, 0.0025, "#2c3e50", "^"),
              ("1.02 / 5.0mm", 1.02, 0.005, "#2980b9", "v"))
    for a, key, title in zip(axes, keys, titles):
        for d, lab in ((15, "D15"), (20, "D20")):
            for label, ov, th, color, mk in styles:
                sub = [r for r in rows
                       if round(r["D_mm"]) == d and r["overlap"] == ov
                       and abs(r["thick_mm"] - th * 1000) < 1e-6]
                sub.sort(key=lambda r: r["wrap_deg"])
                a.plot([r["wrap_deg"] for r in sub], [r[key] for r in sub],
                       mk + ("-" if d == 15 else "--"), color=color,
                       linewidth=1.3 if d == 15 else 1.0,
                       alpha=1.0 if d == 15 else 0.55, markersize=4,
                       label=("%s %s" % (lab, label)) if d == 15 else None)
        a.set_title(title, fontsize=10)
        a.set_xlabel("jaw wrap angle (deg)")
        a.grid(alpha=0.3)
    axes[0].axhline(1.0, color="#999999", linestyle=":", linewidth=1)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("solid = Φ15 (problem case), dashed = Φ20 (baseline)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "jaw_face_sweep.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("       %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="跑全直径验证")
    ap.add_argument("--ratio", action="store_true", help="跑主线场景三爪力比扫描")
    ap.add_argument("--half", type=float, default=44.0)
    ap.add_argument("--overlap", type=float, default=1.02)
    ap.add_argument("--thick", type=float, default=0.005)
    args = ap.parse_args()

    if args.ratio:
        save(ratio_scan(), "_ratio")
        return 0
    if args.full:
        rows = full(args.half, args.overlap, args.thick)
        save(rows, "_full")
    else:
        rows = grid()
        save(rows, "")
        plot_grid(rows)
        # 直接把「Φ15 不互顶且 Φ20 还夹得住」的组合列出来
        ok = []
        for ov in OVERLAPS:
            for th in THICKS:
                for h in GRID_HALVES:
                    a = next(r for r in rows if round(r["D_mm"]) == 15
                             and r["half_deg"] == h and r["overlap"] == ov
                             and abs(r["thick_mm"] - th * 1000) < 1e-6)
                    b = next(r for r in rows if round(r["D_mm"]) == 20
                             and r["half_deg"] == h and r["overlap"] == ov
                             and abs(r["thick_mm"] - th * 1000) < 1e-6)
                    if a["ok"] and b["ok"] and b["contacts"] >= 6:
                        ok.append((h, ov, th, a, b))
        print("\n  互顶这一关可行的组合（Φ15 不互顶 + Φ20 还夹得住）：")
        print("  注意：还要过 --ratio 那一关，主线三爪力比 <= 2.0 才算数。")
        if not ok:
            print("    无。缩接触面单独治不了，看减薄那条曲线。")
        for h, ov, th, a, b in ok:
            print("    half=%2.0f  overlap=%.2f  thick=%.2fmm  "
                  "→ Φ15 互顶 %.0f N / Φ20 接触 %d 点 / 爪力 %.0f N / 同轴 %.2f mm"
                  % (h, ov, th * 1000, a["jaw_jam_N"], b["contacts"],
                     b["force_N"], b["coax_mm"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
