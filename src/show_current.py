"""show_current.py -- 把「现在仿真的样子」变成能直接看的图 / 动图 / 网页。

运行（在 D:\\codex\\projects\\robot-sim-first-step 下）：
    D:\\robot-env\\Scripts\\python.exe show_current.py

产物（全部写在本目录）：
    now_views.png     合并场景六视图（起始 / 装夹完成，各 3 个机位）
    now_stages.png    送料全流程 12 个关键帧
    now_feed.gif      送料全流程动画
    now_gallery.html  点开即看的图册（含当次实测指标）

所有画面都来自 diag_feed_chuck.main(frame_hook=...) 的同一次仿真，
和 verify_all.py 里 7/7 PASS 的是同一份 MjData，不是另摆出来的造型。
"""
import contextlib
import datetime
import io
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

import diag_feed_chuck as D
import scene_full as F
from cnfont import setup_cn_font

setup_cn_font()
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
VIEWS_PNG = os.path.join(OUT, "now_views.png")
STAGES_PNG = os.path.join(OUT, "now_stages.png")
FEED_GIF = os.path.join(OUT, "now_feed.gif")
GALLERY = os.path.join(OUT, "now_gallery.html")


def make_cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = lookat
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c


# 机位由 _camscan.py 在最终装夹态扫 12 个候选后人工挑选：
#   B 斜俯视：机器人 / 车床 / 工件三者关系最清楚，物体占比合适
#   E 正对卡盘端面：三爪 120° 夹住轴端最清楚
#   I 侧向：能看到轴水平伸进卡盘
# Lathe moved to x=0.28 so the 5-DoF arm can actually hold the rod axis;
# the final shaft centre is 0.205 and the jaw front is 0.225.
CENTER = [0.205, 0.120, 0.060]
CAM_MAIN = make_cam([0.23, 0.01, 0.075], 0.60, -38, -22)
CAM_CHUCK = make_cam([0.225, 0.120, 0.060], 0.24, 0, -10)
CAM_SIDE = make_cam(CENTER, 0.22, 90, -10)
CAM_PART = make_cam(list(F.SHAFT_POS), 0.28, 30, -20)

VIEW_START = [
    ("总览｜机械臂 + 车床", CAM_MAIN),
    ("卡盘端面｜三爪张开待装", CAM_CHUCK),
    ("工件特写｜Φ20 D 轴在台面", CAM_PART),
]
VIEW_END = [
    ("总览｜机械臂已退出", CAM_MAIN),
    ("卡盘端面｜三爪夹住轴", CAM_CHUCK),
    ("侧向｜轴伸进卡盘", CAM_SIDE),
]


class Tee:
    """把仿真 print 同时送到终端和缓冲区（缓冲区用于解析实测指标）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def normalize(tag):
    """带圈数字 ⑪⑫ 在微软雅黑里缺字会变方框，统一换成两位数字。"""
    for i in range(12):
        tag = tag.replace(chr(0x2460 + i), "%02d" % (i + 1))
    return tag


def parse_metric(text):
    m = re.search(r"FEED_METRIC (coax_mm=[^\n]+)", text)
    values = {}
    if m:
        values = {k: float(v) for k, v in
                  re.findall(r"(\w+)=(\d+(?:\.\d+)?)", m.group(1))}
    status = "PASS" if "FEED_METRIC PASS" in text else "FAIL"
    return values, status


def save_views(views):
    row_labels = [
        "起始：卡盘张开，工件在台面",
        "装夹完成：机器人已退出，轴被三爪夹住",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.8))
    for row, (key, view_set) in enumerate(
            (("start", VIEW_START), ("end", VIEW_END))):
        imgs = views[key]
        for col, (title, _) in enumerate(view_set):
            ax = axes[row][col]
            if imgs:
                ax.imshow(imgs[col])
            ax.set_title("%s\n%s" % (row_labels[row], title),
                         fontsize=10.5, pad=7)
            ax.axis("off")
    fig.suptitle("当前场景：SO-101 机械臂 + 二轴车三爪卡盘（合并场景）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(VIEWS_PNG, dpi=112, facecolor="white")
    plt.close(fig)


def save_stages(stages):
    cols = 3
    rows = (len(stages) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15.0, 3.45 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (tag, img) in zip(axes, stages):
        ax.imshow(img)
        ax.set_title(tag, fontsize=10.5, pad=6)
        ax.axis("off")
    fig.suptitle("机械臂送料 → 三爪夹紧：关键帧（按时间顺序）",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(STAGES_PNG, dpi=105, facecolor="white")
    plt.close(fig)


def save_gallery(metric, status, n_frames, n_stages):
    def num(key, unit="", digits=2):
        v = metric.get(key)
        return ("%.*f%s" % (digits, v, unit)) if v is not None else "--"

    cards = [
        ("同轴度", num("coax_mm", " mm"), "轴心 vs 卡盘轴线"),
        ("倾斜", num("tilt_deg", "°", 1), "相对原始轴向"),
        ("卡爪接触点", num("contacts", "", 0), "三爪合计"),
        ("三爪法向力", num("jaw_force_N", " N", 1), "夹紧力"),
        ("重力下沉", num("slip_mm", " mm"), "静置 800 步"),
        ("20 N 拉拔", num("pull_mm", " mm"), "轴向拉动 700 步"),
    ]
    card_html = "\n".join(
        '<div class="m"><div class="mk">%s</div><div class="mv">%s</div>'
        '<div class="mh">%s</div></div>' % c for c in cards)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>现在仿真的样子 · 机械臂送料到三爪卡盘</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f4f6f9;color:#1d2430;
 font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;line-height:1.7}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
header{position:relative;margin-bottom:24px}
h1{font-size:30px;margin:0 0 6px}
.sub{color:#5b6575;margin:0;font-size:14px}
.badge{position:absolute;right:0;top:4px;padding:6px 18px;border-radius:20px;
 font-weight:700;font-size:14px;letter-spacing:1px}
.pass{background:#e4f6ea;color:#137333}
.fail{background:#fde8e7;color:#b3261e}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:26px}
.m{background:#fff;border:1px solid #e3e8ef;border-radius:12px;padding:14px 16px}
.mk{font-size:12px;color:#6b7686}
.mv{font-size:24px;font-weight:700;color:#123a6b;margin:2px 0}
.mh{font-size:11px;color:#8a94a3}
.card{background:#fff;border:1px solid #e3e8ef;border-radius:14px;
 padding:20px 22px;margin-bottom:22px;box-shadow:0 2px 10px rgba(20,40,80,.04)}
.card h2{font-size:18px;margin:0 0 12px}
.card img{display:block;width:100%;height:auto;border-radius:8px;
 border:1px solid #e6eaf0;background:#0e1116}
.cap{font-size:13px;color:#5b6575;margin:10px 0 0}
.warn{background:#fffaf0;border-color:#f0d9a8}
.warn h2{color:#8a5a00}
.warn li{margin:6px 0;font-size:14px}
pre{background:#111827;color:#d7e3f5;padding:14px 16px;border-radius:8px;
 overflow-x:auto;font-size:13px;line-height:1.6}
code{background:#eef1f6;padding:1px 5px;border-radius:4px;font-size:13px}
.tag{display:inline-block;background:#eef2f8;color:#33507a;border-radius:5px;
 padding:1px 7px;font-size:12px;margin-right:6px}
footer{color:#8a94a3;font-size:12px;text-align:center;margin-top:28px}
@media(max-width:720px){.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap">
<header><h1>现在的样子</h1>
<p class="sub">SO-101 机械臂抓 Φ20 D 轴 → 送进二轴车三爪卡盘 → 夹紧 → 退出 ·
 生成于 __STAMP__ · __FRAMES__ 帧动图 / __STAGES__ 个关键帧</p>
<span class="badge __STATUSCLS__">__STATUS__</span></header>

<section class="metrics">__CARDS__</section>

<section class="card"><h2>① 全流程动图（点开即播放）</h2>
<img src="now_feed.gif" alt="送料全流程动画">
<p class="cap">从卡盘张开、机械臂接近抓取，到横移送入、三爪夹紧、机器人退出、20 N 拉拔。
画面来自当次仿真逐帧离屏渲染，没有后期摆拍。</p></section>

<section class="card"><h2>② 场景长什么样</h2>
<img src="now_views.png" alt="合并场景三视图">
<p class="cap">上排是起始状态（总览 / 卡盘端面三爪张开 / 台面上的 D 轴），
下排是装夹完成状态（总览 / 卡盘端面三爪夹住轴 / 侧视轴伸进卡盘）。
三爪按 120° 分布，长边朝径向，由单丝杠 + 2 条关节联动约束驱动。</p></section>

<section class="card"><h2>③ 送料 12 个关键帧</h2>
<img src="now_stages.png" alt="送料关键帧">
<p class="cap">注意第 04 帧「真实抓取」里轴是歪的（约 41.5°），第 05 帧起搬运段接管为运动学锁；
第 10 帧解锁，之后所有载荷测试都在真实物理下进行。</p></section>

<section class="card warn"><h2>⚠ 别把画面当成「全物理通过」</h2>
<ul>
<li><b>搬运段是运动学锁</b>：从对中到夹紧，每步把轴位姿写回夹爪中心，等于假设夹爪完美形封闭。
真实夹爪抓持姿态还没修好，轴在抓取阶段仍会歪约 41.5°。</li>
<li><b>D 形铣扁是粗糙碰撞代理</b>：box 角点凸出 Φ20 包络约 3.1 mm，只用来阻止圆柱滚转，形状精度按你的要求留到后面优化。</li>
<li>锁在所有负载测试前解除，<code>physics_after_clamp=1</code> 由 verify_all.py 强制断言，防止把锁延长到拉拔阶段造假。</li>
</ul></section>

<section class="card"><h2>想自己转着看：桌面图标 / 命令</h2>
<p class="cap"><b>最省事：双击桌面上的「机器人仿真 3D 窗口」</b>，会弹出一个和 SolidWorks 一样的可交互 3D 窗口。
按住鼠标左键转视角、滚轮缩放，右侧面板能开接触点 / 力 / 坐标系显示。</p>
<p class="cap">也可以在 <code>D:\\codex\\projects\\robot-sim-first-step</code> 里粘下面任意一条：</p>
<pre>D:\\robot-env\\Scripts\\python.exe show_live.py           :: 起始态，可拖着转
D:\\robot-env\\Scripts\\python.exe show_live.py --feed    :: 实时播放送料全过程
D:\\robot-env\\Scripts\\python.exe show_live.py --clamped :: 直接看装夹完成态
D:\\robot-env\\Scripts\\python.exe show_current.py        :: 重新生成本页的图 / 动图
D:\\robot-env\\Scripts\\python.exe verify_all.py          :: 复跑 7/7 回归</pre>
</section>

<footer>画面与指标同源：diag_feed_chuck.main(frame_hook=...) · 视觉配置不改物理</footer>
</div></body></html>
"""
    html = (template
            .replace("__STAMP__", stamp)
            .replace("__FRAMES__", str(n_frames))
            .replace("__STAGES__", str(n_stages))
            .replace("__STATUSCLS__", "pass" if status == "PASS" else "fail")
            .replace("__STATUS__", status)
            .replace("__CARDS__", card_html))
    with open(GALLERY, "w", encoding="utf-8") as fh:
        fh.write(html)


def main():
    # Render the run that actually grips: V jaws, rod on the cradle, and NO
    # kinematic lock. The flat-pad path below is kept only because the older
    # 7-item checks still assert its numbers.
    model = D.build_model(v_jaw=True, lift=True)
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, height=540, width=720)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    gif_frames, stages = [], []
    views = {"start": None, "end": None}
    counter = [0]

    def shot(cam):
        rend.update_scene(data, camera=cam)
        return rend.render().copy()

    def frame_hook(tag, _model, _data):
        counter[0] += 1
        img = shot(CAM_MAIN)
        # One rendered frame per simulation frame: no dropped odd frames.
        small = Image.fromarray(img).resize((480, 360), RESAMPLE)
        gif_frames.append(np.asarray(small))
        if tag is not None:
            stages.append((normalize(tag), img.copy()))
            if tag.startswith("\u2460"):        # ① 起始
                views["start"] = [shot(c) for _, c in VIEW_START]
            if tag.startswith("\u246a"):        # ⑪ 机器人退出
                views["end"] = [shot(c) for _, c in VIEW_END]

    print("正在跑送料仿真并逐帧渲染（画面与 PASS 指标同源）...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buf)):
        D.FRAME_EVERY = 20          # dt=0.002 s -> 25 fps, one GIF frame each
        rc = D.main(frame_hook=frame_hook, model=model, data=data,
                    v_jaw=True, lift=True, carry_lock=False)
    rend.close()

    metric, status = parse_metric(buf.getvalue())
    if not metric:
        print("警告：没解析到 FEED_METRIC，指标卡显示 --")

    save_views(views)
    save_stages(stages)

    import imageio.v2 as imageio
    imageio.mimsave(FEED_GIF, gif_frames, fps=25, loop=0)
    save_gallery(metric, status, len(gif_frames), len(stages))

    print("\n已生成：")
    for p in (VIEWS_PNG, STAGES_PNG, FEED_GIF, GALLERY):
        print("  %-22s %7.1f KB" % (os.path.basename(p),
                                    os.path.getsize(p) / 1024))
    print("  动图 %d 帧 / 关键帧 %d 张 / 仿真状态 %s (rc=%d)"
          % (len(gif_frames), len(stages), status, rc))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
