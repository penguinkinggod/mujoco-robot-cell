"""demo_05_toolpath.py —— 给机器人画一条"刀路"。

这个 demo 就是回答一个问题：**我怎么知道它对不对？**

在 SolidWorks 里你有刀路：图纸上该走的路画在那里，
机床真走的路可以叠上去比，偏了多少一眼看得见。

仿真里完全一样：

    目标轨迹（target）= 图纸上该走的路     <- 蓝色虚线
    实际轨迹（actual）= 机器人真的走的路   <- 实线
    两者之差         = 过切 / 欠切         <- 偏差曲线

这个 demo 让 Panda 的夹爪中心走两段路：
    1. 一条直线（40 cm）
    2. 一个圆（半径 6.5 cm）

并且同时跑两种做法，让你看见差别：

    开环：先把整条路的关节角全算好，然后闭着眼一路喂下去
    闭环：每走一步都重新看一眼"我现在在哪、目标在哪、差多少"，再修正

跑法：
    D:\\robot-env\\Scripts\\python.exe demo_05_toolpath.py

会生成：
    demo_05_toolpath.png    3D 轨迹对比图（这就是"刀路图"）
    demo_05_scene.png       带轨迹线的场景渲染
"""
import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import looks

# 中文字体：没有这行，图上的字全是方框
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.join(HERE, "models", "panda", "panda.xml")

# ============ 1. 建场景 ============
spec = mujoco.MjSpec.from_file(PANDA)

table = spec.worldbody.add_geom()
table.name = "table"
table.type = mujoco.mjtGeom.mjGEOM_PLANE
table.size = [1.4, 1.4, 0.1]

spec = looks.make_pretty(spec)
model = spec.compile()
data = mujoco.MjData(model)

BID = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
       for n in ("hand", "left_finger", "right_finger")}
JNT_RANGE = model.jnt_range[:7].copy()

# 夹爪中心 = hand 原点 + 0.0862 * hand 的 z 轴（指垫中心，见 diag_gripper_geom.py）
FINGER_OFFSET = 0.0862

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
Q_HOME = data.qpos[:7].copy()

# 末端姿态固定：hand 的 z 轴朝下（从上往下），这就是"刀轴方向"
R_DES = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])


def tool_point(d):
    """当前"刀尖"位置 = 夹爪中心。"""
    R = d.xmat[BID["hand"]].reshape(3, 3)
    return d.xpos[BID["hand"]] + FINGER_OFFSET * R[:, 2]


def rot_error(Rc, Rd):
    e = np.zeros(3)
    for i in range(3):
        e += np.cross(Rc[:, i], Rd[:, i])
    return 0.5 * e


def ik_step(target_pos, q_init, iters=40, gain=0.7, lam=0.05, tol=5e-4):
    """从 q_init 出发解到 target_pos。返回 (q, 残差)。"""
    q = q_init.copy()
    d = mujoco.MjData(model)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    err = np.ones(6)
    for _ in range(iters):
        d.qpos[:7] = q
        mujoco.mj_forward(model, d)
        ep = target_pos - tool_point(d)
        er = rot_error(d.xmat[BID["hand"]].reshape(3, 3), R_DES)
        err = np.concatenate([ep, 0.35 * er])
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacBody(model, d, jp, jr, BID["hand"])
        J = np.vstack([jp[:, :7], 0.35 * jr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        q = np.clip(q + gain * dq, JNT_RANGE[:, 0], JNT_RANGE[:, 1])
    return q, float(np.linalg.norm(err))


# ============ 2. 定义"刀路"（按时间参数化） ============
def ease(s):
    """余弦缓动：起停平滑，像真实机床的加减速。"""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0.0, 1.0))


N_CYCLE = 260          # 控制周期数（每个周期重新算一次）
# 每个控制周期跑多少物理步 -> 决定速度。步数越少 = 跑得越快 = 伺服越跟不上
SPEED_LEVELS = [10, 5, 3]     # 对应 5.2 秒 / 2.6 秒 / 1.6 秒 走完全程


def path_line(p0, p1, n):
    return np.array([p0 + ease(i / (n - 1)) * (p1 - p0) for i in range(n)])


def path_circle(center, radius, n):
    """在 y-z 平面画一个圆（像立式铣床的圆弧插补）。"""
    out = []
    for i in range(n):
        a = ease(i / (n - 1)) * 2 * np.pi
        out.append(center + np.array([0.0, radius * np.sin(a), radius * (1 - np.cos(a))]))
    return np.array(out)


# 工作空间里选一块安全区域（z 都在 0.2 以上，不会碰桌面）
P_START = np.array([0.50, -0.16, 0.26])
P_END = np.array([0.50, 0.22, 0.26])
CENTER = np.array([0.52, 0.04, 0.22])
RADIUS = 0.065

PATHS = {
    "直线 38cm": path_line(P_START, P_END, N_CYCLE),
    "圆 r=6.5cm": path_circle(CENTER, RADIUS, N_CYCLE),
}


# ============ 3. 两种走法 ============
def fresh_sim():
    d = mujoco.MjData(model)
    d.qpos[:7] = Q_HOME
    d.ctrl[:7] = Q_HOME
    d.ctrl[7] = 255.0
    d.qpos[7:9] = [0.04, 0.04]
    mujoco.mj_forward(model, d)
    return d


def move_to(d, target, steps=600):
    """平滑移动到某个点，作为起始动作。"""
    q_goal, _ = ik_step(target, d.qpos[:7], iters=250)
    q_from = d.qpos[:7].copy()
    for i in range(steps):
        a = ease((i + 1) / steps)
        d.ctrl[:7] = (1 - a) * q_from + a * q_goal
        mujoco.mj_step(model, d)
    return d


def run_open_loop(path, substeps):
    """开环：先把整条路的关节角全算好，再闭眼喂下去。

    注意这里的"闭眼"是真的闭眼：
    每步喂的是预规划好的 q_plan[i]，完全不看机器人现在到底在哪。
    """
    # 阶段 1：离线规划整条路的关节角
    q = Q_HOME.copy()
    q_plan = []
    for p in path:
        q, _ = ik_step(p, q, iters=80)
        q_plan.append(q.copy())

    # 阶段 2：从起点出发，闭眼执行
    d = fresh_sim()
    move_to(d, path[0], steps=600)

    actual = []
    for q_cmd in q_plan:
        d.ctrl[:7] = q_cmd
        for _ in range(substeps):
            mujoco.mj_step(model, d)
        actual.append(tool_point(d))
    return np.array(actual)


def run_closed_loop(path, substeps, gain=8.0, lead=1):
    """闭环：每个控制周期都量一下"我实际在哪、目标在哪"，把差值补回去。

    和开环的根本区别：

        开环  ctrl = 预规划好的角度（不看现实）
        闭环  ctrl = 当前实际角度 + 雅可比 × 误差 × 增益（每步看现实）

    为什么这样才有效：
        伺服永远有滞后。开环不知道这件事，误差会一路累积。
        闭环每步重新量误差，把"欠的那部分"加倍补进指令里，
        相当于提前多给一点，把滞后抵消掉。

    两个参数（都是实测调出来的，见 diag_closed_loop.py）：

        gain = 8.0   误差放大倍数。1.0 只能改善 1%，8.0 能改善 22 倍。
                     但不能无限大，太大会抖。

        lead = 1     往未来看一步。因为伺服有延迟，追"现在的位置"永远慢半拍，
                     追"下一步的位置"才刚好。lead=0 时改善只有 10 倍，
                     lead=1 时 22 倍。
    """
    d = fresh_sim()
    move_to(d, path[0], steps=600)

    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))

    actual = []
    n = len(path)
    for i, p in enumerate(path):
        # 前馈：追"未来"的目标，抵消伺服延迟
        p_ff = path[min(i + lead, n - 1)]

        # 量当前实际末端位置
        mujoco.mj_forward(model, d)
        cur = tool_point(d)
        e_pos = p_ff - cur

        # 把末端位置误差映射成关节增量（雅可比伪逆）
        mujoco.mj_jacBody(model, d, jp, jr, BID["hand"])
        J = jp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(3), e_pos)

        # 关键：指令 = 实际角度 + 修正量 × 增益
        q_cmd = np.clip(d.qpos[:7] + gain * dq, JNT_RANGE[:, 0], JNT_RANGE[:, 1])
        d.ctrl[:7] = q_cmd

        for _ in range(substeps):
            mujoco.mj_step(model, d)
        actual.append(tool_point(d))
    return np.array(actual)


print("=" * 74)
print("轨迹跟踪：开环 vs 闭环，三种速度")
print("=" * 74)
print("  控制周期 %d 次；速度靠「每个周期跑几步」控制" % N_CYCLE)
print("  步数越少 = 走得越快 = 伺服越跟不上")
print()

RESULTS = {}
for name, path in PATHS.items():
    print()
    print("=" * 74)
    print("[%s]" % name)
    print("=" * 74)
    print("  %-8s %-10s %-16s %-16s %-8s"
          % ("速度", "用时", "开环平均偏差", "闭环平均偏差", "改善"))
    print("  " + "-" * 66)

    per_speed = []
    for sub in SPEED_LEVELS:
        a_open = run_open_loop(path, sub)
        a_closed = run_closed_loop(path, sub)

        e_open = np.linalg.norm(a_open - path, axis=1)
        e_closed = np.linalg.norm(a_closed - path, axis=1)
        dur = N_CYCLE * sub * model.opt.timestep

        per_speed.append({
            "substeps": sub, "duration": dur,
            "open": a_open, "closed": a_closed,
            "e_open": e_open, "e_closed": e_closed,
        })
        print("  %-8s %-10s %-16s %-16s %-8s"
              % ("%d 步" % sub, "%.1f 秒" % dur,
                 "%.1f mm" % (e_open.mean() * 1000),
                 "%.1f mm" % (e_closed.mean() * 1000),
                 "%.1f 倍" % (e_open.mean() / max(e_closed.mean(), 1e-9))))

    RESULTS[name] = {"target": path, "runs": per_speed}

# ============ 4. 画"刀路图" ============
print()
print("=" * 74)
print("画图")
print("=" * 74)

fig = plt.figure(figsize=(16.5, 10.5))
fig.patch.set_facecolor("#0e1116")

COLORS = {
    "target": "#4da3ff",
    "open": "#ff7a45",
    "closed": "#3ddc84",
    "bg": "#0e1116",
    "fg": "#c9d1d9",
    "dim": "#5a6472",
}

# ---- 上半部分：3D 刀路对比（用最快那档，差异最明显）----
for idx, (name, r) in enumerate(RESULTS.items()):
    run = r["runs"][-1]          # 最快的一档
    ax = fig.add_subplot(3, 2, idx + 1, projection="3d")
    ax.set_facecolor(COLORS["bg"])

    t = r["target"]
    ax.plot(t[:, 0], t[:, 1], t[:, 2], "--", color=COLORS["target"],
            lw=2.6, label="目标（图纸）")
    ax.plot(run["open"][:, 0], run["open"][:, 1], run["open"][:, 2],
            color=COLORS["open"], lw=1.4, alpha=0.95, label="开环实际")
    ax.plot(run["closed"][:, 0], run["closed"][:, 1], run["closed"][:, 2],
            color=COLORS["closed"], lw=1.9, label="闭环实际")

    ax.set_title("%s（%.1f 秒走完）" % (name, run["duration"]),
                 color=COLORS["fg"], fontsize=12, pad=8)
    ax.tick_params(colors=COLORS["dim"], labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#151a21")
        pane.set_edgecolor("#232a33")
    ax.set_xlabel("x (m)", color=COLORS["dim"], fontsize=7)
    ax.set_ylabel("y (m)", color=COLORS["dim"], fontsize=7)
    ax.set_zlabel("z (m)", color=COLORS["dim"], fontsize=7)
    leg = ax.legend(loc="upper left", fontsize=7, framealpha=0.25)
    for txt in leg.get_texts():
        txt.set_color(COLORS["fg"])

# ---- 中下部分：偏差曲线（三档速度各一张）----
for idx, (name, r) in enumerate(RESULTS.items()):
    ax2 = fig.add_subplot(3, 2, 3 + idx)
    ax2.set_facecolor(COLORS["bg"])
    for run in r["runs"]:
        steps = np.arange(len(run["e_open"]))
        alpha = 0.35 + 0.65 * (run["substeps"] == min(SPEED_LEVELS))
        ax2.plot(steps, run["e_open"] * 1000, color=COLORS["open"],
                 lw=1.5 if alpha > 0.9 else 0.9, alpha=alpha,
                 label="开环 %.1fs" % run["duration"])
        ax2.plot(steps, run["e_closed"] * 1000, color=COLORS["closed"],
                 lw=1.6 if alpha > 0.9 else 0.9, alpha=alpha,
                 label="闭环 %.1fs" % run["duration"])
    ax2.set_title("%s — 偏差曲线" % name, color=COLORS["fg"], fontsize=11, pad=6)
    ax2.set_xlabel("控制周期", color=COLORS["dim"], fontsize=8)
    ax2.set_ylabel("偏差 (mm)", color=COLORS["dim"], fontsize=8)
    ax2.tick_params(colors=COLORS["dim"], labelsize=7)
    ax2.grid(True, color="#1e252e", lw=0.7)
    for sp in ax2.spines.values():
        sp.set_color("#232a33")
    leg = ax2.legend(fontsize=6.5, framealpha=0.25, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(COLORS["fg"])

# ---- 最下：速度 vs 平均误差（汇总）----
ax3 = fig.add_subplot(3, 2, 5)
ax3.set_facecolor(COLORS["bg"])
for name, r in RESULTS.items():
    durs = [x["duration"] for x in r["runs"]]
    eo = [x["e_open"].mean() * 1000 for x in r["runs"]]
    ec = [x["e_closed"].mean() * 1000 for x in r["runs"]]
    ls = "-" if "直线" in name else "--"
    ax3.plot(durs, eo, ls, color=COLORS["open"], marker="o", ms=5,
             label="开环 %s" % name)
    ax3.plot(durs, ec, ls, color=COLORS["closed"], marker="s", ms=5,
             label="闭环 %s" % name)
ax3.set_title("越快越不准？", color=COLORS["fg"], fontsize=11, pad=6)
ax3.set_xlabel("走完全程用时 (秒)  —— 越左越快", color=COLORS["dim"], fontsize=8)
ax3.set_ylabel("平均偏差 (mm)", color=COLORS["dim"], fontsize=8)
ax3.tick_params(colors=COLORS["dim"], labelsize=7)
ax3.grid(True, color="#1e252e", lw=0.7)
for sp in ax3.spines.values():
    sp.set_color("#232a33")
leg = ax3.legend(fontsize=6.5, framealpha=0.25, ncol=2)
for txt in leg.get_texts():
    txt.set_color(COLORS["fg"])

# ---- 最右下：说明卡 ----
ax4 = fig.add_subplot(3, 2, 6)
ax4.set_facecolor(COLORS["bg"])
ax4.axis("off")
ax4.text(0.02, 0.95, "这张图在说什么", color="#e6edf3", fontsize=12,
         fontweight="bold", va="top")
ax4.text(0.02, 0.82,
         "蓝虚线 = 图纸上该走的路\n"
         "橙实线 = 开环（先算完再闭眼走）\n"
         "绿实线 = 闭环（每步都重新看）\n\n"
         "1. 慢的时候，开环闭环差不多\n"
         "   —— 伺服跟得上，没有差别\n\n"
         "2. 快的时候，两条线都开始分叉\n"
         "   —— 这是伺服跟不上，不是代码错\n\n"
         "3. 开环的偏差会越走越大\n"
         "   —— 因为它不知道自己已经偏了\n\n"
         "这就是 SolidWorks 里\n"
         "\"过切 / 欠切\" 的同一个东西。",
         color="#9aa5b1", fontsize=9, va="top", linespacing=1.75,
         family="Microsoft YaHei")

fig.suptitle("这就是「刀路对不对」：把目标路径和实际路径叠在一起看",
             color="#e6edf3", fontsize=16, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(HERE, "demo_05_toolpath.png")
fig.savefig(out1, dpi=140, facecolor=fig.get_facecolor())
plt.close(fig)
print("  已保存 %s" % out1)

# ============ 5. 渲染带轨迹线的场景 ============
renderer = mujoco.Renderer(model, height=720, width=1080)
looks.enable_scene_flags(renderer)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.48, 0.04, 0.24]
cam.distance = 1.35
cam.azimuth = 140
cam.elevation = -12

d = fresh_sim()
move_to(d, CENTER, steps=700)

circle_pts = RESULTS["圆 r=6.5cm"]["runs"][-1]["closed"]


def add_line(scn, p1, p2, rgba, width=0.0022):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p1, p2)
    scn.ngeom += 1


renderer.update_scene(d, camera=cam)
for i in range(len(circle_pts) - 1):
    add_line(renderer.scene, circle_pts[i], circle_pts[i + 1],
             [0.30, 0.64, 1.0, 0.95], 0.0018)
img = renderer.render()
out2 = os.path.join(HERE, "demo_05_scene.png")
plt.imsave(out2, img)
print("  已保存 %s" % out2)

renderer.close()

print()
print("=" * 70)
print("怎么读这两张图")
print("=" * 70)
print("""
  左图（3D）：虚线是"图纸上该走的路"，实线是"机器人真走的路"。
      两条线重合 = 走对了。
      分叉       = 走偏了，偏多少看右图。

  右图（偏差曲线）：纵轴是毫米。
      线越低 = 越准。
      线一直爬高 = 误差在累积（开环的典型症状）。
      线贴着底   = 闭环把它压住了。

  这就是 SolidWorks 里"刀路仿真"的同一件事：
      目标路径 vs 实际路径，差多少，一眼看得见。

  你以后判断任何机器人代码"对不对"，都用这个方法：
      先画出目标，再画出实际，把两条线叠在一起。
""")
print("OK")
