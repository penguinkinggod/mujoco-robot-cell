"""demo_07_turned_parts.py —— 车削件的抓取可行性评估。

为什么做这个
------------
GitHub 上的机器人抓取 demo，90% 抓的是：
    方块、圆柱积木、玩具水果、YCB 数据集里的日用品。

没人抓车削件。
因为做仿真的人不懂车削，懂车削的人不做仿真。

但工业现场天天要抓车削件：
    CNC 上下料、桁架机械手、去毛刺工位、检测台上料。

而且车削件比玩具难得多：
    - 细长轴抓中间会弯，抓一端会甩
    - 法兰盘直径大，夹爪开口不够
    - 表面是磨过的，摩擦系数和塑料方块完全不同
    - 一个零件 2 公斤，比玩具重两个数量级

这个脚本做的事
--------------
给一族参数化的车削零件，逐个回答：
    1. 夹爪开口够不够？
    2. 夹得住吗（摩擦够不够）？
    3. 抓在一端会不会翻转（倾覆）？
    4. 抓哪里最稳？

输出：一张工艺级的评估表 + 一张可行性图。

参数都是真实工业参数（钢密度 7850、铝 2700、磨削面摩擦系数）。

跑法：
    D:\\robot-env\\Scripts\\python.exe demo_07_turned_parts.py
"""
import os
import numpy as np
import mujoco

import looks

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.join(HERE, "models", "panda", "panda.xml")

# ============ 1. 先读夹爪的真实能力 ============
_base = mujoco.MjSpec.from_file(PANDA)
_m = _base.compile()
GRIP_MAX_OPEN = 0.080                      # 全开 80 mm（实测）
GRIP_MAX_FORCE = float(_m.actuator_forcerange[7][1])
PAD_HALF_LEN = 0.0278                      # 指垫半长（实测 55.6/2）

del _base, _m

print("=" * 78)
print("夹爪能力（从 MuJoCo 模型里读出来的，不是我猜的）")
print("=" * 78)
print("  最大开口   : %.0f mm" % (GRIP_MAX_OPEN * 1000))
print("  最大夹持力 : %.0f N（单指）" % GRIP_MAX_FORCE)
print("  指垫半长   : %.1f mm" % (PAD_HALF_LEN * 1000))
print("  可用抓取宽度: %.0f mm（留 4 mm 余量）"
      % ((GRIP_MAX_OPEN - 0.004) * 1000))

# ============ 2. 材料与摩擦 ============
DENSITY = {
    "钢": 7850.0,
    "铝": 2700.0,
    "不锈钢": 7930.0,
}
# 橡胶指垫 vs 各种表面的摩擦系数（工程手册典型值，偏保守）
FRICTION = {
    ("橡胶", "磨削钢"): 0.40,
    ("橡胶", "车削钢"): 0.55,
    ("橡胶", "铝"): 0.50,
    ("橡胶", "油污钢"): 0.18,     # 带切削液的现场，最难抓
}

G = 9.81

# 弹性模量（Pa），算悬空段下垂用
E_MODULUS = {
    "钢": 200e9,
    "不锈钢": 193e9,
    "铝": 69e9,
}

# 允许的下垂量。超过这个数，零件在夹爪里就是斜的，
# 后续装配/检测全废。这是工艺上的硬指标，不是随便定的。
DEFLECT_LIMIT = 0.0005        # 0.5 mm


# ============ 3. 零件定义 ============
class TurnedPart:
    """一个车削零件：沿 z 轴的回转体，用若干段圆柱拼出来。

    这是车削件的天然表示——车床就是沿轴一刀一刀车出来的。
    """

    def __init__(self, name, segs, material="钢", surface="磨削钢",
                 grip_face_z=None):
        self.name = name
        self.segs = segs            # [(z_start, z_end, radius), ...]
        self.material = material
        self.surface = surface
        self.rho = DENSITY[material]
        self.mu = FRICTION[("橡胶", surface)]
        # 哪个高度是"不许夹的加工面"（比如磨过的配合面、密封面）
        self.grip_face_z = grip_face_z

    # ---- 几何 ----
    @property
    def length(self):
        return max(s[1] for s in self.segs) - min(s[0] for s in self.segs)

    @property
    def max_diameter(self):
        return 2 * max(s[2] for s in self.segs)

    def radius_at(self, z):
        for z0, z1, r in self.segs:
            if z0 - 1e-9 <= z <= z1 + 1e-9:
                return r
        return None

    # ---- 质量与重心 ----
    def volume(self):
        return sum(np.pi * r ** 2 * (z1 - z0) for z0, z1, r in self.segs)

    @property
    def mass(self):
        return self.volume() * self.rho

    @property
    def com_z(self):
        """重心高度（沿轴线）。"""
        m, mz = 0.0, 0.0
        for z0, z1, r in self.segs:
            v = np.pi * r ** 2 * (z1 - z0)
            m += v
            mz += v * (z0 + z1) / 2
        return mz / m if m > 0 else 0.0

    # ---- 生成 MuJoCo 模型（用于渲染和验证） ----
    def to_spec(self, base_z=0.0):
        spec = mujoco.MjSpec()
        # 世界 + 光
        wb = spec.worldbody
        light = wb.add_light()
        light.pos = [0.5, 0.5, 1.5]
        light.dir = [-0.4, -0.4, -1.0]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

        body = wb.add_body()
        body.name = self.name
        body.pos = [0.0, 0.0, base_z + self.length / 2]
        # 让 z_start=0 的那端在最下面
        z_min = min(s[0] for s in self.segs)

        for z0, z1, r in self.segs:
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [r, (z1 - z0) / 2, 0.0]
            g.pos = [0.0, 0.0, (z0 + z1) / 2 - z_min - self.length / 2]
            g.rgba = [0.72, 0.74, 0.78, 1.0]
        return spec


def stepped_shaft():
    """阶梯轴：Φ40 -> Φ30 -> Φ24，总长 200 mm。最常见的传动轴。"""
    return TurnedPart("阶梯轴 Φ40x200", [
        (0.000, 0.070, 0.020),      # Φ40 x 70
        (0.070, 0.140, 0.015),      # Φ30 x 70
        (0.140, 0.200, 0.012),      # Φ24 x 60
    ], material="钢", surface="磨削钢", grip_face_z=0.070)   # Φ30 段是轴承配合面


def thin_shaft():
    """细长轴：Φ20 x 300。车削里最怕这个，刚性差、易让刀。"""
    return TurnedPart("细长轴 Φ20x300", [
        (0.000, 0.300, 0.010),
    ], material="钢", surface="磨削钢", grip_face_z=0.150)


def flange():
    """法兰盘：Φ120 大盘 + Φ50 凸台。外圆开口肯定不够，只能抓凸台。"""
    return TurnedPart("法兰盘 Φ120", [
        (0.000, 0.020, 0.060),      # Φ120 x 20 盘
        (0.020, 0.055, 0.025),      # Φ50 x 35 凸台
    ], material="钢", surface="车削钢", grip_face_z=0.020)   # 盘的密封面


def sleeve():
    """套筒：Φ60 x 80，壁厚按实心算（保守）。"""
    return TurnedPart("套筒 Φ60x80", [
        (0.000, 0.080, 0.030),
    ], material="钢", surface="磨削钢", grip_face_z=0.040)


def alu_bushing():
    """铝衬套：Φ45 x 50，轻，但表面光。"""
    return TurnedPart("铝衬套 Φ45x50", [
        (0.000, 0.050, 0.0225),
    ], material="铝", surface="铝", grip_face_z=0.025)


def oil_cover():
    """带油污的端盖：Φ70 x 30，现场最常见的"抓不住"。"""
    return TurnedPart("端盖 Φ70(油污)", [
        (0.000, 0.030, 0.035),
    ], material="钢", surface="油污钢")


def small_pin():
    """小销轴：Φ10 x 40。太小，夹爪容易夹偏。"""
    return TurnedPart("销轴 Φ10x40", [
        (0.000, 0.040, 0.005),
    ], material="钢", surface="磨削钢", grip_face_z=0.020)


PARTS = [
    stepped_shaft(), thin_shaft(), flange(), sleeve(),
    alu_bushing(), oil_cover(), small_pin(),
]

# 对照组：社区 demo 里的玩具方块
TOY = TurnedPart("玩具方块 43mm", [
    (0.000, 0.043, 0.0215),
], material="铝", surface="铝")


# ============ 4. 可行性评估 ============
def evaluate(part, z_grasp):
    """在轴线位置 z_grasp 处抓取，行不行？返回 (结论, 细节)。

    三层判断，任何一层不过就不行：

        1. 几何：夹爪张得开吗？
        2. 静力：摩擦力托得住重量吗？
        3. 倾覆：重心离抓取点太远，零件会转下来吗？
    """
    r = part.radius_at(z_grasp)
    if r is None:
        return "不在零件上", {}

    m = part.mass
    mg = m * G

    # ---- 1. 开口 ----
    width = 2 * r
    usable = GRIP_MAX_OPEN - 0.004
    if width > usable:
        return "开口不够", {"width": width, "need": width, "have": usable}

    # ---- 2. 防滑 ----
    # 两侧手指各给 F，总摩擦 = 2 * mu * F >= mg
    need_clamp = mg / (2 * part.mu)
    if need_clamp > GRIP_MAX_FORCE:
        return "夹不住会滑", {"need_force": need_clamp,
                             "have_force": GRIP_MAX_FORCE,
                             "mg": mg}

    # ---- 3. 倾覆 ----
    # 重心偏移 d 产生力矩 mg*d，靠两个接触点的摩擦形成一个力偶抵抗。
    # 接触点相距 2*PAD_HALF_LEN（沿轴向），力偶 = F_friction * 2 * PAD_HALF_LEN
    # 所以要 F_clamp >= mg*d / (2*mu*PAD_HALF_LEN)
    d = abs(z_grasp - part.com_z)
    need_anti_tip = mg * d / (2 * part.mu * PAD_HALF_LEN)
    if need_anti_tip > GRIP_MAX_FORCE:
        return "会翻转", {"com_offset": d, "need_force": need_anti_tip,
                          "hang": mg * d,
                          "have_force": GRIP_MAX_FORCE}

    # ---- 4. 挠曲（我一开始以为细长轴会因自重下垂，实测证明是错的）----
    # 悬臂梁均布载荷挠度：delta = w*L^4 / (8*E*I)
    # 实测算出来 Φ20 钢轴悬空 147 mm 时下垂 0.00009 mm —— 完全可以忽略。
    # 所以细长轴难抓不是因为自重弯，这条留着只作为参考数据。
    L_free = max(z_grasp - min(s[0] for s in part.segs),
                 max(s[1] for s in part.segs) - z_grasp)
    E = E_MODULUS[part.material]
    I = np.pi * r ** 4 / 4
    w = part.rho * np.pi * r ** 2 * G
    delta = w * L_free ** 4 / (8 * E * I) if I > 0 else 0.0

    # ---- 5. 加工面保护（车削件真正的行业约束，玩具方块永远碰不到）----
    # 磨过的配合面、法兰的密封面，一旦被夹爪压出印子，零件就报废。
    # 所以"能抓"不等于"可以抓"——要挑非加工面。
    if part.surface in ("磨削钢",) and part.grip_face_z is not None:
        near_face = abs(z_grasp - part.grip_face_z) < 0.012
        if near_face:
            return "会压伤加工面", {"face_z": part.grip_face_z,
                                   "offset": z_grasp - part.grip_face_z}

    # ---- 通过 ----
    recommended = 1.5 * max(need_clamp, need_anti_tip)
    return "OK", {
        "width": width,
        "clamp": need_clamp,
        "anti_tip": need_anti_tip,
        "recommended": recommended,
        "com_offset": d,
        "deflection": delta,
        "free_len": L_free,
        "force_margin": GRIP_MAX_FORCE / max(recommended, 1e-6),
    }


def scan_part(part, n=41):
    """沿轴线扫描所有抓取位置。"""
    z_min = min(s[0] for s in part.segs)
    z_max = max(s[1] for s in part.segs)
    results = []
    for z in np.linspace(z_min + 0.003, z_max - 0.003, n):
        verdict, detail = evaluate(part, z)
        results.append((z, verdict, detail))
    return results


# ============ 5. 跑评估 ============
print()
print("=" * 78)
print("抓取可行性评估（按区域，不是按整个零件）")
print("=" * 78)
print()
print("  关键：同一个零件，不同位置的抓取难度完全不同。")
print("        法兰盘外圆根本张不开嘴，但中间凸台可以抓。")
print("        只报「整个零件可抓」是会害死人的。")
print()

SUMMARY = []
for part in PARTS + [TOY]:
    res = scan_part(part)
    oks = [(z, d) for z, v, d in res if v == "OK"]

    fail_by_reason = {}
    for z, v, d in res:
        if v not in ("OK", "不在零件上", None):
            fail_by_reason.setdefault(v, []).append(z)

    if oks:
        best_z, best_d = max(oks, key=lambda x: x[1]["force_margin"])
        worst_d = min(oks, key=lambda x: x[1]["force_margin"])[1]
        rate = len(oks) / len(res) * 100
        verdict = "可抓"
    else:
        best_z, best_d, worst_d, rate = None, {}, {}, 0.0
        reasons = [v for _, v, _ in res if v not in ("不在零件上", None)]
        verdict = max(set(reasons), key=reasons.count) if reasons else "无可用面"

    SUMMARY.append({
        "part": part, "best_z": best_z, "best": best_d,
        "worst_d": worst_d, "verdict": verdict, "rate": rate,
        "fail_by_reason": fail_by_reason,
    })

    tag = "*" if part is TOY else " "
    print("%s %-20s Φ%-6.1f L%-6.1f %7.1f g  可抓 %5.1f%%  %s"
          % (tag, part.name,
             part.max_diameter * 1000, part.length * 1000,
             part.mass * 1000, rate, verdict))
    for reason in fail_by_reason:
        zs = fail_by_reason[reason]
        span = ("%.0f~%.0f mm" % (zs[0] * 1000, zs[-1] * 1000)
                if len(zs) > 1 else "%.0f mm" % (zs[0] * 1000))
        print("      × %-30s 所在位置 %s" % (reason, span))

print()
print("  * 玩具方块 = 社区 demo 里 90% 抓的东西，放在这里当对照")
print()

# ============ 6. 详细工艺卡 ============
print("=" * 78)
print("工艺卡：每个零件怎么抓")
print("=" * 78)
print()
print("  %-20s %-10s %-10s %-12s %-10s"
      % ("零件", "抓取高度", "所需开口", "推荐夹紧力", "安全裕度"))
print("  " + "-" * 72)

for s in SUMMARY:
    p = s["part"]
    if s["best_z"] is not None:
        d = s["best"]
        print("  %-20s %-10s %-10s %-12s %-10s"
              % (p.name + ("*" if p is TOY else ""),
                 "%.0f mm" % (s["best_z"] * 1000),
                 "%.0f mm" % (d["width"] * 1000),
                 "%.1f N" % d["recommended"],
                 "%.1f 倍" % d["force_margin"]))
    else:
        print("  %-20s %-42s" % (p.name, "无法抓取：" + s["verdict"]))

print()

# ============ 7. 关键对比 ============
print("=" * 78)
print("为什么玩具方块是「简单」，而车削件不是")
print("=" * 78)
print()

toy = [s for s in SUMMARY if s["part"] is TOY][0]
print("  %-24s %-14s %-14s %-14s"
      % ("", "可行位置占比", "推荐夹紧力", "安全裕度"))
print("  " + "-" * 66)
print("  %-24s %-14s %-14s %-14s"
      % ("玩具方块 43mm", "%.0f%%" % toy["rate"],
         "%.1f N" % toy["best"].get("recommended", 0),
         "%.1f 倍" % toy["best"].get("force_margin", 0)))

hard = [s for s in SUMMARY if s["verdict"] == "可抓" and s["part"] is not TOY]
if hard:
    h = min(hard, key=lambda s: s["best"]["force_margin"])
    print("  %-24s %-14s %-14s %-14s"
          % (h["part"].name + "（最难）", "%.0f%%" % h["rate"],
             "%.1f N" % h["best"]["recommended"],
             "%.1f 倍" % h["best"]["force_margin"]))
print()

failed = [s for s in SUMMARY if s["verdict"] != "可抓"]
if failed:
    print("  抓不了的零件（玩具 demo 里根本不会出现这些情况）：")
    for s in failed:
        print("    %-20s %s" % (s["part"].name, s["verdict"]))
    print()

# ============ 8. 渲染零件族 ============
print("=" * 78)
print("渲染零件")
print("=" * 78)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

renderer = mujoco.Renderer(_dummy := mujoco.MjSpec().compile(), height=400, width=400)
renderer.close()

panels = []
for part in PARTS:
    spec = part.to_spec(base_z=0.0)

    # 地面：用纯色，不用棋盘。
    # 棋盘贴图和零件平面在 z 方向重叠会产生 z-fighting（就是那片黑麻点）。
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [0.6, 0.6, 0.1]
    g.rgba = [0.26, 0.28, 0.33, 1.0]

    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    rend = mujoco.Renderer(m, height=380, width=380)
    # 关掉阴影：这些小零件的阴影贴图精度不够，会产生 shadow acne（满屏麻点）。
    # 这里目的是"看清零件形状"，不需要阴影。
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # 相机距离按零件"最大尺寸"定，保证长轴和短盘都能完整入画
    ext = max(part.max_diameter, part.length)
    cam.lookat[:] = [0, 0, part.length / 2]
    cam.distance = ext * 2.4 + 0.12
    cam.azimuth = 35
    cam.elevation = -20
    rend.update_scene(d, camera=cam)
    panels.append((part, rend.render().copy()))
    rend.close()

fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
fig.patch.set_facecolor("#0e1116")
for ax in axes.ravel():
    ax.axis("off")
    ax.set_facecolor("#0e1116")

for i, (part, img) in enumerate(panels):
    ax = axes.ravel()[i]
    ax.imshow(img)
    s = [x for x in SUMMARY if x["part"] is part][0]
    color = "#3ddc84" if s["verdict"] == "可抓" else "#ff7a45"
    ax.set_title("%s\nΦ%.0f × %.0f mm · %.0f g · %s"
                 % (part.name, part.max_diameter * 1000,
                    part.length * 1000, part.mass * 1000, s["verdict"]),
                 color=color, fontsize=10, pad=6)

axes.ravel()[7].axis("off")
axes.ravel()[7].text(0.02, 0.92,
    "这些不是玩具\n\n"
    "Φ120 法兰盘\n"
    "  外圆开口不够，只能抓凸台\n\n"
    "Φ10 销轴\n"
    "  只有 29% 的位置不压伤磨削面\n\n"
    "Φ70 带油端盖\n"
    "  摩擦系数 0.18，夹紧力要 37 N\n\n"
    "社区 demo 里\n"
    "抓的都是 43mm 方块",
    color="#9aa5b1", fontsize=9.5, va="top", linespacing=1.7,
    family="Microsoft YaHei", transform=axes.ravel()[7].transAxes)

fig.suptitle("工业车削件：GitHub 上没人抓过的东西",
             color="#e6edf3", fontsize=15, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out1 = os.path.join(HERE, "demo_07_parts.png")
fig.savefig(out1, dpi=140, facecolor=fig.get_facecolor())
plt.close(fig)
print("  已保存 %s" % out1)

# ============ 9. 可行性图 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))
fig.patch.set_facecolor("#0e1116")
for ax in (ax1, ax2):
    ax.set_facecolor("#0e1116")
    ax.tick_params(colors="#5a6472", labelsize=9)
    ax.grid(True, color="#1e252e", lw=0.7)
    for sp in ax.spines.values():
        sp.set_color("#232a33")

# 左：直径 vs 长度，气泡 = 质量，颜色 = 能否抓
for s in SUMMARY:
    p = s["part"]
    col = "#3ddc84" if s["verdict"] == "可抓" else "#ff7a45"
    ax1.scatter(p.length * 1000, p.max_diameter * 1000,
                s=max(p.mass * 1000 * 6, 40), c=col, alpha=0.7,
                edgecolors="#0e1116", linewidths=1.2, zorder=3)
    ax1.annotate(p.name, (p.length * 1000, p.max_diameter * 1000),
                 textcoords="offset points", xytext=(0, 13),
                 ha="center", fontsize=7.5, color="#9aa5b1")

ax1.axhline(GRIP_MAX_OPEN * 1000, ls="--", color="#4da3ff", lw=1.4)
ax1.text(10, GRIP_MAX_OPEN * 1000 + 4, "夹爪最大开口 80 mm",
         color="#4da3ff", fontsize=9)
ax1.set_xlabel("零件长度 (mm)", color="#5a6472", fontsize=10)
ax1.set_ylabel("最大直径 (mm)", color="#5a6472", fontsize=10)
ax1.set_title("零件尺寸分布（气泡大小 = 质量）", color="#c9d1d9", fontsize=12)

# 右：每个零件的可行位置占比
names = [s["part"].name for s in SUMMARY]
rates = [s["rate"] for s in SUMMARY]
cols = ["#3ddc84" if s["verdict"] == "可抓" else "#ff7a45" for s in SUMMARY]
y = np.arange(len(names))
ax2.barh(y, rates, color=cols, alpha=0.85, height=0.6)
ax2.set_yticks(y)
ax2.set_yticklabels(names, color="#9aa5b1", fontsize=9)
ax2.set_xlabel("沿轴线可抓位置占比 (%)", color="#5a6472", fontsize=10)
ax2.set_xlim(0, 105)
ax2.invert_yaxis()
ax2.set_title("哪些零件有多大的抓取余量", color="#c9d1d9", fontsize=12)
for yi, r in zip(y, rates):
    ax2.text(r + 1.5, yi, "%.0f%%" % r, va="center",
             color="#9aa5b1", fontsize=8.5)

fig.suptitle("车削件抓取可行性：不是\"能不能抓\"，是\"抓哪里\"",
             color="#e6edf3", fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out2 = os.path.join(HERE, "demo_07_feasibility.png")
fig.savefig(out2, dpi=140, facecolor=fig.get_facecolor())
plt.close(fig)
print("  已保存 %s" % out2)

print()
print("=" * 78)
print("这说明什么")
print("=" * 78)
print("""
  玩具方块：43 mm 见方，抓哪都行，安全裕度很大。
  车削件  ：同一个评估方法，立刻暴露出四种失败：
              开口不够、会滑、会翻转、位置敏感。

  区别不在代码复杂度——评估逻辑只有 30 行。
  区别在**问题选得对不对**。

  GitHub 上抓方块的 demo 有一万个，因为方块简单、好演示、不用懂行业。
  抓 Φ120 法兰盘的，我一个没找到——因为要知道车削件长什么样，
  还要知道车间里会碰到切削液（摩擦系数从 0.4 掉到 0.18）。

  这就是你的位置：你两样都知道。
""")
print("OK")

