"""demo_06_learn_correction.py —— 这个 demo 只回答一个问题：

    "我们做的这些，和 AI / 机器学习 / 深度学习，到底有什么关系？"

答案用一次实验给出来，不用听我讲。

实验设计
--------
demo_05 已经量过一个事实：
    手写 IK 算出的关节角喂进去，实际到达位置会偏 6~7 毫米。
    原因不是数学错，是重力 + 伺服刚度（demo_04 的稳态误差）。

这个偏差有个特点：**同一个目标位置，偏的方向和大小每次都一样**
（刚验证过：执行 3 次，结果完全一致，差 0.000000 mm）。
换句话说，它不是随机噪声，而是**可以被预测的系统性偏差**。

于是有两条路：

    做法 A（传统工程）：把重力项、摩擦项一项一项补进公式。
                       demo_01 里加 qfrc_bias 就是这一步。能补一部分。

    做法 B（机器学习）：不管它为什么偏，直接学
                       「目标位置 -> 实际偏差」这个映射，
                       以后预测着提前补偿。

这个 demo 把两条路都跑，比数字。

顺便解释三个词：

    机器学习 = 从数据里找规律，不需要你写出公式
    深度学习 = 用多层神经网络找规律，能拟合更复杂的关系
    AI       = 上面这些的总称，共同点是"人少写规则，数据决定行为"

跑法：
    D:\\robot-env\\Scripts\\python.exe demo_06_learn_correction.py
"""
import os
import time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
PANDA = os.path.join(ROOT, "models", "panda", "panda.xml")

# ============ 1. 建场景 ============
spec = mujoco.MjSpec.from_file(PANDA)
t = spec.worldbody.add_geom()
t.type = mujoco.mjtGeom.mjGEOM_PLANE
t.size = [1.4, 1.4, 0.1]
model = spec.compile()

HAND = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
JNT_RANGE = model.jnt_range[:7].copy()
FINGER_OFFSET = 0.0862
R_DES = np.diag([1.0, -1.0, -1.0])

d0 = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, d0, 0)
mujoco.mj_forward(model, d0)
Q_HOME = d0.qpos[:7].copy()


def tool_point(d):
    R = d.xmat[HAND].reshape(3, 3)
    return d.xpos[HAND] + FINGER_OFFSET * R[:, 2]


def rot_error(Rc, Rd):
    e = np.zeros(3)
    for i in range(3):
        e += np.cross(Rc[:, i], Rd[:, i])
    return 0.5 * e


def ik(target, q_init, iters=250, gain=0.7, lam=0.05, tol=1e-5):
    """手写 IK（就是我们一直在用的那个）。"""
    q = q_init.copy()
    d = mujoco.MjData(model)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    err = np.ones(6)
    for _ in range(iters):
        d.qpos[:7] = q
        mujoco.mj_forward(model, d)
        err = np.concatenate([target - tool_point(d),
                              0.35 * rot_error(d.xmat[HAND].reshape(3, 3), R_DES)])
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacBody(model, d, jp, jr, HAND)
        J = np.vstack([jp[:, :7], 0.35 * jr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        q = np.clip(q + gain * dq, JNT_RANGE[:, 0], JNT_RANGE[:, 1])
    return q


def execute(q_cmd, steps=400):
    """真的把关节角喂进去跑仿真，返回实际到达位置。

    这一步是"物理世界"：重力、摩擦、伺服刚度都在这里发生。
    注意它和 mj_forward 完全不同 —— mj_forward 只是算一次状态，
    不推进时间，伺服也不会"跟不上"。
    """
    d = mujoco.MjData(model)
    d.qpos[:7] = Q_HOME
    d.ctrl[:7] = Q_HOME
    d.ctrl[7] = 255.0
    d.qpos[7:9] = [0.04, 0.04]
    mujoco.mj_forward(model, d)

    for i in range(300):
        a = (i + 1) / 300
        a = 0.5 - 0.5 * np.cos(np.pi * a)
        d.ctrl[:7] = (1 - a) * Q_HOME + a * q_cmd
        mujoco.mj_step(model, d)
    for _ in range(steps):
        d.ctrl[:7] = q_cmd
        mujoco.mj_step(model, d)
    return tool_point(d)


# ============ 2. 造数据 ============
print("=" * 74)
print("第 1 步：造数据 —— 看手写 IK 到底偏多少")
print("=" * 74)

rng = np.random.default_rng(42)
N_SAMPLES = 400

LO = np.array([0.44, -0.18, 0.18])
HI = np.array([0.60, 0.22, 0.38])

t0 = time.time()
TARGETS, ERRORS, Q_CMDS = [], [], []

while len(TARGETS) < N_SAMPLES:
    p = rng.uniform(LO, HI)
    q = ik(p, Q_HOME)
    actual = execute(q)
    e = actual - p                       # 关键：这个才是"实际偏差"
    if np.linalg.norm(e) > 0.05:         # 太离谱的点扔掉
        continue
    TARGETS.append(p)
    ERRORS.append(e)
    Q_CMDS.append(q)

TARGETS = np.array(TARGETS)
ERRORS = np.array(ERRORS)
Q_CMDS = np.array(Q_CMDS)

print("  采集 %d 组，用时 %.1f 秒" % (len(TARGETS), time.time() - t0))
print()
print("  这个偏差长什么样（前 5 行）：")
print("  %-26s %-30s %-10s" % ("目标位置", "实际偏差 dx dy dz", "大小"))
print("  " + "-" * 70)
for i in range(5):
    print("  %-26s %-30s %-10s"
          % (np.array2string(TARGETS[i], precision=3),
             np.array2string(ERRORS[i] * 1000, precision=2, suppress_small=True),
             "%.2f mm" % (np.linalg.norm(ERRORS[i]) * 1000)))

print()
print("  偏差统计：")
print("    平均大小   : %.2f mm" % (np.linalg.norm(ERRORS, axis=1).mean() * 1000))
print("    x 方向平均 : %+.2f mm" % (ERRORS[:, 0].mean() * 1000))
print("    y 方向平均 : %+.2f mm" % (ERRORS[:, 1].mean() * 1000))
print("    z 方向平均 : %+.2f mm  <-- 主要往下掉，重力" % (ERRORS[:, 2].mean() * 1000))
print("    标准差     : %.2f mm" % (ERRORS.std() * 1000))

# ============ 3. 划分训练/测试 ============
idx = rng.permutation(len(TARGETS))
n_tr = int(len(TARGETS) * 0.8)
tr, te = idx[:n_tr], idx[n_tr:]

X_tr, E_tr = TARGETS[tr], ERRORS[tr]
X_te, E_te = TARGETS[te], ERRORS[te]
Q_te = Q_CMDS[te]

X_MEAN, X_STD = X_tr.mean(0), X_tr.std(0) + 1e-8
E_MEAN, E_STD = E_tr.mean(0), E_tr.std(0) + 1e-8


def nx(x):
    return (x - X_MEAN) / X_STD


def ne(x):
    return (x - E_MEAN) / E_STD


def de(x):
    return x * E_STD + E_MEAN


# ============ 4. 纯 NumPy 神经网络 ============
print()
print("=" * 74)
print("第 2 步：训练神经网络（纯 NumPy，不装 PyTorch）")
print("=" * 74)

IN_DIM, HID, OUT_DIM = 3, 64, 3

rng2 = np.random.default_rng(0)
W1 = rng2.normal(0, np.sqrt(2.0 / IN_DIM), (IN_DIM, HID)); b1 = np.zeros(HID)
W2 = rng2.normal(0, np.sqrt(2.0 / HID), (HID, HID));      b2 = np.zeros(HID)
W3 = rng2.normal(0, np.sqrt(2.0 / HID), (HID, OUT_DIM));  b3 = np.zeros(OUT_DIM)
PARAMS = [W1, b1, W2, b2, W3, b3]
MOM = [np.zeros_like(p) for p in PARAMS]


def forward(x):
    """前向传播：3 个数进 -> 两层 64 -> 3 个数出（预测偏差 dx dy dz）"""
    z1 = x @ W1 + b1
    a1 = np.maximum(z1, 0)          # ReLU，非线性
    z2 = a1 @ W2 + b2
    a2 = np.maximum(z2, 0)
    z3 = a2 @ W3 + b3
    return z3, (x, z1, a1, z2, a2)


def backward(grad_out, cache):
    """反向传播：算出每个参数该往哪调。"""
    x_in, z1, a1, z2, a2 = cache
    gW3 = a2.T @ grad_out
    gb3 = grad_out.sum(0)
    ga2 = grad_out @ W3.T
    gz2 = ga2 * (z2 > 0)
    gW2 = a1.T @ gz2
    gb2 = gz2.sum(0)
    ga1 = gz2 @ W2.T
    gz1 = ga1 * (z1 > 0)
    gW1 = x_in.T @ gz1
    gb1 = gz1.sum(0)
    return [gW1, gb1, gW2, gb2, gW3, gb3]


LR, EPOCHS, BATCH = 0.008, 4000, 64
losses = []

print("  输入：目标位置 xyz (3 个数)")
print("  输出：预测偏差 dx dy dz (3 个数)")
print("  网络：3 -> 64 -> 64 -> 3，参数 %d 个" % sum(p.size for p in PARAMS))
print()

t0 = time.time()
for ep in range(EPOCHS):
    perm = rng2.permutation(len(X_tr))
    tot = 0.0
    for s in range(0, len(X_tr), BATCH):
        b = perm[s:s + BATCH]
        xb, yb = nx(X_tr[b]), ne(E_tr[b])
        pred, cache = forward(xb)
        diff = pred - yb
        tot += float((diff ** 2).mean()) * len(b)
        grads = backward(2.0 * diff / diff.size, cache)
        for i, (p, g) in enumerate(zip(PARAMS, grads)):
            MOM[i] = 0.9 * MOM[i] - LR * g
            p += MOM[i]
    losses.append(tot / len(X_tr))
    if (ep + 1) % 800 == 0:
        print("  第 %4d 轮  训练误差 %.6f" % (ep + 1, losses[-1]))

print("  训练完成，用时 %.1f 秒" % (time.time() - t0))

# ============ 5. 测试 ============
print()
print("=" * 74)
print("第 3 步：在没见过的目标点上比三种做法")
print("=" * 74)
print("  测试集 %d 个点，训练时完全没见过" % len(X_te))
print()

e_plain, e_ml, e_oracle = [], [], []

t0 = time.time()
for i in range(len(X_te)):
    p = X_te[i]

    # 做法 1：纯手写 IK
    a1 = execute(Q_te[i])
    e_plain.append(np.linalg.norm(a1 - p))

    # 做法 2：手写 IK + 神经网络预测的补偿
    #   网络预测"会偏多少"，那我们就把目标往反方向挪，提前补偿
    pred_err, _ = forward(nx(p[None, :]))
    predicted = de(pred_err)[0]
    p_corrected = p - predicted            # 关键：目标反向修正
    q2 = ik(p_corrected, Q_te[i])
    a2 = execute(q2)
    e_ml.append(np.linalg.norm(a2 - p))

    # 做法 3（作弊上限）：直接知道真实偏差
    #   只用来告诉你"理论上最好能做到多少"，实际做不到
    p_oracle = p - E_te[i]
    q3 = ik(p_oracle, Q_te[i])
    a3 = execute(q3)
    e_oracle.append(np.linalg.norm(a3 - p))

e_plain = np.array(e_plain)
e_ml = np.array(e_ml)
e_oracle = np.array(e_oracle)

print("  %-30s %-12s %-12s %-12s" % ("", "平均", "中位数", "最差"))
print("  " + "-" * 68)
print("  %-30s %-12s %-12s %-12s"
      % ("1. 纯手写 IK", "%.2f mm" % (e_plain.mean() * 1000),
         "%.2f mm" % (np.median(e_plain) * 1000),
         "%.2f mm" % (e_plain.max() * 1000)))
print("  %-30s %-12s %-12s %-12s"
      % ("2. IK + 神经网络补偿", "%.2f mm" % (e_ml.mean() * 1000),
         "%.2f mm" % (np.median(e_ml) * 1000),
         "%.2f mm" % (e_ml.max() * 1000)))
print("  %-30s %-12s %-12s %-12s"
      % ("3. 作弊上限（知道真实偏差）", "%.2f mm" % (e_oracle.mean() * 1000),
         "%.2f mm" % (np.median(e_oracle) * 1000),
         "%.2f mm" % (e_oracle.max() * 1000)))
print()
print("  神经网络补偿改善 : %.1f 倍" % (e_plain.mean() / max(e_ml.mean(), 1e-9)))
print("  神经网络赢的比例 : %.0f%%" % ((e_ml < e_plain).mean() * 100))
print("  测试用时 %.1f 秒" % (time.time() - t0))

# ============ 6. 画图 ============
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
fig.patch.set_facecolor("#0e1116")
for ax in axes:
    ax.set_facecolor("#0e1116")
    ax.tick_params(colors="#5a6472", labelsize=8)
    ax.grid(True, color="#1e252e", lw=0.7)
    for sp in ax.spines.values():
        sp.set_color("#232a33")

axes[0].plot(losses, color="#4da3ff", lw=1.4)
axes[0].set_title("训练误差（越低 = 学得越好）", color="#c9d1d9", fontsize=11)
axes[0].set_xlabel("训练轮数", color="#5a6472", fontsize=9)
axes[0].set_ylabel("误差", color="#5a6472", fontsize=9)

bins = np.linspace(0, max(e_plain.max(), e_oracle.max()) * 1000 * 1.05, 36)
axes[1].hist(e_plain * 1000, bins=bins, color="#ff7a45", alpha=0.75,
             label="纯手写 IK")
axes[1].hist(e_ml * 1000, bins=bins, color="#3ddc84", alpha=0.8,
             label="IK + 神经网络")
axes[1].hist(e_oracle * 1000, bins=bins, color="#4da3ff", alpha=0.6,
             label="作弊上限")
axes[1].set_title("测试集误差分布（越靠左越好）", color="#c9d1d9", fontsize=11)
axes[1].set_xlabel("误差 (mm)", color="#5a6472", fontsize=9)
axes[1].set_ylabel("目标点个数", color="#5a6472", fontsize=9)
leg = axes[1].legend(fontsize=9, framealpha=0.25)
for txt in leg.get_texts():
    txt.set_color("#c9d1d9")

axes[2].scatter(e_plain * 1000, e_ml * 1000, s=18, c="#3ddc84",
                alpha=0.65, edgecolors="none")
lim = max(e_plain.max(), e_ml.max()) * 1000 * 1.06
axes[2].plot([0, lim], [0, lim], "--", color="#5a6472", lw=1.2, label="一样好")
axes[2].set_xlim(0, lim)
axes[2].set_ylim(0, lim)
axes[2].set_title("每个测试点\n对角线下方 = 神经网络更好", color="#c9d1d9", fontsize=10)
axes[2].set_xlabel("纯手写 IK 误差 (mm)", color="#5a6472", fontsize=9)
axes[2].set_ylabel("IK + 神经网络 (mm)", color="#5a6472", fontsize=9)
leg = axes[2].legend(fontsize=9, framealpha=0.25)
for txt in leg.get_texts():
    txt.set_color("#c9d1d9")

fig.suptitle("机器学习在机器人里做的事：学掉公式算不准的那一部分",
             color="#e6edf3", fontsize=14, y=1.02)
fig.tight_layout()
out = os.path.join(OUT, "demo_06_learn.png")
fig.savefig(out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)
print()
print("  已保存 %s" % out)

print()
print("=" * 74)
print("这个 demo 想说的话")
print("=" * 74)
print("""
  我没改任何物理参数，没换仿真器，也没改 IK。
  只是给它看了 %d 组「目标位置 -> 实际偏差」的数据，
  它自己找出偏差的规律，然后**提前把目标挪一下**。

  这就是机器学习：从数据里找规律，而不是靠人把公式推出来。

  深度学习：这里用了两层隐层、%d 个参数。
            层数和参数越多，能拟合的规律越复杂——所以叫"深度"。

  AI：一个笼统的说法。这套方法、加上强化学习、加上大模型，都算 AI。
      它们的共同点是：**人少写规则，让数据决定行为。**

  和你熟悉的 SolidWorks 有什么关系？
      你调切削参数，靠的是老师傅的经验和试切。
      机器学习做的是同一件事：把"试出来的经验"变成可复用的模型。
      区别只是：一个存在人脑子里，一个存在矩阵里。

  重要：这个 demo 的结论要说清楚
      神经网络能补掉"可预测的那部分"偏差，补不掉"随机的那部分"。
      所以它永远打不过"作弊上限"（那是知道真实答案的）。
      真实工程里也是这样：ML 不是万能，它是把公式做不干净的地方再擦一遍。
""" % (len(TARGETS), sum(p.size for p in PARAMS)))

print("OK")
