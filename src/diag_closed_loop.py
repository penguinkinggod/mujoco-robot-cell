"""调闭环增益：找出能真正压住滞后的增益值。

为什么需要这一步
----------------
第一版闭环用了"每步重解 IK"，结果和开环一模一样（1.0 倍）。
原因：IK 收敛后的解与初值无关，所以"重解"根本不构成反馈。

第二版改成"指令 = 实际角度 + 雅可比 × 位置误差 × 增益"，
但 gain=1.0 时只改善 0.4 mm。

这里扫一遍增益，看能不能压住。
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
PANDA = os.path.join(ROOT, "models", "panda", "panda.xml")

spec = mujoco.MjSpec.from_file(PANDA)
t = spec.worldbody.add_geom()
t.type = mujoco.mjtGeom.mjGEOM_PLANE
t.size = [1.4, 1.4, 0.1]
model = spec.compile()
data = mujoco.MjData(model)

HAND = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
JNT_RANGE = model.jnt_range[:7].copy()
FINGER_OFFSET = 0.0862

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
Q_HOME = data.qpos[:7].copy()

R_DES = np.diag([1.0, -1.0, -1.0])


def tool_point(d):
    R = d.xmat[HAND].reshape(3, 3)
    return d.xpos[HAND] + FINGER_OFFSET * R[:, 2]


def rot_error(Rc, Rd):
    e = np.zeros(3)
    for i in range(3):
        e += np.cross(Rc[:, i], Rd[:, i])
    return 0.5 * e


def ik_step(target, q_init, iters=80, gain=0.7, lam=0.05, tol=5e-4):
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
    return q, float(np.linalg.norm(err))


def ease(s):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0.0, 1.0))


N = 260
P0 = np.array([0.50, -0.16, 0.26])
P1 = np.array([0.50, 0.22, 0.26])
PATH = np.array([P0 + ease(i / (N - 1)) * (P1 - P0) for i in range(N)])


def fresh():
    d = mujoco.MjData(model)
    d.qpos[:7] = Q_HOME
    d.ctrl[:7] = Q_HOME
    d.ctrl[7] = 255.0
    d.qpos[7:9] = [0.04, 0.04]
    mujoco.mj_forward(model, d)
    return d


def move_to(d, target, steps=600):
    qg, _ = ik_step(target, d.qpos[:7], iters=250)
    qf = d.qpos[:7].copy()
    for i in range(steps):
        a = ease((i + 1) / steps)
        d.ctrl[:7] = (1 - a) * qf + a * qg
        mujoco.mj_step(model, d)


def open_loop(path, sub):
    q = Q_HOME.copy()
    q_plan = [ik_step(p, q, iters=80)[0] for p in path]
    # 上一行的列表推导有问题：初值没更新。重写
    q = Q_HOME.copy()
    q_plan = []
    for p in path:
        q, _ = ik_step(p, q, iters=80)
        q_plan.append(q.copy())

    d = fresh()
    move_to(d, path[0])
    out = []
    for qc in q_plan:
        d.ctrl[:7] = qc
        for _ in range(sub):
            mujoco.mj_step(model, d)
        out.append(tool_point(d))
    return np.array(out)


def closed_loop(path, sub, gain, lead=0.0):
    """gain: 位置误差反馈增益；lead: 前馈提前量（目标点往前看几步）"""
    d = fresh()
    move_to(d, path[0])
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    out = []
    n = len(path)
    for i, p in enumerate(path):
        # 前馈：往未来看 lead 步的目标
        j = min(i + int(lead), n - 1)
        p_ff = path[j]

        mujoco.mj_forward(model, d)
        cur = tool_point(d)

        # 误差 = 未来目标 - 当前位置
        e = p_ff - cur
        mujoco.mj_jacBody(model, d, jp, jr, HAND)
        J = jp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(3), e)
        qc = np.clip(d.qpos[:7] + gain * dq, JNT_RANGE[:, 0], JNT_RANGE[:, 1])
        d.ctrl[:7] = qc
        for _ in range(sub):
            mujoco.mj_step(model, d)
        out.append(tool_point(d))
    return np.array(out)


print("=" * 70)
print("闭环增益 + 前馈提前量 扫描（最快档 sub=3）")
print("=" * 70)
SUB = 3
a_open = open_loop(PATH, SUB)
eo = np.linalg.norm(a_open - PATH, axis=1)
print("  开环：平均 %.1f mm  最大 %.1f mm" % (eo.mean() * 1000, eo.max() * 1000))
print()
print("  %-8s %-10s %-14s %-14s %-10s" % ("gain", "lead", "平均偏差", "最大偏差", "改善"))
print("  " + "-" * 58)
best = (1e9, None, None)
for gain in (1.0, 2.0, 3.0, 5.0, 8.0):
    for lead in (0, 1, 2, 3):
        ac = closed_loop(PATH, SUB, gain, lead)
        ec = np.linalg.norm(ac - PATH, axis=1)
        imp = eo.mean() / max(ec.mean(), 1e-9)
        flag = ""
        if ec.mean() < best[0]:
            best = (ec.mean(), gain, lead)
            flag = "  <-- 最好"
        print("  %-8.1f %-10d %-14s %-14s %-10s%s"
              % (gain, lead,
                 "%.1f mm" % (ec.mean() * 1000),
                 "%.1f mm" % (ec.max() * 1000),
                 "%.2f 倍" % imp, flag))

print()
print("  最好组合：gain=%.1f lead=%d -> %.1f mm"
      % (best[1], best[2], best[0] * 1000))
print("OK")
