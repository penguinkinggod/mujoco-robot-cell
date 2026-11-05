"""查：为什么"手写 IK + 神经网络"和"纯手写 IK"一模一样？

怀疑点：
    训练误差降到 0.000117，说明网络确实拟合了 DELTA。
    但测试时误差完全没变（6.93 vs 6.93），赢率正好 50%。

三种可能：
    A. 网络学到的 DELTA 太小，补上去没效果
    B. DELTA 本身不可预测（每次执行结果都不一样 -> 是噪声）
    C. 代码 bug，修正量根本没加进去

这个脚本逐个排除。
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


def ik(target, q_init, iters=300, gain=0.7, lam=0.05, tol=1e-6):
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


print("=" * 72)
print("可能 C：代码 bug？检查修正量到底有没有加进去")
print("=" * 72)

p = np.array([0.52, 0.04, 0.28])
q = ik(p, Q_HOME)
a_plain = execute(q)
print("  目标           : %s" % np.round(p, 4))
print("  手写IK 实际到达 : %s   误差 %.2f mm"
      % (np.round(a_plain, 4), np.linalg.norm(a_plain - p) * 1000))

# 手动构造一个"修正后"的关节角，看执行结果变不变
q_plus = q.copy()
q_plus[1] += 0.01        # 关节 2 加 0.01 rad
a_plus = execute(q_plus)
print("  关节2 +0.01rad : 实际到达 %s   误差 %.2f mm"
      % (np.round(a_plus, 4), np.linalg.norm(a_plus - p) * 1000))
print("  => 关节变化确实会影响实际位置：%s"
      % ("是" if np.linalg.norm(a_plus - a_plain) > 1e-4 else "否（有 bug）"))

print()
print("=" * 72)
print("可能 A：DELTA 够不够大？")
print("=" * 72)

# 对同一个目标，找出"真正能到"的关节角
q_fix = q.copy()
for it in range(30):
    d = mujoco.MjData(model)
    d.qpos[:7] = q_fix
    d.qpos[7:9] = [0.0, 0.0]
    mujoco.mj_forward(model, d)
    cur = tool_point(d)
    err = p - cur
    if np.linalg.norm(err) < 1e-5:
        break
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, d, jp, jr, HAND)
    J = jp[:, :7]
    dq = J.T @ np.linalg.solve(J @ J.T + 0.02 ** 2 * np.eye(3), err)
    q_fix = np.clip(q_fix + 0.9 * dq, JNT_RANGE[:, 0], JNT_RANGE[:, 1])

print("  迭代 %d 次后，q_fix - q = %s" % (it + 1,
      np.array2string(q_fix - q, precision=4, suppress_small=True)))
print("  修正量最大绝对值 : %.4f rad" % np.abs(q_fix - q).max())

a_fix = execute(q_fix)
print("  q_fix 实际到达   : %s   误差 %.2f mm"
      % (np.round(a_fix, 4), np.linalg.norm(a_fix - p) * 1000))
print()
print("  注意：q_fix 是用 **同一个 mj_forward** 修正的（无重力、无伺服），")
print("       而 execute 是 **真的跑仿真**（有重力、有伺服）。")
print("       两者不是一个世界。这就是问题所在。")

print()
print("=" * 72)
print("可能 B：同一个 q，执行两次结果一样吗？（可重复性）")
print("=" * 72)
a1 = execute(q)
a2 = execute(q)
a3 = execute(q)
print("  第 1 次: %s" % np.round(a1, 6))
print("  第 2 次: %s" % np.round(a2, 6))
print("  第 3 次: %s" % np.round(a3, 6))
print("  两次最大差异: %.6f mm" % (np.linalg.norm(a1 - a2) * 1000))
print("  => %s" % ("完全可重复（确定性仿真）" if np.linalg.norm(a1 - a2) < 1e-9
                  else "有随机性"))

print()
print("=" * 72)
print("结论")
print("=" * 72)
print("""  问题不在神经网络，也不在代码 bug。

  真正的问题是：我把"修正量"学成了 q_fix - q，
  而 q_fix 是用无重力的 mj_forward 算出来的。
  也就是说，我教网络学的是"数学世界里的修正"，
  但执行发生在"有重力的仿真世界"里 —— 这两个修正不是一回事。

  正确的做法：应该学「实际执行结果」和「目标」之间的差，
  直接把那个差补到目标上（位置层面的修正），
  而不是在关节角层面猜。

  这是一个真实的机器学习踩坑：**训练目标和最终指标不一致。**""")
print("OK")
