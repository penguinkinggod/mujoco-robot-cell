"""demo_08_pick_and_load.py —— 完整动作：抓 D 轴 -> 搬到台钳 -> 插入 -> 松开。

动作序列
--------
    home -> 移到轴上方 -> 下降 -> 夹紧 -> 抬起
         -> 横移到钳口上方 -> 下降 -> 松开 -> 退开

关键参数（全部实测标定，不是猜的）
--------------------------------
1. wrist_roll = 1.55 rad
   实测（diag_grasp_pose.py）：这个角度下两个 pad 的连线是 [-0.065, 0.996, 0.068]
   - 与工件轴线 x 的夹角 86.3°（接近垂直，能左右夹住）
   - 与水平面夹角 3.9°（两个 pad 等高，水平张开）
   其他角度看 pad 连线会变成竖直，只能夹竖直的东西。

2. 夹爪开合（diag_grasp_geometry.py 实测）
       ctrl = 0.00 -> pad 间距 14.8 mm
       ctrl = 0.50 -> 52.7 mm
       ctrl = 1.74 -> 128  mm
   Φ20 工件：闭合到 ctrl ≈ -0.08（间距 18 mm），张开用 ctrl = 0.5

3. IK 必须用 mj_forward
   mj_kinematics 只算位置，雅可比会返回全 0（实测过）。

4. 目标点是 pad 中心，不是 site
   实测 pad 中点比 gripperframe 偏 [31.2 mm]（沿 pad 连线方向）。

跑法：
    D:\\robot-env\\Scripts\\python.exe demo_08_pick_and_load.py
"""
import os

import numpy as np
import mujoco

import scene_full as S

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll"]

WRIST_ROLL_GRASP = 1.55      # 实测最佳
GRIP_OPEN = 0.50
# 标定 (diag_grasp_geometry.py): pad 中心距 mm = 14.8 + 75.8 * ctrl。
# 旧值 -0.08 -> 间距 8.7 mm，比 D20 轴还窄，夹爪把轴从两片之间挤飞
# (接触穿透 6 mm)，抬升 0.0 mm。0.15 -> 28.7 mm，实测抬升 97 mm；
# >= 0.20 又太松夹不住。这是全项目抓取失败的根因。
GRIP_CLOSE = 0.15
GRIP_FREE = 1.20


class Arm:
    def __init__(self, model, data, renderer=None):
        self.rend = renderer
        self.frames = []
        self.m = model
        self.d = data
        self.qadr = [model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ARM_JOINTS]
        self.dadr = [model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ARM_JOINTS]
        self.lo = np.array([model.jnt_range[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
            for n in ARM_JOINTS])
        self.hi = np.array([model.jnt_range[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
            for n in ARM_JOINTS])
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      "gripperframe")
        self.grip_qadr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]
        self.grip_act = 5
        self.pads = [i for i in range(model.ngeom)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                         or "").find("pad") >= 0]

    def pad_center(self, d=None):
        d = d or self.d
        if len(self.pads) < 2:
            return d.site_xpos[self.site].copy()
        return (d.geom_xpos[self.pads[0]] + d.geom_xpos[self.pads[1]]) / 2

    def pad_offset(self):
        """pad 中心相对 site 的世界偏移（当前姿态下）。"""
        return self.pad_center() - self.d.site_xpos[self.site]

    def ik(self, target, q_init=None, iters=600, lam=0.04, step=0.7,
           lock_wrist_roll=None):
        """位置 IK。

        lock_wrist_roll: 如果给了值，wrist_roll 就固定不动，
                         只用其余 4 个关节求解。

        为什么要锁 wrist_roll：
            抓取姿态要求 wrist_roll = 1.55。
            如果不锁，IK 会为了够到目标把 wrist_roll 转走，
            夹爪开口方向就歪了，抓不住轴。
            上一版是解完再覆盖 wrist_roll，那样等于破坏了 IK 解，
            残差会飙到 100+ mm（实测过）。
        """
        d = self.d
        q = np.array([d.qpos[a] for a in self.qadr]) if q_init is None \
            else np.array(q_init, dtype=float)

        # 参与求解的关节索引（锁定 wrist_roll 时排除 index 4）
        free_idx = [0, 1, 2, 3] if lock_wrist_roll is not None else [0, 1, 2, 3, 4]
        dadr_sub = [self.dadr[i] for i in free_idx]

        if lock_wrist_roll is not None:
            q[4] = lock_wrist_roll

        tmp = mujoco.MjData(self.m)
        tmp.qpos[:] = d.qpos
        for a, v in zip(self.qadr, q):
            tmp.qpos[a] = v
        mujoco.mj_forward(self.m, tmp)

        jacp = np.zeros((3, self.m.nv))
        for _ in range(iters):
            err = target - tmp.site_xpos[self.site]
            if np.linalg.norm(err) < 5e-5:
                break
            mujoco.mj_forward(self.m, tmp)
            mujoco.mj_jacSite(self.m, tmp, jacp, None, self.site)
            J = jacp[:, dadr_sub]
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), err)
            for k, idx in enumerate(free_idx):
                q[idx] = np.clip(q[idx] + step * dq[k],
                                 self.lo[idx], self.hi[idx])
            for a, v in zip(self.qadr, q):
                tmp.qpos[a] = v
        mujoco.mj_forward(self.m, tmp)
        return q, np.linalg.norm(target - tmp.site_xpos[self.site])

    def move_to(self, target_pad, n_steps=400, grip=None, wrist_roll=None):
        """把 pad 中心移到 target_pad。

        因为 IK 控的是 site，而我们要控 pad，所以先换算偏移。
        注意：偏移会随姿态变化，所以这里在【当前】姿态下取偏移，
        然后用同样的姿态去解 IK（wrist_roll 锁死），偏移就基本不变。
        """
        if grip is None:
            grip = self.d.ctrl[self.grip_act]
        if wrist_roll is not None:
            self.d.qpos[self.qadr[4]] = wrist_roll
            self.d.ctrl[4] = wrist_roll
            mujoco.mj_forward(self.m, self.d)

        off = self.pad_offset()
        site_target = np.asarray(target_pad, dtype=float) - off

        q_goal, _ = self.ik(site_target, lock_wrist_roll=wrist_roll)

        q_start = np.array([self.d.qpos[a] for a in self.qadr])
        for i in range(n_steps):
            frac = (i + 1) / n_steps
            s = frac * frac * (3 - 2 * frac)
            q = q_start + s * (q_goal - q_start)
            for a, v in zip(self.qadr, q):
                self.d.ctrl[a] = v
            self.d.ctrl[self.grip_act] = grip
            mujoco.mj_step(self.m, self.d)

        # ★ 关键：到最后再"钉"若干步，让关节真正收敛到目标。
        #   只按插值给 ctrl 是不够的——位置伺服有稳态误差和惯性滞后，
        #   实测残差会到 100 mm（因为工件被夹住后给手臂加了负载）。
        for _ in range(260):
            for a, v in zip(self.qadr, q_goal):
                self.d.ctrl[a] = v
            self.d.ctrl[self.grip_act] = grip
            mujoco.mj_step(self.m, self.d)
            # 每 4 步抓一帧，这样能看到运动过程
            if self.rend is not None and _ % 4 == 0:
                self.rend.update_scene(self.d, camera=self.cam)
                self.frames.append(self.rend.render().copy())

        return np.linalg.norm(self.pad_center() - np.asarray(target_pad))


def make_camera():
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = [0.26, 0.03, 0.10]
    c.distance = 0.95
    c.azimuth = 22
    c.elevation = -26
    return c


def main():
    print("=" * 76)
    print("完整动作：抓 D 轴 -> 搬到台钳 -> 插入 -> 松开")
    print("=" * 76)
    print()

    model, data = S.make_scene(with_vise=True, with_fixture=True, with_shaft_flat=True)

    rend = mujoco.Renderer(model, height=520, width=700)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    arm = Arm(model, data, renderer=rend)
    arm.cam = make_camera()

    shaft_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]

    # ---- 先摆到抓取姿态，让工件落稳 ----
    arm.d.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    arm.d.ctrl[4] = WRIST_ROLL_GRASP
    arm.d.ctrl[arm.grip_act] = GRIP_OPEN
    for i in range(600):
        for k in range(5):
            arm.d.ctrl[k] = arm.d.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)

    shaft_home = data.qpos[shaft_qadr:shaft_qadr + 3].copy()
    print("  工件静止位置  : %s" % np.round(shaft_home, 4))
    print("  pad 中心      : %s" % np.round(arm.pad_center(), 4))
    print("  pad 相对 site : %s" % np.round(arm.pad_offset(), 4))
    print()

    # 工件几何：沿 x 躺，半径 SHAFT_D/2
    R = S.SHAFT_D / 2
    grasp_point = np.array([shaft_home[0], shaft_home[1], shaft_home[2]])

    # 钳口目标：工件要停在这里
    jaw_c = S.jaw_center()
    JAW_X, JAW_Y, JAW_Z = jaw_c[0], jaw_c[1], jaw_c[2]

    # ---- 放置点 ----
    # 工件要座进钳口缝隙里，所以工件中心应该到 JAW_Z（钳口中心高度）。
    # 但夹爪 pad 是抱着工件的，pad 中心 = 工件中心，所以 pad 也降到 JAW_Z。
    # 下降终点再往下多给 2 mm，让工件压实（不然会浮在钳口顶面）。
    PLACE_Z = JAW_Z - 0.002

    stages = [
        ("① 移到工件上方",
         [grasp_point[0], grasp_point[1], grasp_point[2] + R + 0.075],
         550, GRIP_OPEN),
        ("② 下降到工件（pad 贴到轴两侧）",
         [grasp_point[0], grasp_point[1], grasp_point[2] + 0.002],
         400, GRIP_OPEN),
        ("③ 夹紧",
         [grasp_point[0], grasp_point[1], grasp_point[2] + 0.002],
         320, GRIP_CLOSE),
        ("④ 抬起",
         [grasp_point[0], grasp_point[1], grasp_point[2] + 0.13],
         420, GRIP_CLOSE),
        ("⑤ 横移到钳口上方",
         [JAW_X, JAW_Y, JAW_Z + 0.12],
         600, GRIP_CLOSE),
        ("⑥ 下降到钳口（工件入座）",
         [JAW_X, JAW_Y, PLACE_Z],
         520, GRIP_CLOSE),
        ("⑦ 松开",
         [JAW_X, JAW_Y, PLACE_Z],
         340, GRIP_FREE),
        ("⑧ 退开",
         [JAW_X - 0.13, JAW_Y - 0.09, JAW_Z + 0.13],
         460, GRIP_FREE),
    ]

    log = []
    for name, target, nsteps, grip in stages:
        target = np.array(target, dtype=float)
        # 分 3 轮逼近：每轮重新按当前姿态算 pad 偏移再求解，
        # 因为 pad 偏移会随关节角变化（实测第一版只算一次，
        # 结果误差累计到 143 mm）。
        err = None
        for rnd in range(3):
            err = arm.move_to(target, n_steps=max(80, nsteps // 3),
                              grip=grip, wrist_roll=WRIST_ROLL_GRASP)
            if err < 0.004:
                break
        q = data.qpos[shaft_qadr:shaft_qadr + 3].copy()
        log.append({"stage": name, "shaft": q, "grip": grip})
        print("  %-34s pad残差 %5.1f mm   工件 %s"
              % (name, (err or 0) * 1000, np.round(q, 4)))

    print()
    print("=" * 76)
    print("结果")
    print("=" * 76)
    print()
    shaft_final = data.qpos[shaft_qadr:shaft_qadr + 3]
    lift = (log[3]["shaft"] - shaft_home)[2]
    total = np.linalg.norm(shaft_final - shaft_home)

    print("  起点      : %s" % np.round(shaft_home, 4))
    print("  抬起时    : %s" % np.round(log[3]["shaft"], 4))
    print("  最终      : %s" % np.round(shaft_final, 4))
    print()
    print("  抬升高度  : %+.1f mm" % (lift * 1000))
    print("  总位移    : %.1f mm" % (total * 1000))
    print()

    ok = lift > 0.02
    print("  %s 抓取" % ("✓ 成功" if ok else "✗ 失败"))
    if ok:
        d_jaw = np.linalg.norm(shaft_final[:2] - np.array([JAW_X, JAW_Y]))
        print("  %s 搬运（最终距钳口中心 %.1f mm）"
              % ("✓ 成功" if d_jaw < 0.05 else "△ 偏了", d_jaw * 1000))

    # ---- 存帧图 ----
    print()
    print("  捕获帧数: %d" % len(arm.frames))
    if arm.frames:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cnfont import setup_cn_font
        setup_cn_font()

        n = len(arm.frames)
        idx = [int(i * (n - 1) / 7) for i in range(8)]
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
        fig.suptitle("动作序列：抓 D 轴 -> 搬到钳口 -> 插入 -> 松开",
                     fontsize=15, fontweight="bold")
        labels = ["①上方", "②下降", "③夹紧", "④抬起",
                  "⑤横移", "⑥下降", "⑦松开", "⑧退开"]
        for ax, k, lb in zip(axes.ravel(), idx, labels):
            ax.imshow(arm.frames[k])
            ax.set_title(lb, fontsize=10)
            ax.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(OUT, "demo_08_sequence.png")
        plt.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        print("  已保存 %s" % out)

    rend.close()
    print()
    print("OK")


if __name__ == "__main__":
    main()
