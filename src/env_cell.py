"""env_cell.py -- 车床上下料工位，跑在官方 SO-101 tabletop 场景里。

为什么有这个文件
================

仓库里原来的 `diag_feed_chuck.py` 是一条**写死的脚本**：固定起点、固定路线、
跑完打印一行 FEED_METRIC。它把物理做实了（8/8，搬运段无运动学锁），但它
没法回答"工件换个位置还成不成"，也没法出数据集、接视觉。

这些事官方已经定好约定了 —— `lerobot/so101-tabletop-mujoco`：

    关节用**角度制**，夹爪用 **0-100 开合**，30 fps 绝对位置目标
    `reset(rng)` 按种子随机化（关节噪声 / 物体位姿 / 相机位置）
    画面走**场景里命名的相机**：front、top、wrist_cam

这个文件把我们的车床工位塞进那套约定，而不是另发明一套。搬过来的只有
**场景**（80x80 cm 台面、房间、三个相机）；机械臂模型和 V 槽爪**不换**，
因为我们的 `so101_new_calib.xml` 带 `fixed_jaw_pad` / `moving_jaw_pad`，
整个 V 槽几何都建在它上面，官方那份没有这两块。

台面高度对得上：官方 tabletop 的台面就是 z = 0，本项目所有位姿也标定到
z = 0，所以 IK 目标、卡盘原点、V 槽几何**一个数都不用改**。

它不替代旧路径
==============

`python src/diag_feed_chuck.py --v-jaw --lift --no-lock` 仍是物理基线（8/8）。
这个文件是**同一套物理的第二个入口**，多出来的是标准接口。

跑法
====

    python src/env_cell.py                 # 建场景、随机 3 个工件位姿、出图
    python src/env_cell.py --run           # 在官方场景里跑完整送料，出 gif
    python src/env_cell.py --seeds 8       # 多随机几组
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import diag_feed_chuck as D          # noqa: E402
import scene_full as F               # noqa: E402

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ModuleNotFoundError:          # 没装也能用，只是不能 gym.make
    gym = None
    spaces = None
    _GYM = False


# 官方 env 的关节顺序，单位和它一致：前五个是度，第六个是 0-100。
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")
FPS = 30

# 卡盘轴线（和 diag_feed_chuck 里的 axis 同一个点，别各写一份）
CHUCK_AXIS_POINT = np.array([D.LATHE_ORIGIN[0] - 0.016,
                             D.LATHE_ORIGIN[1],
                             D.LATHE_ORIGIN[2]])

# 工件随机化范围。沿轴（x）可以大一些：V 托架是沿 x 的一条长槽，滑一段
# 还在槽里；横向（y）只给 2 mm，因为托架 V 深 8 mm，给多了工件会掉出槽，
# 那不是"随机化"而是"换了个初始条件"。
SHAFT_JITTER_X = 0.008
SHAFT_JITTER_Y = 0.002


class CellEnv(gym.Env if _GYM else object):
    """车床上下料工位。接口照官方 SO-101 env，物理仍是我们标定过的那套。

    动作：6 维，[肩 pan, 肩 lift, 肘, 腕 flex, 腕 roll] 用度，夹爪 0-100。
          **绝对位置目标**，不是增量。
    观测：`state` 同样是 6 维，单位与动作一致（读的是 qpos，不是 ctrl）。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(self, v_jaw=True, lift=True, width=480, height=360,
                 render_mode="rgb_array"):
        self.width, self.height, self.render_mode = width, height, render_mode
        self.model = D.build_model(v_jaw=v_jaw, lift=lift, official_world=True)
        self.data = mujoco.MjData(self.model)

        # --- 动作/观测空间：直接从模型的关节量程推，不手抄数字 ---
        self.qadr = np.array([
            self.model.jnt_qposadr[self.model.joint(j).id] for j in JOINTS])
        # 这个工位比官方 tabletop 多一个执行器：卡盘的三爪（jaw_act）。
        # 所以 ctrl 是 7 维，而动作只有 6 维 —— 只能写手臂那 6 个。
        self.act = np.array([self.model.actuator(j).id for j in JOINTS])
        rng_lo, rng_hi = [], []
        for j in JOINTS:
            r = self.model.jnt_range[self.model.joint(j).id]
            rng_lo.append(np.rad2deg(r[0]))
            rng_hi.append(np.rad2deg(r[1]))
        self.lo = np.deg2rad(np.array(rng_lo))
        self.hi = np.deg2rad(np.array(rng_hi))
        # 夹爪那一路对外是 0-100，内部是弧度
        self.grip_lo, self.grip_hi = self.lo[5], self.hi[5]

        if _GYM:
            deg_lo = np.rad2deg(self.lo[:5])
            deg_hi = np.rad2deg(self.hi[:5])
            a_lo = np.concatenate([deg_lo, [0.0]])
            a_hi = np.concatenate([deg_hi, [100.0]])
            self.action_space = spaces.Box(a_lo, a_hi, dtype=np.float64)
            self.observation_space = spaces.Dict(
                {"state": spaces.Box(a_lo, a_hi, dtype=np.float32)})

        # 每个 step 内部跑多少物理步。**不改 model.opt.timestep**：
        # diag_feed_chuck 里所有 move() 的步数都是按原生 timestep 标定的，
        # 按官方 env 那样把 timestep 重设成 1/(fps*substeps) 会把 8/8 直接跑挂。
        self.substeps = max(1, round(1.0 / (FPS * self.model.opt.timestep)))
        self._renderer = None

        self.shaft_body = self.model.body("shaft").id
        self.shaft_qadr = self.model.jnt_qposadr[
            self.model.joint("shaft_free").id]
        self.shaft_home = self.data.qpos[self.shaft_qadr:self.shaft_qadr + 3].copy()
        mujoco.mj_forward(self.model, self.data)
        self.shaft_home = self.data.qpos[
            self.shaft_qadr:self.shaft_qadr + 3].copy()

    # ---------- 单位换算（官方 env 的语义） ----------
    def to_ctrl(self, values):
        """[度 x5, 夹爪 0-100] -> 执行器目标（弧度）。"""
        v = np.asarray(values, float)
        ctrl = np.deg2rad(v[:6].copy())
        ctrl[5] = self.grip_lo + v[5] / 100.0 * (self.grip_hi - self.grip_lo)
        return np.clip(ctrl, self.lo, self.hi)

    def joint_state(self):
        """qpos -> [度 x5, 夹爪 0-100]，与动作同单位。"""
        q = self.data.qpos[self.qadr].copy()
        out = np.rad2deg(q)
        out[5] = (q[5] - self.grip_lo) / (self.grip_hi - self.grip_lo) * 100.0
        return out.astype(np.float32)

    # ---------- 复位 ----------
    def reset(self, rng=None, shaft_dx=0.0, shaft_dy=0.0, settle=200):
        """回到初始状态，可选给工件一个种子偏移。

        settle 步先让工件在 V 托架上落稳 —— 直接改 qpos 会让它悬在槽口
        上方几毫米，一放就砸下来，量到的"位姿"是砸完的，不是给的。
        """
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos[self.shaft_qadr:self.shaft_qadr + 7].copy()
        q[0] += shaft_dx
        q[1] += shaft_dy
        self.data.qpos[self.shaft_qadr:self.shaft_qadr + 7] = q
        # 手臂先回到张开、停住的姿态，否则第一帧就在动。
        # 卡盘执行器保持模型默认值，不在这里动它。
        self.data.ctrl[self.act] = np.concatenate(
            [self.data.qpos[self.qadr[:5]], [self.grip_hi]])
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle):
            mujoco.mj_step(self.model, self.data)
        obs = {"state": self.joint_state()}
        return (obs, {}) if _GYM else obs

    def sample_reset(self, rng):
        return self.reset(rng,
                          shaft_dx=float(rng.uniform(-SHAFT_JITTER_X, SHAFT_JITTER_X)),
                          shaft_dy=float(rng.uniform(-SHAFT_JITTER_Y, SHAFT_JITTER_Y)))

    # ---------- 一步 ----------
    def step(self, action):
        self.data.ctrl[self.act] = self.to_ctrl(action)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        obs = {"state": self.joint_state()}
        info = self.cell_metrics()
        if _GYM:
            return obs, 0.0, False, False, info
        return obs, 0.0, False, info

    # ---------- 出图 ----------
    def render(self, camera="cell"):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height,
                                             width=self.width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    @property
    def cameras(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(self.model.ncam)]

    # ---------- 工位指标（复用项目里的判据语义） ----------
    def cell_metrics(self):
        """同轴度 / 轴线偏离 / 三爪接触 —— 和 diag_feed_chuck 同一套定义。

        同轴度用的是**横向距离**，轴线偏离用的是**轴向夹角**：圆柱在 V 槽里
        滚转会让四元数转角变成几十度，但轴线根本没动，那是假报警。
        """
        p = self.data.qpos[self.shaft_qadr:self.shaft_qadr + 3]
        coax = float(np.hypot(p[1] - CHUCK_AXIS_POINT[1],
                              p[2] - CHUCK_AXIS_POINT[2]) * 1000)
        ax = self.data.xmat[self.shaft_body].reshape(3, 3)[:, 2]
        axis_dev = float(np.degrees(np.arccos(min(1.0, abs(float(ax[0]))))))

        shaft_geoms = {g for g in range(self.model.ngeom)
                       if (mujoco.mj_id2name(self.model,
                                             mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                       .startswith("shaft_")}
        contacts, force = 0, 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 not in shaft_geoms and c.geom2 not in shaft_geoms:
                continue
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if "jaw_geom" not in (n1 + "|" + n2):
                continue
            contacts += 1
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            force += abs(f6[0])
        return {"coax_mm": coax, "axis_dev_deg": axis_dev,
                "jaw_contacts": contacts, "jaw_force_N": force,
                "shaft_xyz": p.copy()}


def _seed_demo(n=3):
    from PIL import Image
    env = CellEnv()
    print("官方场景就绪：%d 个相机 %s" % (env.model.ncam, env.cameras))
    print("动作空间: 5 x 度 + 夹爪 0-100；timestep %.4f s，每步 %d 物理子步"
          % (env.model.opt.timestep, env.substeps))
    tiles = []
    for i in range(n):
        rng = np.random.default_rng(1000 + i)
        env.sample_reset(rng)
        m = env.cell_metrics()
        print("  seed %d  工件 xyz=%s  同轴=%6.2f mm  轴向偏=%4.1f deg  三爪接触=%d"
              % (1000 + i, np.round(m["shaft_xyz"], 4), m["coax_mm"],
                 m["axis_dev_deg"], m["jaw_contacts"]))
        tiles.append(Image.fromarray(env.render("cell")))
    sheet = Image.new("RGB", (480 * len(tiles), 360))
    for i, t in enumerate(tiles):
        sheet.paste(t, (480 * i, 0))
    out = os.path.join(ROOT, "_build", "env_cell_seeds.png")
    sheet.save(out)
    print("拼图 -> %s" % out)
    return 0


def _make_hook(env):
    """录 cell 机位。默认 FRAME_EVERY=60 太稀，录出来只有 30 多帧，一跳一跳。"""
    from PIL import Image
    D.FRAME_EVERY = 20
    frames, calls = [], [0]

    def hook(tag, model, data):
        calls[0] += 1
        if calls[0] % 4:
            return
        frames.append(Image.fromarray(env.render("cell")))

    hook.frames = frames
    return hook


def _run_demo(no_video=False):
    """在官方场景里跑完整送料，顺手录 cell 机位。"""
    env = CellEnv()
    print("CAMS %d %s" % (env.model.ncam, env.cameras))
    hook = None if no_video else _make_hook(env)
    rc = D.main(frame_hook=hook, model=env.model, data=env.data,
                v_jaw=True, lift=True, carry_lock=False)
    if hook is not None and hook.frames:
        out = os.path.join(ROOT, "_build", "cell_official.gif")
        hook.frames[0].save(out, save_all=True, append_images=hook.frames[1:],
                            duration=70, loop=0, optimize=True)
        print("gif -> %s (%d 帧)" % (out, len(hook.frames)))
    return rc


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="车床工位（官方 SO-101 场景）")
    ap.add_argument("--run", action="store_true", help="在官方场景里跑完整送料")
    ap.add_argument("--no-video", action="store_true", help="不录 gif（回归检查用）")
    ap.add_argument("--seeds", type=int, default=3, help="随机几组工件位姿")
    a = ap.parse_args()
    raise SystemExit(_run_demo(a.no_video) if a.run else _seed_demo(a.seeds))
