"""exp_pad_align.py -- 「架不住」的真正根因：两片指垫不平行，形成楔口。

测量结果（_probe_roll.py + 坐标系探查）：
  夹合方向 与轴 87-90 deg ...... 这个是对的（夹爪确实是从两侧合拢）
  fixed_jaw_pad 自身姿态 ...... 与夹爪体一致（x 朝上、z 沿轴）
  moving_jaw_pad ............ 相对 fixed 转了 26.4 deg

动指是铰链式，合拢过程中那一侧指垫会跟着转。于是两片平板不是平行的：
靠动指一侧张成 26.4 deg 的楔口。圆柱一被挤，就顺着楔口的开口方向滑出去，
同时把轴带歪 —— 这解释了为什么倾角对摩擦系数几乎不敏感（0.1/0.3/0.8 完全相同）：
不是打滑，是被楔口挤出去的。

本实验做一次受控对照：把动指指垫绕夹合轴反向转回来（只改 model.geom_quat，
不改几何、不改动作），看倾角是否塌下来。

跑法：
    D:\\robot-env\\Scripts\\python.exe exp_pad_align.py
"""
import numpy as np
import mujoco

import diag_feed_chuck as D
from demo_08_pick_and_load import Arm, GRIP_OPEN, WRIST_ROLL_GRASP
from exp_plain_cyl import one_run

GRIP = 0.17


def pad_frames(model, data):
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    out = {}
    for key, name in (("fixed", "fixed_jaw_pad"), ("moving", "moving_jaw_pad")):
        i = gid(name)
        body = model.geom_bodyid[i]
        out[key] = dict(
            gid=i, body=body,
            Rw=data.geom_xmat[i].reshape(3, 3).copy(),
            Rb=data.xmat[body].reshape(3, 3).copy(),
        )
    return out


def reach_grasp(model, data, grip=None):
    """走到「下降到位」的状态，用来量真实工作位姿。

    grip=None 时保持张开（量接近姿态）；给 0.17 时先夹紧，量**工作开度**下的
    指垫姿态——第一版把这一步漏了，在张开的 0.50 开度上做补偿，结果等于没改，
    因为动指是铰链，开度一变它自己的角度也跟着变。
    """
    arm = Arm(model, data)
    sj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
    sq = model.jnt_qposadr[sj]
    mujoco.mj_resetData(model, data)
    data.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    data.ctrl[4] = WRIST_ROLL_GRASP
    data.ctrl[arm.grip_act] = GRIP_OPEN
    for _ in range(600):
        for k in range(5):
            data.ctrl[k] = data.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)
    rest = data.qpos[sq:sq + 3].copy()
    for tgt, ns in (([rest[0], rest[1], rest[2] + 0.085], 450),
                    ([rest[0], rest[1], rest[2] + 0.002], 380)):
        arm.move_to(np.array(tgt), n_steps=ns, grip=GRIP_OPEN,
                    wrist_roll=WRIST_ROLL_GRASP)
    if grip is not None:
        arm.move_to(np.array([data.qpos[sq], data.qpos[sq + 1],
                              data.qpos[sq + 2] + 0.002]),
                    n_steps=320, grip=grip, wrist_roll=WRIST_ROLL_GRASP)
    return arm


def relative_twist(deg=True):
    """返回两片指垫的姿态差（度）。"""
    model = D.build_model()
    data = mujoco.MjData(model)
    reach_grasp(model, data)
    f = pad_frames(model, data)
    c = (np.trace(f["fixed"]["Rw"].T @ f["moving"]["Rw"]) - 1) / 2
    a = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return a if deg else np.radians(a)


def make_parallel(model, data):
    """把动指指垫的姿态对齐到定指指垫（绕自身中心转，不改位置）。"""
    f = pad_frames(model, data)
    R_des_world = f["fixed"]["Rw"]                     # 目标：与定指平行
    R_local = f["moving"]["Rb"].T @ R_des_world        # 换到动指体坐标系
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R_local).reshape(-1))
    model.geom_quat[f["moving"]["gid"]] = q
    return q


def main():
    print("=" * 78)
    print("对照：动指指垫「原样」vs「转正与定指平行」（其他一切不动）")
    print("=" * 78)

    def twist_at(grip):
        model = D.build_model()
        data = mujoco.MjData(model)
        reach_grasp(model, data, grip)
        f = pad_frames(model, data)
        c = (np.trace(f["fixed"]["Rw"].T @ f["moving"]["Rw"]) - 1) / 2
        # 指垫面的法向（局部 x）与世界 x 的夹角：理想应为 90 deg
        ax = np.array([1.0, 0.0, 0.0])
        face = np.degrees(np.arccos(np.clip(
            abs(f["moving"]["Rw"][:, 0] @ ax), -1, 1)))
        return np.degrees(np.arccos(np.clip(c, -1, 1))), face

    t_open, f_open = twist_at(None)
    t_close, f_close = twist_at(GRIP)
    print("两片指垫姿态差：张开 %.1f deg / 工作开度 %.1f deg" % (t_open, t_close))
    print("动指垫面法向与轴夹角：张开 %.1f / 工作开度 %.1f （理想 90）"
          % (f_open, f_close))
    print()

    print("  %-14s %-11s %-11s %-10s %-9s %s"
          % ("方案", "夹紧时倾角", "抬起后倾角", "抬升", "pad-轴", "判定"))
    print("  " + "-" * 72)

    for tag, fix in (("原样", False), ("动指转正", True)):
        model = D.build_model()
        data = mujoco.MjData(model)
        if fix:
            reach_grasp(model, data, GRIP)      # 关键：在工作开度上校准
            make_parallel(model, data)
        r = one_run(model, data, GRIP)
        print("  %-14s %-11s %-11s %-10s %-9s %s"
              % (tag, "%.1f deg" % r["tilt_close"], "%.1f deg" % r["tilt_lift"],
                 "%+.1f mm" % r["risen"], "%.1f mm" % r["dpad"],
                 r["verdict"]))
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
