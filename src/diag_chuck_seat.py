"""diag_chuck_seat.py -- 卡盘基准：只摆料，不用机械臂，不用对中力场。

【验收边界，先读这段】
本脚本的结果只能记作「卡盘基准测试」，**不能记作机械臂送料通过**。
机械臂送料必须另跑 src/diag_feed_chuck.py --v-jaw --lift --no-lock，
由机械臂实际抓料、送入、交接才算数。

为什么要单独测
--------------
2026-11-07 用户否掉了上一版送料定案：交接期一根以「理论主轴轴线」为
目标的径向数学弹簧（SEAT_RADIAL_K）替机械臂完成了最后对中，不能算机械臂
送料成功。于是先把机械臂完全拿掉，只问卡盘自己：

    给定插入深度 + 初始偏心 + 初始倾角，三爪闭合后棒料回到轴线了吗？
    夹紧后清掉一切外力，20 N 拉拔还夹得住吗？

本脚本的诚实边界
----------------
- 不建任何指向轴线的弹簧 / 挡块 / 导套 / 托力，唯一例外是可选的纯竖直
  mg 托力（只沿世界 z，不知道轴线在哪），用于和"关重力"互相印证。
- 夹紧期默认关重力：卡爪张开内径 44 mm、棒料 Ø20，若不托住，料会在
  丝杠走完之前掉到台面上（第一版六组全部 sink 50 mm / 接触 0 就是这个
  摆法错误，不是卡盘夹不住）。关重力是全局物理设置，不参与对中。
- 三爪咬合后恢复重力、显式清零 qfrc_applied 与 xfrc_applied 并打印
  实际残值，再测下沉与拉拔，杜绝"被托成假通过"。
- 统计与棒料接触的几何名，确认机械臂没有参与。

Run:
    D:\\robot-env\\Scripts\\python.exe src/diag_chuck_seat.py
"""
import numpy as np
import mujoco

import scene_machining as S

SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])
JAW_FRONT = S.LATHE_X - S.JAW_LEN - 0.002
PULL_N = 20.0


def quat_tilt_y(deg):
    """在基准姿态上叠加绕世界 y 的小角度转动（棒料在 x-z 面内歪）。"""
    a = np.deg2rad(deg) / 2.0
    q = np.array([np.cos(a), 0.0, np.sin(a), 0.0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, SHAFT_Q0)
    return out


def run(insert_mm, tilt_deg, ecc_mm=0.0, ecc_dir="y",
        gravity_mode="off", clamp_ctrl=None):
    """gravity_mode: off = 夹紧期关重力；mg = 开重力 + 纯竖直托力。"""
    model, data, ids = S.make_scene()
    sq = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    ja = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_slide_%d" % k)] for k in range(3)]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
    bid = ids["shaft"]
    clamp_ctrl = S.CHUCK_CLAMP_CTRL if clamp_ctrl is None else clamp_ctrl
    mg = float(model.body_mass[bid] * abs(model.opt.gravity[2]))

    tip_x = JAW_FRONT + insert_mm / 1000.0
    mujoco.mj_resetData(model, data)
    ecc = ecc_mm / 1000.0
    dy = ecc if ecc_dir in ("y", "yz") else 0.0
    dz = ecc if ecc_dir in ("z", "yz") else 0.0
    data.qpos[sq:sq + 3] = [tip_x - S.SHAFT_L / 2,
                            S.LATHE_Y + dy, S.LATHE_Z + dz]
    data.qpos[sq + 3:sq + 7] = quat_tilt_y(tilt_deg)
    data.qvel[:] = 0.0
    # 机械臂一律不驱动：除卡爪丝杠外的所有执行器目标保持 0。
    data.ctrl[:] = 0.0

    g0 = model.opt.gravity.copy()
    if gravity_mode == "off":
        model.opt.gravity[:] = 0.0
    mujoco.mj_forward(model, data)

    contact_names = set()
    robot_hits = 0

    def scan_contacts():
        nonlocal robot_hits
        for i in range(data.ncon):
            c = data.contact[i]
            if gid not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == gid else c.geom1
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, other) or "?"
            contact_names.add(name)
            low = name.lower()
            if any(k in low for k in ("pad", "gripper", "finger", "arm", "link")):
                robot_hits += 1

    for i in range(500):
        t = (i + 1) / 500.0
        s = t * t * (3.0 - 2.0 * t)
        for a in ids["jaw_acts"]:
            data.ctrl[a] = S.CHUCK_OPEN_CTRL + s * (clamp_ctrl - S.CHUCK_OPEN_CTRL)
        # 纯竖直托力：只抵消自重，没有任何指向轴线的分量。
        data.xfrc_applied[:] = 0.0
        if gravity_mode == "mg":
            data.xfrc_applied[bid, 2] = mg
        mujoco.mj_step(model, data)

    # 三爪已咬合：恢复重力，清掉一切外力，静置。
    model.opt.gravity[:] = g0
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0
    for _ in range(1500):
        for a in ids["jaw_acts"]:
            data.ctrl[a] = clamp_ctrl
        mujoco.mj_step(model, data)
        scan_contacts()

    ext_max = float(max(np.abs(data.qfrc_applied).max(),
                        np.abs(data.xfrc_applied).max()))
    p = data.qpos[sq:sq + 3].copy()
    R = data.xmat[bid].reshape(3, 3)
    rod = R[:, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(rod[0]))))))
    tip = p + rod * (S.SHAFT_L / 2)
    centre_ecc = float(np.hypot(p[1] - S.LATHE_Y, p[2] - S.LATHE_Z) * 1000)
    tip_ecc = float(np.hypot(tip[1] - S.LATHE_Y, tip[2] - S.LATHE_Z) * 1000)
    sink = float((S.LATHE_Z - p[2]) * 1000)

    forces = [0.0, 0.0, 0.0]
    contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if gid not in (c.geom1, c.geom2):
            continue
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                 for g in (c.geom1, c.geom2)]
        for k in range(3):
            if any(n.startswith("jaw_geom_%d" % k) for n in names):
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f)
                forces[k] += abs(f[0])
                contacts += 1

    # 20 N 轴向拉拔：外力只在拉拔期存在，之后立刻清零。
    x0 = float(data.qpos[sq])
    data.xfrc_applied[:] = 0.0
    data.xfrc_applied[bid, :3] = [-PULL_N, 0.0, 0.0]
    for _ in range(500):
        for a in ids["jaw_acts"]:
            data.ctrl[a] = clamp_ctrl
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0.0
    pull = float(abs(data.qpos[sq] - x0) * 1000)

    jaw_q = np.array([data.qpos[a] * 1000 for a in ja])
    return dict(insert_mm=insert_mm, tilt0_deg=tilt_deg, ecc0_mm=ecc_mm,
                ecc_dir=ecc_dir, gravity_mode=gravity_mode,
                tilt_deg=tilt, centre_ecc_mm=centre_ecc, tip_ecc_mm=tip_ecc,
                sink_mm=sink, contacts=contacts, forces=forces,
                pull_mm=pull, ext_max=ext_max, robot_hits=robot_hits,
                contact_names=sorted(contact_names),
                jaw_q=jaw_q, spread=float(jaw_q.max() - jaw_q.min()))


def main():
    print("=" * 100)
    print("卡盘基准测试：只摆料、不用机械臂、不用指向轴线的力场")
    print("结果只记作『卡盘能不能夹正』，不计入机械臂送料验收")
    print("=" * 100)
    print("  卡爪前端 x=%.3f  爪长 %.0f mm  软爪半径间隙 %.2f mm  拉拔 %.0f N"
          % (JAW_FRONT, S.JAW_LEN * 1000, S.JAW_ARC_CLEAR * 1000, PULL_N))
    print("  判据参考：头端偏 <= 2.0 mm  倾斜 <= 10 deg  20N拉拔 <= 2.0 mm")
    print()
    hdr = ("  %-7s %-7s %-7s %-5s | %8s %9s %9s %8s %6s %9s %9s %8s %6s"
           % ("插入mm", "初倾°", "初偏mm", "方向", "最终倾°",
              "中心偏mm", "头端偏mm", "下沉mm", "接触", "三爪力N",
              "20N拉mm", "外力max", "臂碰"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for insert in (25, 47):
        for tilt in (0.0, 1.0, 2.0):
            for ecc in (0.0, 1.0, 2.0, 4.0):
                r = run(insert, tilt, ecc)
                ok = (r["tip_ecc_mm"] <= 2.0 and r["tilt_deg"] <= 10.0
                      and r["pull_mm"] <= 2.0 and r["ext_max"] < 1e-9
                      and r["robot_hits"] == 0)
                print("  %-7.0f %-7.1f %-7.1f %-5s | %8.2f %9.2f %9.2f "
                      "%8.3f %6d %9s %9.3f %8.1e %6d %s"
                      % (r["insert_mm"], r["tilt0_deg"], r["ecc0_mm"],
                         r["ecc_dir"], r["tilt_deg"], r["centre_ecc_mm"],
                         r["tip_ecc_mm"], r["sink_mm"], r["contacts"],
                         str(np.round(r["forces"], 1)), r["pull_mm"],
                         r["ext_max"], r["robot_hits"],
                         "OK" if ok else "X"))
        print()

    print("  对照：开重力 + 纯竖直 mg 托力（无径向对中项）")
    for ecc_dir in ("z", "yz"):
        for ecc in (1.0, 4.0):
            r = run(25, 1.0, ecc, ecc_dir=ecc_dir, gravity_mode="mg")
            print("    偏心%4.1fmm/%-2s  mg托力  最终倾%5.2f° 中心偏%6.2f "
                  "头端偏%6.2f 拉拔%6.3f 外力%.1e 臂碰%d"
                  % (ecc, ecc_dir, r["tilt_deg"], r["centre_ecc_mm"],
                     r["tip_ecc_mm"], r["pull_mm"], r["ext_max"], r["robot_hits"]))
    print()
    r = run(25, 0.0, 0.0)
    print("  与棒料接触的几何：%s" % ", ".join(r["contact_names"]))
    print()
    print("  读法：头端偏 = 爪里那段有没有夹正；中心偏 = 悬在外面那段被倾斜放大多少。")
    print("  外力max 必须是 0（静置/拉拔前已显式清零）；臂碰必须是 0。")
    print("OK")


if __name__ == "__main__":
    main()
