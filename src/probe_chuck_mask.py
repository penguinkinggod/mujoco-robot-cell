"""probe_chuck_mask.py -- 卡盘到底谁在挡路。

三个问题一次量清楚：
  1. 卡盘各部件（本体/床头箱/主轴/三爪）当前的碰撞掩码是什么，
     和机器人、工件各自碰不碰。
  2. 送料全流程里，机器人有没有真的撞过卡盘任何部件（现在应该 0 次）。
  3. 如果把卡盘本体改成"挡机器人、不挡工件"，这条送料路径会不会被卡住。

用法：
    python src/probe_chuck_mask.py            # 只看掩码
    python src/probe_chuck_mask.py --run      # 跑一遍送料，统计接触
"""
import argparse
import sys

import numpy as np
import mujoco

import scene_full as F
import diag_feed_chuck as D
from demo_08_pick_and_load import Arm, GRIP_OPEN

MACHINE = ("chuck_body", "headstock", "spindle")


def collide(ct1, ca1, ct2, ca2):
    """MuJoCo 的判定：两边 contype/conaffinity 有共同位就碰。"""
    return bool((ca1 & ct2) or (ca2 & ct1))


def dump_masks(m):
    print("=" * 66)
    print("A. 掩码现状（contype, conaffinity）")
    print("=" * 66)
    print("  约定：CT_ROBOT=%s  CT_WORK=%s  CT_FIXTURE=%s  CT_STATIC=%s"
          % (F.CT_ROBOT, F.CT_WORK, F.CT_FIXTURE, F.CT_STATIC))
    print()
    print("  %-18s %-10s %-9s %-9s %-9s" % ("geom", "mask", "vs 机器人",
                                           "vs 工件", "vs 卡爪"))
    groups = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                 m.geom_bodyid[g]) or ""
        key = None
        if name in MACHINE:
            key = name
        elif name.startswith("jaw_geom"):
            key = "jaw_geom_*"
        if key is None:
            continue
        groups.setdefault(key, (int(m.geom_contype[g]),
                                int(m.geom_conaffinity[g])))
    jaw = groups.get("jaw_geom_*", (0, 0))
    for key in MACHINE + ("jaw_geom_*",):
        if key not in groups:
            print("  %-18s <缺失>" % key)
            continue
        ct, ca = groups[key]
        print("  %-18s (%d,%d)     %-9s %-9s %-9s" % (
            key, ct, ca,
            "碰" if collide(ct, ca, *F.CT_ROBOT) else "不碰",
            "碰" if collide(ct, ca, *F.CT_WORK) else "不碰",
            "碰" if collide(ct, ca, *jaw) else "不碰"))
    print()
    print("  结论：标'不碰 vs 机器人'的部件，机器人可以整块穿过去。")
    return groups


def run_flow(m, verbose=True):
    """跑一遍全物理送料，统计机器人 vs 卡盘各部件的接触。"""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    machine = set()
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n in MACHINE or n.startswith("jaw_geom"):
            machine.add(g)
    robot = set(g for g in range(m.ngeom)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                     m.geom_bodyid[g]) in
                ("gripper", "moving_jaw", "fixed_jaw", "wrist", "lower_arm",
                 "upper_arm", "shoulder", "base"))

    hits = {}
    deepest = [0.0, ""]

    def watch(tag, _m, dd):
        for i in range(dd.ncon):
            c = dd.contact[i]
            a, b = c.geom1, c.geom2
            if not ((a in machine) or (b in machine)):
                continue
            na = mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_GEOM, a) or ("geom%d" % a)
            nb = mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_GEOM, b) or ("geom%d" % b)
            ba = mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_BODY,
                                   _m.geom_bodyid[a]) or "?"
            bb = mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_BODY,
                                   _m.geom_bodyid[b]) or "?"
            k = "%s(%s) x %s(%s)" % (na, ba, nb, bb)
            hits[k] = hits.get(k, 0) + 1
            if c.dist < deepest[0]:
                deepest[0] = c.dist
                deepest[1] = k

    D.FRAME_EVERY = 40
    rc = D.main(frame_hook=watch, model=m, data=d,
                v_jaw=True, lift=True, carry_lock=False)

    print()
    print("=" * 66)
    print("C. 送料过程中「机器人/工件 vs 卡盘部件」的接触")
    print("=" * 66)
    if not hits:
        print("  0 次接触 —— 机器人和工件全程没碰过卡盘任何部件。")
        print("  这就是穿模：不是穿过去了，是物理上根本不存在这张面。")
    else:
        for k, v in sorted(hits.items(), key=lambda x: -x[1]):
            print("  %-46s %d 帧" % (k, v))
        print("  最深穿透 %.2f mm：%s" % (-deepest[0] * 1000, deepest[1]))
    return rc, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="跑一遍送料统计接触")
    a = ap.parse_args()

    m = D.build_model(v_jaw=True, lift=True)
    dump_masks(m)
    if a.run:
        run_flow(m)


if __name__ == "__main__":
    main()
