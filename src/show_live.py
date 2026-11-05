"""show_live.py -- 打开一个可以拖着转的真 3D 窗口，看当前场景。

在 D:\\codex\\projects\\robot-sim-first-step 下运行：
    D:\\robot-env\\Scripts\\python.exe show_live.py
        起始态：卡盘张开、工件在台面。可拖动旋转 / 滚轮缩放。
    D:\\robot-env\\Scripts\\python.exe show_live.py --feed
        在窗口里实时播放「抓取 → 送料 → 夹紧 → 退出」全过程。
    D:\\robot-env\\Scripts\\python.exe show_live.py --clamped
        先跑完送料，再打开窗口，停在装夹完成的状态慢慢看。

窗口右侧 UI 可以打开接触点 / 接触力 / 坐标系显示，
这是看「三爪到底有没有同时压住轴」最直接的办法。
"""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
from mujoco import viewer

import diag_feed_chuck as D
import scene_machining as SM
from demo_08_pick_and_load import Arm, WRIST_ROLL_GRASP, GRIP_OPEN


def warmup(model, data):
    """摆好抓取姿态并让工件落稳，和送料脚本的 reset 完全一致。"""
    arm = Arm(model, data)
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
    data.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    data.ctrl[4] = WRIST_ROLL_GRASP
    data.ctrl[arm.grip_act] = GRIP_OPEN
    data.ctrl[jaw] = SM.CHUCK_OPEN_CTRL
    for _ in range(600):
        for k in range(5):
            data.ctrl[k] = data.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)
    return arm


def main():
    args = set(sys.argv[1:])
    unknown = args - {"--feed", "--clamped"}
    if unknown:
        print("未知参数：%s" % ", ".join(sorted(unknown)))
        return 2
    if "--feed" in args and "--clamped" in args:
        print("--feed 和 --clamped 不能一起用")
        return 2

    model = D.build_model()
    data = mujoco.MjData(model)
    mode = "--feed 实时播放" if "--feed" in args else (
        "--clamped 装夹完成态" if "--clamped" in args else "起始态")
    print("打开 3D 窗口：%s（关掉窗口即退出）" % mode)

    if "--feed" in args:
        warmup(model, data)
        with viewer.launch_passive(model, data) as v:
            steps = [0]

            def hook(tag, _m, _d):
                if not v.is_running():
                    raise KeyboardInterrupt
                steps[0] += 1
                if tag:
                    print("  " + tag)
                v.sync()
                time.sleep(0.012)

            try:
                D.main(frame_hook=hook, model=model, data=data)
            except KeyboardInterrupt:
                print("窗口已关闭，提前结束。")
            else:
                print("送料结束，窗口保留在最终状态，关掉窗口退出。")
                while v.is_running():
                    v.sync()
                    time.sleep(0.03)
        return 0

    if "--clamped" in args:
        D.main(model=model, data=data)
    else:
        warmup(model, data)
    viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
