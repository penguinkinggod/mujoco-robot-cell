"""live_bridge.py -- 常驻 3D 窗口 + 后台仿真实时推送。

两个独立进程，窗口不用重启：
    python live_bridge.py view   # 常驻 3D 窗口，读 live_state.npy 刷新画面
    python live_bridge.py run    # 跑送料仿真，每帧把 qpos 推给窗口

我这边每次重跑 run（改参数 / 改代码 / 换动作），画面就跟着新仿真动，
窗口一直开着，直到手动关闭。状态文件用原子替换，避免读到半帧。
"""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import mujoco
from mujoco import viewer

import diag_feed_chuck as D

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "live_state.npy")
TMP = os.path.join(HERE, "_live_tmp.npy")   # 必须以 .npy 结尾，否则 numpy 会自动补后缀


def push(qpos):
    np.save(TMP, qpos)
    try:
        os.replace(TMP, STATE)
    except OSError:
        # Windows 上窗口正在读这一帧时替换会被拒绝；跳过这帧，下一帧再写。
        pass


def view():
    model = D.build_model()
    data = mujoco.MjData(model)
    if os.path.exists(STATE):
        os.remove(STATE)
    print("VIEW_READY", flush=True)
    last = None
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            if os.path.exists(STATE):
                try:
                    q = np.load(STATE)
                    if q.shape == data.qpos.shape and (
                            last is None or not np.array_equal(q, last)):
                        data.qpos[:] = q
                        mujoco.mj_forward(model, data)
                        v.sync()
                        last = q
                except (OSError, ValueError):
                    pass        # 正在原子替换，下一帧再读
            time.sleep(0.02)
    print("VIEW_CLOSED", flush=True)
    return 0


def run():
    D.FRAME_EVERY = 20          # 每 20 步推一帧，画面连续不跳
    model = D.build_model()
    data = mujoco.MjData(model)
    frames = [0]

    def hook(tag, _m, d):
        push(d.qpos.copy())
        frames[0] += 1
        time.sleep(0.015)

    rc = D.main(frame_hook=hook, model=model, data=data)
    print("PUSH_DONE frames=%d rc=%d" % (frames[0], rc), flush=True)
    return rc


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "view"
    if mode == "view":
        raise SystemExit(view())
    if mode == "run":
        raise SystemExit(run())
    print("usage: live_bridge.py view|run")
    raise SystemExit(2)
