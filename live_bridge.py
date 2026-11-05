"""live_bridge.py -- 常驻 3D 窗口 + 后台仿真实时推送。

两个独立进程：
    python live_bridge.py view   # 常驻 3D 窗口，读 live_state.npy 刷新画面
    python live_bridge.py run    # 跑送料仿真，每帧把 qpos 推给窗口

两种改动，两种生效方式（这是 MuJoCo 的硬限制，不是偷懒）：

1) 只改「姿势」-- 关节角、轨迹、参数。run 把 qpos 推进 live_state.npy，
   view 每帧读进来 mj_forward + sync。窗口不重启，机械臂当场跟着动。

2) 改「结构」-- 增删零件、换工件形状。qpos 只是关节值，画面上有哪些
   物体是由「编译后的模型」决定的；Mujoco 的 viewer 一旦启动就持有
   那份模型，改不了它的 geom 列表。所以这里做了自动热重载：view 每轮
   都在比对场景源码的 mtime/size，一变就重新 import → 重新编译 →
   自动重开窗口，并把你当前的视角原样搬过去。你不用手动重启。

状态文件用原子替换，避免读到半帧；重载时旧的 qpos 会被丢掉，防止把
老模型的关节值套到新模型上。
"""
import importlib
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

# 这些文件一变，说明「场景结构」变了，需要重编译模型。
WATCH = ("scene_full.py", "scene_machining.py",
         "demo_08_pick_and_load.py", "diag_feed_chuck.py")
RELOAD_ORDER = ("scene_full", "scene_machining",
                "demo_08_pick_and_load", "diag_feed_chuck")


def model_signature():
    """场景源码的指纹。mtime + 大小，够用且不用读文件内容。"""
    out = []
    for name in WATCH:
        try:
            st = os.stat(os.path.join(HERE, name))
            out.append((name, int(st.st_mtime), st.st_size))
        except OSError:
            out.append((name, 0, 0))
    return tuple(out)


def reload_modules():
    """按依赖顺序重新 import，让本进程看到磁盘上的最新场景代码。"""
    for name in RELOAD_ORDER:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    return sys.modules["diag_feed_chuck"]


def push(qpos):
    np.save(TMP, qpos)
    try:
        os.replace(TMP, STATE)
    except OSError:
        # Windows 上窗口正在读这一帧时替换会被拒绝；跳过这帧，下一帧再写。
        pass


def _cam_of(v):
    """抓当前视角，重开窗口后原样搬过去。"""
    return (v.cam.lookat.copy(), float(v.cam.distance),
            float(v.cam.azimuth), float(v.cam.elevation))


def _apply_cam(v, cam):
    if cam is None:
        return
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.lookat[:] = cam[0]
    v.cam.distance = cam[1]
    v.cam.azimuth = cam[2]
    v.cam.elevation = cam[3]


def view():
    cam = None
    while True:
        sig = model_signature()
        try:
            Dm = reload_modules()
            model = Dm.build_model()
        except Exception as exc:                      # 改到一半的语法错误
            print("VIEW_RELOAD_WAIT %s: %s" % (type(exc).__name__, exc),
                  flush=True)
            time.sleep(1.0)
            continue
        data = mujoco.MjData(model)
        if os.path.exists(STATE):
            try:
                os.remove(STATE)      # 旧结构的帧不能套到新模型上
            except OSError:
                pass
        print("VIEW_READY ngeom=%d nbody=%d nq=%d reload=%d"
              % (model.ngeom, model.nbody, model.nq, 0 if cam is None else 1),
              flush=True)

        last = None
        stale = False
        cam_now = None
        with viewer.launch_passive(model, data) as v:
            _apply_cam(v, cam)
            while v.is_running():
                if model_signature() != sig:
                    stale = True
                    cam_now = _cam_of(v)
                    break
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
        if not stale:
            print("VIEW_CLOSED", flush=True)
            return 0
        cam = cam_now
        time.sleep(0.6)      # 等我改完文件，别在保存到一半时重编译
        print("VIEW_RELOAD scene source changed -> recompiling", flush=True)


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
