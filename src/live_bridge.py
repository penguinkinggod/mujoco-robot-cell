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
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
STATE = os.path.join(OUT, "live_state.npy")
TMP = os.path.join(OUT, "_live_tmp.npy")   # 必须以 .npy 结尾，否则 numpy 会自动补后缀

# 画面里放哪一套场景。view 和 run 必须用同一套，否则同轴、接触这些
# 指标是对着 V 槽算的，画面上却是平板爪 —— 看着抓住了，其实不是那只手。
SCENE = dict(v_jaw=True, lift=True)
RUN_ARGS = dict(carry_lock=False)      # 搬运段不用运动学锁，靠真实接触

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


PUSH_STATS = {"ok": 0, "fail": 0}


def push(qpos):
    """Atomically publish one qpos frame, tolerating Windows file locks.

    A single fixed temp name can still be held open by a reader or AV scan;
    np.save() then raises PermissionError and aborts the whole run. Rotate
    four temp names and catch the whole save+replace transaction, so a busy
    file costs one frame instead of killing the simulation.
    """
    slot = (PUSH_STATS["ok"] + PUSH_STATS["fail"]) % 4
    tmp = TMP.replace(".npy", ".%d.npy" % slot)
    try:
        np.save(tmp, qpos)
        os.replace(tmp, STATE)
        PUSH_STATS["ok"] += 1
    except OSError:
        PUSH_STATS["fail"] += 1
        try:
            os.remove(tmp)
        except OSError:
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
            model = Dm.build_model(**SCENE)
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
                time.sleep(0.01)
        if not stale:
            print("VIEW_CLOSED", flush=True)
            return 0
        cam = cam_now
        time.sleep(0.6)      # 等我改完文件，别在保存到一半时重编译
        print("VIEW_RELOAD scene source changed -> recompiling", flush=True)


def run(loop=False):
    """跑仿真并把每帧关节值推给窗口。

    没有这个进程，view 那个窗口只会停在初始姿态上 —— 它就是一只
    「等着收帧」的手，没人推就一直不动。桌面 bat 两个都要起。
    """
    frames = [0]

    def hook(tag, _m, d):
        push(d.qpos.copy())
        frames[0] += 1
        # The view process polls the state file at ~100 Hz. A long sleep here
        # used to throttle the whole playback to a visibly jerky ~15 fps.
        time.sleep(0.002)

    while True:
        try:
            # dt=0.002 s, so 10 steps = ~50 fps. The old 20-step stride plus
            # a 15 ms hook sleep made the motion look like stop-motion.
            D.FRAME_EVERY = 10          # reload 会把模块常量还原成 60
            model = D.build_model(**SCENE)
            data = mujoco.MjData(model)
            rc = D.main(frame_hook=hook, model=model, data=data,
                        **SCENE, **RUN_ARGS)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print("RUN_ERROR %s: %s" % (type(exc).__name__, exc), flush=True)
            rc = 1
        ok, bad = PUSH_STATS["ok"], PUSH_STATS["fail"]
        print("PUSH_DONE frames=%d ok=%d fail=%d (丢帧 %.1f%%) rc=%d"
              % (frames[0], ok, bad, 100.0 * bad / max(1, ok + bad), rc),
              flush=True)
        if not loop:
            return rc
        time.sleep(1.5)         # 喘口气再跑一遍，窗口不会一下停住
        for name in RELOAD_ORDER:      # 循环时热点改代码也能生效
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        globals()["D"] = sys.modules["diag_feed_chuck"]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "view"
    if mode == "view":
        raise SystemExit(view())
    if mode == "run":
        raise SystemExit(run(loop="--loop" in sys.argv))
    print("usage: live_bridge.py view|run [--loop]")
    raise SystemExit(2)
