"""move_arm.py -- 用最短的一段 Python 让机械臂在常驻 3D 窗口里动一下。

准备：双击桌面「机器人仿真 常驻窗口.bat」（窗口已经开着就跳过）。
运行：D:\\robot-env\\Scripts\\python.exe move_arm.py

想换动作，只改下面 4 个数字：JOINTS / AMP / CYCLES / SPEED。

原理 —— 「编程让机械臂动」本质上就这三步：
    1. 把关节角度写进 data.qpos[关节地址]
    2. mujoco.mj_forward(model, data) 更新所有连杆的位置
    3. push(data.qpos) 把这一帧发给常驻窗口，窗口 v.sync() 画出来
循环起来就是动画。这里是运动学摆动，故意不跑物理：工件、卡盘、
7/7 回归基线全都不受影响。想跑带物理的完整抓取送料，用
live_bridge.py run（那个才是真仿真）。
"""
import math
import time

import numpy as np
import mujoco

import diag_feed_chuck as D
from demo_08_pick_and_load import Arm, ARM_JOINTS
from live_bridge import push

# ---- 只改这些数字就能换动作 --------------------------------------------
JOINTS = [0, 1, 2]      # 动哪几个关节：0 肩转 / 1 肩抬 / 2 肘（共 5 个）
AMP = 0.55              # 摆幅，占该关节可用行程的比例（0 ~ 1）
CYCLES = 3              # 来回摆几次
SPEED = 0.025           # 每帧停顿秒数，越大越慢
FRAMES = 360            # 总帧数
# ------------------------------------------------------------------------


def main():
    model = D.build_model()
    data = mujoco.MjData(model)
    arm = Arm(model, data)

    lo = np.asarray(arm.lo, dtype=float)
    hi = np.asarray(arm.hi, dtype=float)
    span = hi - lo
    mid = (lo + hi) / 2
    # 没有限位（range 为 0）的关节不动，避免除零 / 超程
    half = np.where(span > 1e-6, span / 2 * AMP, 0.0)

    print("MOVE_START joints=%s frames=%d" %
          ([ARM_JOINTS[j] for j in JOINTS], FRAMES), flush=True)
    for f in range(FRAMES):
        s = math.sin(2 * math.pi * CYCLES * f / FRAMES)
        for j in JOINTS:
            data.qpos[arm.qadr[j]] = mid[j] + half[j] * s
        mujoco.mj_forward(model, data)
        push(data.qpos.copy())
        time.sleep(SPEED)
    print("MOVE_DONE frames=%d" % FRAMES, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
