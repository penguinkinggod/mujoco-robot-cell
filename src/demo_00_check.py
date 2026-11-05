"""第 0 步：验证 MuJoCo 能在你这台电脑上跑起来并出图。

运行：
    .venv\\Scripts\\python.exe demo_00_check.py
看到 "OK" 和一张 demo_00.png 就说明环境通了。
"""
import os
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked

print("mujoco version:", mujoco.__version__)

# MuJoCo 自带几个官方模型，先拿最简单的用
MODEL_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".8 .8 .8 1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" rgba=".2 .5 .9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(MODEL_XML)
data = mujoco.MjData(model)

# 模拟 1 秒（每步 2ms，共 500 步）
for _ in range(500):
    mujoco.mj_step(model, data)

print("box 落地后的高度 z =", round(float(data.qpos[2]), 4))
print("仿真步数 nstep =", data.time)

# 尝试离屏渲染（不开窗口，直接出 png）
os.environ.setdefault("MUJOCO_GL", "glfw")
try:
    from mujoco import Renderer
    renderer = Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0, 0.4]
    cam.distance = 2.0
    cam.elevation = -20
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    try:
        from PIL import Image
        out = os.path.join(OUT, "demo_00.png")
        Image.fromarray(img).save(out)
        print("已保存", out)
    except ImportError:
        out = os.path.join(OUT, "demo_00.npy")
        np.save(out, img)
        print("已保存", out, "（装了 pillow 就能存 png）")
except Exception as e:
    print("离屏渲染没成功，但物理仿真是好的：", type(e).__name__, e)

print("OK")
