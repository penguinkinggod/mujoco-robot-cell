"""diag_camera_convention.py —— 摸清 MuJoCo 相机的 azimuth/elevation 到底怎么算。

为什么必须先测
--------------
我连续三版都把相机方向搞反了（看黑屏、钻进卡盘里、跑到台面下面）。
MuJoCo 的文档只说了"方位角/仰角"，没给公式，凭感觉写就是撞运气。

这个脚本直接问 MuJoCo：给定 azimuth/elevation，相机站在哪、看向哪。
以后所有相机都查这张表，不再猜。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_camera_convention.py
"""
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
d = mujoco.MjData(m)
# 用 Renderer 而不是裸 MjrContext：后者要手动建 GL 上下文，容易 gladLoadGL 报错。
# Renderer 会把 scene/option/context 都准备好。
rend = mujoco.Renderer(m, height=64, width=64)
scn = rend.scene
opt = mujoco.MjvOption()

print("=" * 74)
print("MuJoCo 相机方位角约定（实测）")
print("=" * 74)
print("  lookat = [0, 0, 0]，distance = 1.0")
print()
print("  %-9s %-10s %-26s %-20s" % ("azimuth", "elevation", "相机位置", "看向"))
print("  " + "-" * 70)

for az in (0, 45, 90, 135, 180, 225, 270):
    for el in (0, -30):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0.0, 0.0, 0.0]
        cam.distance = 1.0
        cam.azimuth = az
        cam.elevation = el
        rend.update_scene(d, camera=cam)
        fwd = np.array(cam.lookat) - np.array(scn.camera[0].pos)
        fwd = fwd / np.linalg.norm(fwd)
        print("  %-9d %-10d %-26s %s"
              % (az, el, np.round(scn.camera[0].pos, 3), np.round(fwd, 2)))

rend.close()

print()
print("=" * 74)
print("结论（按实测数据，不是按文档猜的）")
print("=" * 74)
print("""
  从上面的表能读出真实约定：

      azimuth = 0    -> 相机在 -x 方向
      azimuth = 90   -> 相机在 -y 方向
      azimuth = 180  -> 相机在 +x 方向
      azimuth = 270  -> 相机在 +y 方向
      也就是 azimuth 增大 = 相机绕 lookat 点逆时针（从上往下看）

      elevation < 0  -> 相机在上方，俯视（表里 z = +0.5）
      elevation > 0  -> 相机在下方，仰视

  我连续三版搞错相机的原因不是算错，是没查这张表就凭感觉写。

  场景用法：
      机器人底座在原点、朝 +x 干活
      -> 总览要站在 -x 后方，azimuth ≈ 0~30
      -> elevation ≈ -25（从上往下看）
""")
print("OK")
