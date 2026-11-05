# 原理笔记

每一步一张纸，讲清"这段代码在干什么、为什么这么写、你改哪一行会看到什么"。
不需要先学完 Python，看现象 → 改参数 → 再看现象，就是最快的学法。

| 文件 | 对应 demo |
|---|---|
| `01-what-is-simulator.md` | `demo_00_check.py` |
| `02-control-and-steady-state-error.md` | `demo_01_pendulum.py` |
| `03-kinematics-and-jacobian.md` | `demo_02_arm.py` |
| `04-embodied-data.md` | `demo_03_camera_dataset.py` |
| `05-grasp-pipeline.md` | `demo_04_panda_grasp.py` |
| `06-why-numerically-exact-still-fails.md` | 四层误差诊断（IK 0.5 mm → 实际 38 mm） |
| `07-community-learning-paths.md` | 社区学习路径调研（别人怎么走的） |

## 推荐阅读顺序

如果你只想读两篇：**先 `01`，再 `06`**。

`01` 告诉你仿真器到底在算什么；`06` 告诉你为什么算对了还会错——这两篇合起来是整个仓库的地基，其他都是它的应用。

然后按 demo 顺序走 `02` → `05`。`07` 不用按顺序读，等你有「我是不是走偏了」的疑问时再看。
