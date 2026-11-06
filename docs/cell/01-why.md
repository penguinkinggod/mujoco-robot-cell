# 为什么做这个

> README 里拆出来的，数字截至 2026-11-06。

想做一个能逐日迭代的地方，而不是又一份一次性 demo。每天学的东西有地方落，指标能复现，
失败也记下来。做久了，才可能拿出「泛化」上的东西：换个位姿也能成，而不只是复现固定
路线。

## 这个仓库不重造仿真器

2026-11-06 查过一轮，结论放前面：仿真基础设施别人已经做完了，这部分不重做。

- Google Intrinsic 在 ROSCon 2026（9 月 22 日）开源了 `intrinsic-core` + `intrinsic-omts`
  （Open Machine Tending Solution，CNC 机床上下料参考设计，支持 FANUC / UR），
  Apache-2.0，含 FoundationPose 位姿估计、MoveIt2 规划、Gazebo 数字孪生、手眼标定。
- LeRobot 官方就有 SO-101 的 MuJoCo 环境（`lerobot/so101-tabletop-mujoco`），30 fps 关节
  角动作 + 种子随机化 + front / top / wrist_cam 三个机位。
- 「抓取到底成不成立」的判据在灵巧抓取文献里是标准项（simulation displacement、sliding
  ratio、penetration depth、Ferrari-Canny Q1），GraspIt! 二十年前就有完整实现。

所以这里只留三件别人不做的：

| 留下的 | 为什么是我们 |
|---|---|
| 具体工位：SO-101 + Φ20 圆柱 + 三爪车床卡盘 | 基准任务都是桌面方块，没有卡盘、没有轴向进给 |
| 几何归因：为什么这个夹爪在这个工件上会失败 | 指垫先天错开 14.8 mm 这种，方块任务里永远碰不到 |
| 能跑的判据：真夹住 / 假夹住，一键验 | 文献指标是离线评估手的，不是给二指臂工位用的门槛 |

场景直接搬官方的（`build_scene(official_world=True)`），自己只留工位和判据。臂模型没换：
`so101_new_calib.xml` 带 `fixed_jaw_pad` / `moving_jaw_pad`，整套 V 槽几何建在上面，
官方那份没有这两块。

## 为什么用 MuJoCo 不用 Isaac Sim

| | MuJoCo | Isaac Sim |
|---|---|---|
| 安装体积 | < 300 MB | 数十 GB |
| 出第一张图 | 5 分钟 | 半天起 |
| 显卡 | 不需要 | 强依赖 |

接触、摩擦、伺服误差这些决定上下料成败的东西，MuJoCo 几毫秒一步，笔记本 CPU 就够。
等要跑大规模并行强化学习了再换，那时换的只是 API 名字。

回到 [README](../../README.md) · 全部拆分见 [索引](README.md)
