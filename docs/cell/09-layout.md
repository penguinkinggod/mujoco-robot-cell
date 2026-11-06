# 目录

> README 里拆出来的，数字截至 2026-11-06。

```
mujoco-robot-cell/
├─ verify_all.py             一键 9/9（唯一入口，刻意留在根目录）
├─ README.md  HANDOFF.md  THIRD_PARTY.md
├─ src/                      全部 Python
│  ├─ scene_machining.py       车床 + 三爪弧爪卡盘 + 工件（主线）
│  ├─ scene_full.py            SO-101 + 工作台（带开关，可关虎钳/槽台）
│  ├─ diag_feed_chuck.py       送料全流程 + 指标（核心，含 V 槽生成）
│  ├─ diag_jaw_wrap.py         爪包角 / 段数 / 力限扫描（四组数据）
│  ├─ diag_diameter_sweep.py    Φ15~Φ25 变径 × 两种镗孔方式对照
│  ├─ diag_jaw_face_sweep.py    接触面扫描：包角 / 重叠 / 厚度 + 主线力比
│  ├─ diag_chuck_health.py     卡盘体检：同步、偏心、接触、可达
│  ├─ show_current.py          离屏渲染：动图、关键帧、指标图册
│  ├─ gif_slim.py              动图去重、调色板压缩、首尾淡入淡出（20 MB → 2.3 MB）
│  ├─ release_media.py         图片传 Releases 附件：同名替换，URL 不变、不堆积
│  ├─ env_cell.py              官方 SO-101 场景里的工位 env
│  ├─ live_bridge.py           常驻 3D 窗口 + 后台推帧
│  ├─ demo_08                SO-101 基座：IK / 抓放 / 五轴轴线 IK
│  ├─ demo_09_try_versions     三个送料版本对比（完整序列 3/3）
│  ├─ demo_00 – demo_07        入门教学线
│  └─ probe_*.py  arc_jaws.py  诊断留档（arc_jaws 已废弃，留作记录）
├─ models/
│  ├─ panda/                 Franka Panda（Menagerie, Apache-2.0）
│  ├─ simready/              工作台 / 虎钳 / 工件资产
│  └─ _tmp_so101/            LeRobot SO-101 环境包（Apache-2.0）
├─ docs/                     原理笔记
├─ assets/                   本地渲染的动图，不进主干（README 里引的是 Releases 附件）
├─ tools/glb2obj             格式转换
└─ _build/                   运行时生成的图与数据（gitignore，不进仓库）
```

`src/` 现在是平铺的，没按 `core/` + `scenes/` 拆。脚本之间有 30 多处交叉 `import`，真拆
要改一堆 `sys.path`。等第二个工位落地时一起做，那时才知道 `core/` 到底该抽什么。

## 入门教学线（demo_00–07）

从「仿真器是什么」走到「怎么判断结果对不对」：

| 文件 | 教你什么 | 结果 |
|---|---|---|
| `demo_00_check.py` | `model` + `data` + `mj_step` 三件套 | 打印 `OK` |
| `demo_01_pendulum.py` | PD 为什么有静差，重力前馈补掉 | 误差 0.00° |
| `demo_02_arm.py` | 关节角 ↔ 末端位置，雅可比伪逆 | 末端误差 0.0000 m |
| `demo_03_camera_dataset.py` | 具身智能的数据长什么样 | `(图像, 关节角, 动作, 目标)` |
| `demo_04_panda_grasp.py` | 真机模型抓取：IK + 伺服 + 接触摩擦 | 抬高 0.157 m |
| `demo_05_toolpath.py` | 开环 vs 闭环，误差怎么累积 | 3D 轨迹对比 |
| `demo_06_learn_correction.py` | 机器学习和仿真的关系 | 一次实验说清 |
| `demo_07_turned_parts.py` | 车削件几何与工艺可行性 | 参数扫描 |

都在 `src/` 下，逐个跑即可，比如 `python src/demo_00_check.py`。

回到 [README](../../README.md) · 全部拆分见 [索引](README.md)
