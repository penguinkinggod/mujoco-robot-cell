# 代码地图

这是找代码的入口，不是每个脚本的全量清单。脚本有新增、删除时，先看名字和用途，再用 `verify_all.py` 确认。

## 项目根目录

```text
README.md             项目首页：当前状态、如何运行、学习方向
verify_all.py         一键回归检查
requirements.txt      Python 依赖
THIRD_PARTY.md        第三方模型、代码和许可证说明
src/                  场景、控制、诊断、教学 demo
models/               机械臂和工件模型
assets/               需要提交的静态展示资源；GIF 由 Release 托管
_build/               运行时生成物，不提交
```

## 从哪段代码开始

- 想看场景怎么拼：`src/scene_full.py`、`src/scene_machining.py`
- 想看完整送料流程：`src/diag_feed_chuck.py`
- 想看卡盘几何和接触：`src/scene_machining.py`、`src/diag_chuck_health.py`
- 想看完整送料的插入深度扫描：`src/diag_feed_insert_sweep.py`（输出 `_build/feed_insert_sweep.csv`）
- 想看卡盘单站变径与卡爪接触面扫描：`src/diag_diameter_sweep.py`、`src/diag_jaw_face_sweep.py`
- 想看一键验收：`verify_all.py` 和 [检查说明](station/07-checks.md)
- 想看最小入门程序：`src/demo_00_check.py` 到 `src/demo_07_turned_parts.py`

`src/` 目前按用途平铺，诊断脚本和实验脚本都保留在原处。不要为了目录看起来整齐就大规模移动；等第二个实际工位出现，再根据重复的代码决定是否抽公共层。

## 生成物放哪

图片、CSV、临时模型和渲染结果统一放 `_build/`。它们可由脚本重建，不是唯一资料。只有要展示的静态图才放 `assets/`；首屏 GIF 由 GitHub Release 附件托管。

## 入门教学线

```text
src/demo_00_check.py       model / data / mj_step
src/demo_01_pendulum.py   PD、重力前馈和静差
src/demo_02_arm.py        末端位置、雅可比和 IK
src/demo_03_camera_dataset.py 观测、动作和数据集
src/demo_04_panda_grasp.py     IK、伺服和接触
src/demo_05_toolpath.py        开环与闭环
src/demo_06_learn_correction.py 仿真与学习
src/demo_07_turned_parts.py    车削件参数扫描
```

这些 demo 是学习线路，不等于当前车床工位的全部实现。当前工位主要看 `scene_*`、`diag_*` 和 `verify_all.py`。
