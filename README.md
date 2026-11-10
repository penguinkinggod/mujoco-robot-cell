# mujoco-robot-cell

这是一个面向学习和研究的 MuJoCo 机械臂仿真项目：SO-101 抓取圆柱棒料，将它送入三爪卡盘，并记录完整交接过程中的成功、失败和物理指标。

> 状态更新：2026-11-10。P5.8.13 诊断了名义抓取接近轨迹：接近/侧向进入期间机器人没有碰到工件，但计划夹爪姿态分别变化约 `35°/19°`；接近动作由三段纠偏组成，第一段末端计划跟踪位置差最高 `24.9 mm`。P5.8.14-P5.8.15 显示最大棒料中心位移 `3.591 mm` 早于三爪首碰，当时机器人 V 爪和导套均接触，尚不能分摊二者贡献。P5.8.16 固定导套、卡盘张开，对照夹爪保持与原地松爪各两次；全开命令仍有 4 个 V 爪接触点，故只证明模型接触力下降，不能声称机器人已脱离或导套独立承托。项目没有真机验证或微调模型。

## 目前做到哪里

- P4.4 完成了 243 组受控扰动条件：M0、M1 各 81/81 PASS；M2 为 68/81 PASS，13 个失败集中在 M2 与 25 mm 插入深度组合。结论仅适用于当时冻结的模型和测试范围。
- P4.5 完成了一项独立自由工件诊断。该诊断使用重力补偿，暂定抓稳门槛未全部通过，也没有卡盘交接；不能把它算作真实重力抓取成绩。
- P5.1 已冻结静态 V 托架上的 Ø20 × 80 mm 棒料 nominal 条件，采用真实重力、600 步等待、M1 真值规划和无运动学锁搬运。
- P5.2 对最后 100 步的动作前窗口测得端点最大位移 0.000000751 mm；两侧托架全程接触、真实重力开启、工件无外加力，稳定性按项目暂定预算 PASS。JSON 名义起始位置已从送料模型 reset 状态读取。
- 同次完整送料按现行判据 FEED_METRIC PASS：同轴误差 0.55 mm、倾角 1.5°、交接 z 峰谷 2.95 mm、余量偏心 1.42 mm。但 JSON 同时记录搬运阶段 V 爪接触对约 2.99 mm 的几何重叠，现行送料 PASS 没检查它；因此还不是已证明无穿透的成熟基线。
- P5.3 schema v11/v12 分别补充接触重叠审计和分阶段位移诊断；稳定 nominal PASS、受控抓空 ABORT、完整执行后的质量失败都有独立证据。它们是仿真诊断，不是实物校准或普适抓稳结论。
- P5.5/P5.6 固定 nominal 的端到端结果和逐步相对运动轨迹均可重复。最大相对位移 `8.151944 mm` 出现在连续搬运轨迹；P5.7 发现它会随速度和接触参数变化，但仍无实物容许值，不证明真实抓持合格。
- P5.8.5 对 5 个固定 M2 误差分别配对运行 `-0.10/-0.08`：两者同为 2/5 PASS，3 个失败均为余量偏心。候选的平均余量偏心高 `0.129 mm`、搬运相对平移高 `0.298 mm`，但交接 Z 峰谷低 `0.188 mm`；没有全面赢家。详细结果见 [P5.8.5 配对复核](docs/p5.8.5-grip-close-m2-errors-2026-11-09.md)。
- P5.8.10 验证了新的有支撑偏差输入：支撑架与工件一起整体平移，`dy=+3 mm` 在真实重力静置后仍保留 `+3.000 mm`，左右支撑各 100/100 步连续接触。随后一例 M1 真值规划完整送料通过当前判据。该样本不是估计误差或总体鲁棒性结论。详情见 [P5.8.10 报告](docs/p5.8.10-supported-incoming-offset-2026-11-10.md)。
- P5.8.11 在这个固定偏差下对照 M0/M1/M2，各一次全物理送料均 PASS。M0 下降阶段机器人-工件接触较多，至夹紧结束工件中心最大移动 `5.302 mm`、轴线变化 `6.343°`；M0 最终偏心反而较低，因此指标不是单一排名。详见 [P5.8.11 报告](docs/p5.8.11-m0-m1-m2-supported-offset-2026-11-10.md)。
- P5.8.12 将同一物理来料偏移镜像为 `dy=-3 mm`，M0/M1/M2 各跑一次全物理送料，三例也均 PASS。M0 的下降中心移动只有 `0.045 mm`，但夹紧阶段仍累计移动 `3.972 mm`；与正向单例相比，方向效应主要出现在 M0。M1/M2 的夹紧结束中心移动约 `1.64/1.35 mm`，与正向样本接近。详见 [P5.8.12 报告](docs/p5.8.12-m0-m1-m2-supported-offset-2026-11-10.md)。
- schema v18 在 `FEED_TRACE_CLAMP_HANDOVER=1` 时可选记录三爪交接阶段的工件位姿、三爪/机器人/导套/托架接触；可显式对照卡盘闭合前夹爪保持或原地松爪，并记录最后一步的命令与真实接触。此前 schema v17 的三爪接触后提前松爪诊断仍可用。`FEED_TRACE_PICKUP_MOTION=1` 和 `FEED_TRACE_RELATIVE_MOTION=1` 分别记录抓取前运动与搬运/进给相对位姿；逐步历史默认不开启，分析器兼容读取 schema v14-v18。

![送料全过程](https://github.com/penguinkinggod/mujoco-robot-cell/releases/download/media-v1/now_feed.gif)

## 现在的下一步

P5.8.13-P5.8.16 的诊断没有修改默认送料动作参数。下一卡 P5.8.17 先松爪再沿轴向撤离，分开记录撤离过程和实际无 V 爪接触后的工件状态；卡盘继续张开、导套保持闭合、外加力为零。这样才能确认当前导套条件是否能在机器人真正脱离后维持工件。详见 [P5.8.13 报告](docs/p5.8.13-free-space-arm-nod-diagnostic-2026-11-10.md)、[P5.8.14 报告](docs/p5.8.14-chuck-clamp-workpiece-diagnostic-2026-11-10.md)、[P5.8.15 报告](docs/p5.8.15-robot-release-timing-2026-11-10.md)、[P5.8.16 报告](docs/p5.8.16-pre-chuck-gripper-release-2026-11-10.md)。

- [现场状态与当前任务](CURRENT.md)
- [项目任务总账](docs/project-taskbook.md)
- [机械臂项目路线图](docs/project-plan.md)
- [论文研究路线图](docs/paper-roadmap.md)
- [P5.1 来料条件和复现命令](docs/p5.1-incoming-condition-2026-11-09.md)
- [P5.2 动作前稳定性验收](docs/p5.2-pre-motion-stability-2026-11-09.md)
- [P5.3 接触重叠审计](docs/p5.3-contact-overlap-audit-2026-11-09.md)
- [P5.5 nominal 基线复核](docs/p5.5-nominal-baseline-2026-11-09.md)
- [P5.6 搬运相对位移逐步追踪](docs/p5.6-relative-motion-trace-2026-11-09.md)
- [P4.4 实验矩阵报告](docs/p4.4-perturbation-matrix-2026-11-08.md)

## 运行与验证

项目使用 Python 3.12 与 MuJoCo 3.17。新环境可按下面安装：

    git clone https://github.com/penguinkinggod/mujoco-robot-cell.git
    cd mujoco-robot-cell
    uv venv --python 3.12 .venv
    uv pip install --python .venv/Scripts/python.exe -r requirements.txt
    python verify_all.py

P5.8.13-P5.8.16 新增可选抓取前轨迹与卡盘交接诊断记录；相关分析入口兼容读取 schema v14-v18。回归验证结果以 [现场状态](CURRENT.md) 的最新记录为准。

要看实时画面，需要两个终端：

    python src/live_bridge.py view
    python src/live_bridge.py run --loop

当前名义送料复现命令和参数见 [P5.2 验收报告](docs/p5.2-pre-motion-stability-2026-11-09.md)；初始工况定义见 [P5.1 来料条件](docs/p5.1-incoming-condition-2026-11-09.md)，目录和脚本入口见 [代码地图](docs/code-map.md)。

## 研究边界

- 现有成功和失败只对记录的模型、材料参数、夹具、控制方式和判据成立，不代表真机或行业工艺标准。
- P4.4 的 243 个结果不是尺寸边界、无限姿态覆盖，也没有证明某种优化方法优于固定方案。
- 第一篇研究先完成可信基线、文献查新和公平对照；微调模型与代理模型都是后续可选方向，不能先贴 AI 标签再找问题。
- 当前开发仅在本地分支。commit 是本地版本记录，推送 GitHub 另行决定；目前不推送。

学习入口见 [docs/README.md](docs/README.md)，各技术实验和旧过程记录见 [工位资料索引](docs/station/README.md) 与 [归档索引](docs/archive/README.md)。
