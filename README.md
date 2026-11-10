# mujoco-robot-cell

这是一个面向学习和研究的 MuJoCo 机械臂仿真项目：SO-101 抓取圆柱棒料，将它送入三爪卡盘，并记录完整交接过程中的成功、失败和物理指标。

> 状态更新：2026-11-10。P5.8.7 在五个固定 M2 位姿估计误差下比较送料前馈角：有界 `1°` 为 2/5 PASS，`2°` 为 5/5 PASS；每个条件仅一次，不能解释为统计成功率。P5.8.8 补跑名义 M1 的 1°/2°各一次，两者也都 PASS。全局 `2°` 与 “M1=1°/M2=2°” 在受测条件上有指标取舍，不证明全局最优。P5.8.9 确认代码中的 2.0 mm 余量按 Ø20→Ø16 示例推导，仍是未经图纸确认的项目暂定预算。P5.8.10 通过支撑架随来料整体偏移，在真实重力下把 `dy=+3 mm` 保留到抓取前，随后一个 M1 真值规划案例完整送料 PASS；这验证的是输入设计单例，不是鲁棒性或感知精度。项目仍没有真机验证、已证明的整体优化收益或微调模型。

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
- schema v14 可通过 `FEED_TRACE_RELATIVE_MOTION=1` 选择性记录每个物理步的工具坐标位移、夹爪位姿和棒料接触对，并记录支撑是否随初始来料位姿变换；逐步历史默认不开启。

![送料全过程](https://github.com/penguinkinggod/mujoco-robot-cell/releases/download/media-v1/now_feed.gif)

## 现在的下一步

P5.8.10 已证明通过刚体移动支撑架可在真实重力下保留 `dy=+3 mm` 来料偏差。下一步 P5.8.11 在这个固定条件上做小型 M0/M1/M2 配对诊断，分清固定目标、真值重定位和估计误差；暂不调参或扩成大矩阵。详情见 [P5.8.10 报告](docs/p5.8.10-supported-incoming-offset-2026-11-10.md)。

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

P5.8.10 验证：Python 单测 `66/66`、`verify_all.py` `11/11`。后续验证以 [现场状态](CURRENT.md) 的最新记录为准。

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
