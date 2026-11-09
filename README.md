# mujoco-robot-cell

这是一个面向学习和研究的 MuJoCo 机械臂仿真项目：SO-101 抓取圆柱棒料，将它送入三爪卡盘，并记录完整交接过程中的成功、失败和物理指标。

> 状态更新：2026-11-09。P5.2 已验证一个真实重力 nominal 的动作前稳定性并修正结果记录；当前机械主线进入 P5.3。项目还没有真机、视觉定位、参数优化优胜结果或微调模型。

## 目前做到哪里

- P4.4 完成了 243 组受控扰动条件：M0、M1 各 81/81 PASS；M2 为 68/81 PASS，13 个失败集中在 M2 与 25 mm 插入深度组合。结论仅适用于当时冻结的模型和测试范围。
- P4.5 完成了一项独立自由工件诊断。该诊断使用重力补偿，暂定抓稳门槛未全部通过，也没有卡盘交接；不能把它算作真实重力抓取成绩。
- P5.1 已冻结静态 V 托架上的 Ø20 × 80 mm 棒料 nominal 条件，采用真实重力、600 步等待、M1 真值规划和无运动学锁搬运。
- P5.2 对最后 100 步的动作前窗口测得端点最大位移 0.000000751 mm；两侧托架全程接触、真实重力开启、工件无外加力，稳定性按项目暂定预算 PASS。JSON 名义起始位置已从送料模型 reset 状态读取。
- 同次完整送料按现行判据 FEED_METRIC PASS：同轴误差 0.55 mm、倾角 1.5°、交接 z 峰谷 2.95 mm、余量偏心 1.42 mm。但 JSON 同时记录搬运阶段 V 爪接触对约 2.99 mm 的几何重叠，现行送料 PASS 没检查它；因此还不是已证明无穿透的成熟基线。

![送料全过程](https://github.com/penguinkinggod/mujoco-robot-cell/releases/download/media-v1/now_feed.gif)

## 现在的下一步

只做 P5.3：先复核搬运阶段约 2.99 mm 的棒料/V 爪重叠，再按接近推料、夹持保持、搬运滑移和卡盘接管分阶段验收。暂不扩成姿态大扫描，不训练模型。

- [现场状态与当前任务](CURRENT.md)
- [项目任务总账](docs/project-taskbook.md)
- [机械臂项目路线图](docs/project-plan.md)
- [论文研究路线图](docs/paper-roadmap.md)
- [P5.1 来料条件和复现命令](docs/p5.1-incoming-condition-2026-11-09.md)
- [P5.2 动作前稳定性验收](docs/p5.2-pre-motion-stability-2026-11-09.md)
- [P4.4 实验矩阵报告](docs/p4.4-perturbation-matrix-2026-11-08.md)

## 运行与验证

项目使用 Python 3.12 与 MuJoCo 3.17。新环境可按下面安装：

    git clone https://github.com/penguinkinggod/mujoco-robot-cell.git
    cd mujoco-robot-cell
    uv venv --python 3.12 .venv
    uv pip install --python .venv/Scripts/python.exe -r requirements.txt
    python verify_all.py

P5.2 相关单测为 32/32。完整回归以 [现场状态](CURRENT.md) 的最新记录为准。

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
