# 学习据点

项目当前状态以 [CURRENT.md](../CURRENT.md) 和[项目任务总账](project-taskbook.md)为准。带日期的验收报告是当时的实验快照，其中“下一步”可能已经被后续工作覆盖。

这个仓库按三条线走：

1. **先学仿真**：看懂模型、控制、运动学、接触和数据。
2. **再做一个具体工位**：SO-101 抓取棒料并送进车床卡盘。
3. **最后接入学习方法**：视觉、遥操作、模仿学习。前一层没站稳，不急着训练。

## 从哪里开始

| 想做什么 | 先看什么 |
|---|---|
| 第一次接触 MuJoCo | [01-what-is-simulator.md](learn/01-what-is-simulator.md) |
| 想知道为什么“算对了还会错” | [06-why-numerically-exact-still-fails.md](learn/06-why-numerically-exact-still-fails.md) |
| 想按顺序学基础 | [01](learn/01-what-is-simulator.md) → [02](learn/02-control-and-steady-state-error.md) → [03](learn/03-kinematics-and-jacobian.md) → [04](learn/04-embodied-data.md) → [05](learn/05-grasp-pipeline.md) |
| 想看当前车床工位 | [项目定位](project-position.md) → [项目计划](project-plan.md) |
| 想把项目做成论文 | [论文路线图](paper-roadmap.md) |
| 想看现在正在改什么、下一步做什么 | [现场记录](../CURRENT.md) |
| 想看项目已经跨过哪些阶段 | [项目里程碑](milestones.md) |
| 想看棒料与 V 爪接触重叠的时空审计 | [P5.3 接触审计](p5.3-contact-overlap-audit-2026-11-09.md) |
| 想看代码和脚本放哪 | [代码地图](code-map.md) |
| 想查每个送料指标怎么测、阈值能不能改 | [机械臂送料指标字典](metric-dictionary.md) |
| 想看 Git、GitHub、commit 怎么用 | [版本与提交](git-and-commits.md) |
| 想查实验数据 | [工位资料索引](station/README.md) |
| 想看相关研究 | [相关工作](research/related-work.md) |
| 想看旧过程和失败版本 | [归档](archive/README.md) |

## 当前边界

这里不是大创申报书，也不是把每一次小改动都写成报告。仓库只保留：当前项目状态、能帮助下次继续的学习笔记、能复现的实验和重要失败原因。

大段历史不放在首屏；需要时从 [归档](archive/README.md) 或 Git 历史回看。
