# mujoco-robot-cell

**一个 MuJoCo 机器人工作站，也是我学习机器人仿真与具身操作的长期项目。**

这里不追求把所有东西一次做完，也不把“用了具身智能”当目标。先把一个具体任务做清楚：SO-101 夹住圆柱棒料，把它送进车床三爪卡盘；再逐步学习感知、控制和数据驱动方法。

数字截至 2026-11-07。

现在正在查什么、改到哪一步、下一步做什么，看 [CURRENT.md](CURRENT.md)。

当前工位能完成物理送料和夹紧：棒料插入卡盘 25 mm，搬运段不使用运动学锁。抓取采用按 V 槽实际开口方向侧向进入，抓取段棒料最大位移 0.109 mm（三次重复一致）。交接期用实体导套 + 端面/径向预定位先把棒料摆在轴线上，先松夹爪再闭合三爪（辅助力在负载测试前全部清零，最终夹紧由卡爪自身承担），三爪首次接触推挤 0.53 mm、交接全程位移 1.05 mm；完整送料同轴误差 0.53 mm、倾斜 1.8°、30 个卡爪接触、重力下沉 0.01 mm、20 N 拉拔位移 0.00 mm，三爪实测接触力 165.3 N。项目 9 项回归全部通过。

![送料全过程](https://github.com/penguinkinggod/mujoco-robot-cell/releases/download/media-v1/now_feed.gif)

## 30 秒跑起来

```bash
git clone https://github.com/penguinkinggod/mujoco-robot-cell.git
cd mujoco-robot-cell

uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

python verify_all.py
```

Windows / Linux / macOS 都能跑。当前验证环境是 Python 3.12、MuJoCo 3.17；完整版本见 `requirements.txt`。

要看实时画面，需要两个终端：

```bash
python src/live_bridge.py view        # 打开 3D 窗口
python src/live_bridge.py run --loop  # 运行仿真并推送画面
```

## 这次做到了什么

重做了卡盘通孔和摆位之后，重新扫描五档棒料直径、两种爪型。旧扫描把实心卡盘当成有孔，工件被穿透力顶住，旧结果不能用。

当前摆位下，Φ18 / Φ20 / Φ22 / Φ25 共八组通过；Φ15 两组都会因爪间互顶失败。详细数据和判定在 [变径与包角扫描](docs/station/06-sweeps.md)。

## 已经做到了什么

- SO-101 从台面抓起圆柱棒料，沿轴向送进三爪卡盘并完成交接。
- 有一条不使用运动学锁的全物理送料流程，核心指标可复现。
- `verify_all.py` 一次跑完 9 项检查；每项检查的意义见 [检查说明](docs/station/07-checks.md)。
- 项目目录、入门 demo 和主要脚本见 [代码地图](docs/code-map.md)。
- 关键能力阶段见 [项目里程碑](docs/milestones.md)。

第一次来可以从 [学习入口](docs/README.md) 开始；想看某个技术结论，再去 [工位资料索引](docs/station/README.md)。

## 下一步

先把现有送料工位变成一个**可优化的基准问题**：固定任务和扰动范围，选少量可调参数，用成功率、同轴误差、倾斜、拉拔位移、动作时间等指标比较不同方案。先用网格搜索或随机搜索跑出一个可信基线，不急着上神经网络。

这一步稳定后，再接**视觉引导**：让相机给出工件位姿，把视觉结果接进同一个送料流程。之后才考虑遥操作数据、模仿学习或策略优化。当前项目想做到的是一条能解释、能复现、能比较的“看见—操作—验证”闭环，不承诺通用机器人，也不把 sim-to-real 当近期门槛。

路线和阶段边界见 [项目计划](docs/project-plan.md)。

## 为什么做这个

我想用一个长期项目把机器人仿真真正学会：先看懂模型和接触，再能测量失败原因，然后把感知接进动作，最后比较规则方法与学习方法。作品集是结果，不是每一步都要包装成成果。

GitHub 是项目的公开入口；Git 保存文件版本；commit 是一个可回看、可比较的进度点。它们不需要各自再抄一份日志。怎么查历史、什么时候提交，见[Git、GitHub 和 commit 怎么用](docs/git-and-commits.md)。

## 目前的局限

- 默认回归中的部分旧检查仍使用运动学锁；全物理流程要显式运行 `python src/diag_feed_chuck.py --v-jaw --lift --no-lock`。
- 卡盘通孔由 24 片 box 近似，不能用来研究精细孔壁接触。
- 摩擦和材料参数尚未按真实设备标定；目前是可重复的仿真结果，不是真机性能声明。
- 还没有视觉引导、遥操作、模仿学习或 sim-to-real。

更多已知边界见[局限与经验](docs/station/10-limits.md)。

## 踩过的坑

- 之前把实心卡盘误当成有孔，导致变径数据看起来很好；接触力和几何必须一起核对。
- 三爪互相顶住也会产生很大的力，不能把“力大”直接当成“夹得牢”。
- 单看 IK 位置误差不够，最终动作还要看接触、滑移和姿态。

更多例子见[踩坑记录](docs/station/11-pitfalls.md)。

## 参考与声明

使用的模型、代码和许可见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。项目代码采用 MIT License，见 [`LICENSE`](LICENSE)。

正文以中文为主；关键结果优先写清楚，暂不维护整份双语文档。
