# 相关工作

> README 里拆出来的（2026-11-06），数字对应提交 `c3f1f9f`。

开工前查过一轮，大方向都有人做过，所以这里不装首创。专注的是最后那格没人给判据的问题。

| 方向 | 已有的东西 | 和这里的关系 |
|---|---|---|
| V 槽夹圆柱、几何自定心 | [US9808936](https://patents.google.com/patent/US9808936)、Goldberg 的 [v-grip 论文](https://goldberg.berkeley.edu/pubs/gopal-icra02.pdf) | 给出槽深约束 `H ≤ R/sinθ`：90° / Φ20 是 14.14 mm 上限 |
| 三爪卡盘软爪 | 车床行业标准做法：装爪后就地镗孔 | 这里的 5 段拼弧是它在凸包碰撞下的近似；镗孔间隙 0.2 mm 同理 |
| 形封闭理论 | [Multi-Fingered Robotic Grasping: A Primer](https://robotics.ucmerced.edu/sites/g/files/ufvvjh1576/f/page/documents/graspingprimer.pdf) | 3D 无摩擦形封闭需 ≥7 点接触；V 槽 4 线接触且轴向无约束，得补挡边 |
| SO-101 仿真 + LeRobot 采集 | [tuul-ai/so101_sim](https://github.com/tuul-ai/so101_sim)、[makolon/lerobot_sim](https://github.com/makolon/lerobot_sim) | 环境、多相机、数据集生成比这里完整；训练阶段应基于它们 |
| CNC 上下料工作站 | [intrinsic-ai/intrinsic-omts](https://github.com/intrinsic-ai/intrinsic-omts) | 工业答案是 ROS 2 + FoundationPose + MoveIt2；这里是低成本臂的接触级验证 |
| 大模型设计爪型 | [Lang2Morph](https://alphaxiv.org/abs/2509.18937) | 验证「参数化 + 仿真验证闭环」可行；它把 50+ 参数缩到每指 4 个，正是 `_v_plates()` 的思路 |
| 通用 benchmark / 现成夹爪 | [robosuite](https://github.com/ARISE-Initiative/robosuite)、[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 换爪之前先翻现成 MJCF，别从 box 拼起 |

现有项目判断「抓住了没有」，多用 task-level attachment / kinematic lock、渲染能跑通，
或端到端成功率倒推。这里给的是低摩擦 + 定量下拉 + 抬起跟走 + 逐条接触力的可复现判据，
配上装夹指标（同轴 / 倾斜 / 插入深度 / 拉拔）。这大概是值得拿出去的那部分。

调研记录、槽深扫描表和否决结论在 [`HANDOFF.md`](../../HANDOFF.md) §18。

回到 [README](../../README.md) · 全部拆分见 [索引](README.md)
