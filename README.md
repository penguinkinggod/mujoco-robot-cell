# mujoco-robot-cell

**一个 MuJoCo 机器人工作站，一站一站往上搭。**
*A MuJoCo robot cell, built one station at a time.*

现在跑通的是车床上下料：SO-101 抓起 Φ20 圆柱工件，送进三爪卡盘夹紧，7 项物理指标全部实测通过。

> Station 1 online — lathe tending. A SO-101 picks a Φ20 cylinder and seats it in a 3-jaw chuck.
> All 7 physics checks below were measured, not assumed.

![送料全过程](assets/now_feed.gif)

```
同轴度        1.05 mm      工件轴线 vs 卡盘轴线偏差
轴线倾斜       0.0°        夹紧后姿态
接触点           6 个       三爪各 2 点，全部真实接触
重力下沉      0.00 mm      解锁后靠自重不掉
20 N 拉拔     0.00 mm      轴向拉力下的位移
三爪法向力    21.2 N       卡盘实际夹持力
抓取倾角       2.5°        手臂交接时的姿态误差
```

`verify_all.py` **8/8 PASS，27 秒**，纯 CPU。

*Coaxiality 1.05 mm · tilt 0.0° · 6 contacts · 0.00 mm sink under gravity · 0.00 mm pull-out under 20 N · 21.2 N grip force · 2.5° grasp tilt.*

---

## 最近在做什么

> **2026-11-06** · 工件真的被抓住了：搬运段拿掉运动学锁，8/8 通过

以前 7/7 里的搬运段是把工件**钉**在手上的，夹爪其实什么都没夹住。
今天把这件事做实了，全物理跑通：

```
同轴 0.94 mm · 轴向偏 0.0° · 三爪接触 6 点
重力下沉 0.00 mm · 20 N 拉拔 0.92 mm · 三爪力 21.1 N
carry = physics（无锁）
```

`python src/diag_feed_chuck.py --v-jaw --lift --no-lock`

根因是三个叠在一起的几何错误，**每一个单独看都像"摩擦不够"**：

1. **手指闭合方向斜着搓工件。** `wrist_roll = 1.55` 下活动指行程是
   `[16.3, 25.6, 4.5] mm`——一半在顺着工件轴向推。量化后发现 1.55 有
   50% 行程沿轴，−0.95 只有 12%。改成 −0.95。
2. **两片指垫根本不在同一轴向位置。** 原装 SO-101 固定指面比活动指面
   靠后 **14.8 mm**，各自切一个 V 槽，两个槽口就沿轴错开 **15.4 mm**，
   夹爪是在"啃"圆柱不是"托"圆柱。把两个槽口统一到同一轴向工位后
   错开降到 1.9 mm，轴线偏离 20° → 4.3°。
3. **瞄准点随开合漂。** 只有一片指在动，两片指垫的中点每次开合会滑
   **21 mm**。手臂瞄准工件，闭合后爪子却落在旁边 21 mm。把两个抓取
   参考点都钉在不动的那片指上（TCP 恒定），瞄准误差归零。

还有个隐蔽的坑：中途读到的"倾斜 77°"是圆柱**绕自身轴滚转**，
不是轴线歪了——卡盘一夹就回到 0°。判据必须用轴线方向，不能用四元数
转角，否则会去修一个不存在的问题。

> **2026-11-05** · 21 次提交 · 从 6/7 修回 7/7

- 虎钳和浅槽平台从场景里删掉，工件换成纯 Φ20 圆柱
- 定位到**带载稳态下沉**：重复发同一个绝对目标是错的，改积分补偿（残差 24.8 → 1.2 mm）
- 定位到**交接时序**才是真卡点：机器人松手瞬间手臂仍在退让，会把工件一起拖走
- 卡爪 28 → 55 mm；夹紧量扫 4 档，取 `-0.010`
- 3D 窗口改成热重载，改场景不用重启
- 加了常驻窗口 + 后台推帧，跑仿真能实时看到画面
- 仓库重新分层：代码进 `src/`，生成物进 `_build/`，根目录从 70 项降到 13 项

下一步：相机标定，做视觉引导抓取（详见[路线图](#路线图)）。

---

## 这个 cell 里有四个工位

仓库不是"一个车床项目"，是一个可以不断加工位的仿真工作站。车床只是第一个做完的：

| 工位 | 状态 | 靠什么 | 目录 |
|---|---|---|---|
| **车床上下料** | ✅ 8/8 通过 | 手写 IK + 位置伺服 | `src/scene_machining.py` |
| 视觉引导抓取 | ○ 下一步 | 相机标定 + 位姿估计 | 计划 `scenes/vision_pick/` |
| 手势遥操作 | ○ 计划 | 手部关键点 → 末端映射 | 计划 `teleop/` |
| 视频示教 / 模仿学习 | ○ 计划 | BC / ACT 训练 | 计划 `learning/` |

**这四个不是四选一，是一条数据链**：视觉引导先解决"看得见"；手势遥操作解决"人怎么教"，顺带录下 `(观测, 动作)` 序列；那批序列就是模仿学习的训练集。顺序做错——比如先搞训练——会发现数据集本身是废的。

---

## 第一个工位解决了一个什么问题

真实车床上下料最难的**不是**抓起来，是**交出去**。

工件从机器人手里转到卡爪手里，中间有一个瞬间谁都不能碰它——机器人还夹着、卡爪又去夹，两个夹具抢同一根轴，输的那台伺服被拖偏（实测 6.46 mm）；机器人松手了但手还在原位，卡爪把工件往里推，会推着还张着的指垫后退（实测 12.07 mm）。

答案是：**先把工件钉死在世界坐标 → 手臂退开 55 mm → 三爪再夹紧**。

`lock()` 只固定 `qpos`，不消除接触力。卡爪推工件的力会经指垫传到手臂，手臂一退让，锚在手臂上的锁就把工件一起搬走。所以必须有一个"谁都不碰它"的瞬间。

这条不在任何教程里，是 3 次失败实测出来的。

---

## 30 秒跑起来

```bash
git clone https://github.com/penguinkinggod/mujoco-robot-cell.git
cd mujoco-robot-cell

uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

python verify_all.py
```

Windows / Linux / macOS 都能跑，**不需要显卡**，MuJoCo 纯 CPU。

实测版本：Python 3.12.15 / mujoco 3.17.0 / numpy 2.5.4 / matplotlib 3.11.2 / imageio 2.38.1。

想直接看画面：

```bash
python src/live_bridge.py view          # 常驻 3D 窗口
python src/live_bridge.py run --feed    # 后台跑送料，窗口里实时跟着动
```

---

## 7 项指标怎么验的

`verify_all.py` 一次跑完 4 个检查，任何一项不过退出码非 0：

| 检查 | 验什么 | 通过标准 |
|---|---|---|
| `diag_feed_chuck` | 送料全流程 | 同轴 ≤ 2 mm，接触 ≥ 6，20 N 拉拔不位移 |
| `diag_chuck_health` | 卡盘 IK 可达性 | 卡盘口残差 ≤ 15 mm |
| `diag_shaft_slip` | 打滑 | 夹紧后位移 ≤ 1 mm |
| `demo_09_try_versions` | 版本对比 | 3/3 完成 |

**这些数字是量出来的，不是设的。** 每个都有对应诊断脚本，可以单独跑、单独改参数。

---

## 底层是可以复用的

换工位时，下面这几层不换：

```
core/（计划）
├─ lock.py        把刚体钉在世界坐标，切断接触力传递
├─ handoff.py     两个执行器交接同一物体时的时序
├─ metrics.py     同轴度 / 接触数 / 拉拔位移 的量测
└─ camera.py      相机标定与图像采集

scene_machining   抓取 → 交接 → 夹紧 → 拉拔测试
scene_assembly    抓取 → 对齐 → 插入 → 紧固测试     （计划）
scene_vision      识别 → 定位 → 抓取 → 验证          （计划）
```

这次车床踩出来的"钉死 → 退开 → 夹紧"，装夹场景一模一样要用。**场景会换，交接逻辑不换。**

---

## 入门教学线（demo_00–07）

从"仿真器是什么"走到"怎么判断结果对不对"：

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

全部在 `src/` 下，逐个跑即可：`python src/demo_00_check.py`。生成的图落在 `_build/`，不进版本库。

---

## 踩过的坑（每条都有实测数字）

- **碰撞组掩码**：`jaw_mask` 用 `(2,3)`，机器人用 `(1,3)`，看代码像 bug。实测 MuJoCo 判定是"按位与"，掩码是对的。真因是机器人和卡盘**同时在夹同一根轴**。
- **伺服带载下沉**：原来把同一个绝对目标重复发 3 次，永远停在同一处。瞄准 (184,120,60)，指垫停短 24.8 mm，工件根本没进卡爪。改积分补偿 `tgt += 0.8 * (goal - 实测)` 后残差 24.8 → 1.2 mm。
- **卡爪长度决定抗倾覆**：28 mm 的爪夹 80 mm 的轴，轴在里面像球铰，一解锁就转到 10.9°，20 N 拉拔直接拽出 **4172 mm**。加到 55 mm 才压住。
- **夹紧量有窗口**：`-0.003` 太松（倾斜 18°），`-0.014` 太紧（把轴挤偏 7.34 mm），`-0.010` 可用。扫了 4 档。
- **摩擦不是主因**：怀疑指垫太光滑，扫了摩擦系数——不是。两片平板夹圆柱只有**线接触**，扛不住倾覆力矩。

完整工程日志在 [`HANDOFF.md`](HANDOFF.md)，935 行，含所有失败版本和当时数字。

---

## 已知限制（诚实写清楚，比吹牛有用）

1. **默认路径的搬运段仍用 `kinematic_lock`。** 8/8 里第 7 项（默认 flat pad）
   仍是钉住的，第 8 项（`--v-jaw --lift --no-lock`）才是全物理。默认值没换，
   是因为 7 项旧判据都建立在旧数字上，换默认会把历史对比打断。
2. **全物理那组是"V 槽 + μ=0.05 + 加大夹紧力"换来的。** 41 N 正压力 × 0.05
   = 2.05 N 摩擦，对上 1.93 N 自重刚好只够吊住，所以夹紧量必须给到 −0.10。
   真实车间的钢-尼龙摩擦是 0.2~0.3，这里仍偏保守。
3. **`demo_09` 的 3/3 不检查姿态。** 只验位置，工件歪着放进去也算过。
4. **没有 sim-to-real。** 无真机、无标定、无参数迁移。
5. **没有自动避障。** 路点是手给的，没有规划器。
6. **视觉、遥操作、模仿学习都还没开始。** 上面路线图里那些是计划，不是已完成。

---

## 目录（现在实际长这样）

```
mujoco-robot-cell/
├─ verify_all.py             一键 7/7（唯一入口，刻意留在根目录）
├─ README.md  HANDOFF.md  THIRD_PARTY.md
├─ src/                      全部 Python，50 个脚本
│  ├─ scene_machining.py       车床 + 三爪卡盘 + 工件（主线）
│  ├─ scene_full.py            SO-101 + 工作台（带开关，可关虎钳/槽台）
│  ├─ diag_feed_chuck.py       送料全流程 + 7 项指标（核心）
│  ├─ live_bridge.py           常驻 3D 窗口 + 后台推帧
│  ├─ demo_00 – demo_07        入门教学线（10 个）
│  └─ diag_*.py                诊断脚本（22 个，排查历史的留档）
├─ models/
│  ├─ panda/                 Franka Panda（Menagerie, Apache-2.0）
│  ├─ simready/              工作台 / 虎钳 / 工件资产
│  └─ _tmp_so101/            LeRobot SO-101 环境包（Apache-2.0）
├─ docs/                     原理笔记（8 篇）
├─ assets/                   README 用图（送料动图）
├─ tools/glb2obj             格式转换
└─ _build/                   运行时生成的图与数据（gitignore，不进仓库）
```

> `src/` 现在是**平铺**的，没按 `core/` + `scenes/` 拆——脚本之间有 30 多处交叉 `import`，真拆要改一堆 `sys.path`。等第二个工位（视觉引导）落地时一并做，那时才知道 `core/` 到底该抽什么。

---

## 为什么用 MuJoCo 而不是 Isaac Sim

| | MuJoCo | Isaac Sim |
|---|---|---|
| 安装体积 | < 300 MB | 数十 GB |
| 出第一张图 | 5 分钟 | 半天起 |
| 显卡 | 不需要 | 强依赖（官方最低 RTX 4080 / 16 GB） |

接触、摩擦、伺服误差这些**决定上下料成败的东西**，MuJoCo 几毫秒一步，笔记本 CPU 足够。等要跑大规模并行强化学习了再换，那时换的只是 API 名字。

---

## 路线图

- [x] 车床上下料闭环，7 项指标验证
- [ ] 相机标定 + 视觉引导抓取
- [ ] 手势遥操作（手部关键点 → 末端映射）
- [ ] 遥操作数据采集 + BC/ACT 训练
- [x] 换 V 块爪，把搬运段从 `kinematic_lock` 拿掉（全物理 8/8）
- [ ] 目录重构：`core/` + `scenes/` 分层
- [ ] 自动避障规划
- [ ] 变径工件（Φ15~Φ25）通用夹持

想一起做，开 issue 就行。

### 训练成果往哪发：Hugging Face

代码留在这个仓库，**权重、数据集、在线 demo 走 Hugging Face**——它的搜索是按 model / dataset 找的，没人按代码找，把 `src/` 搬过去既搜不到也白占空间。

| HF 的柜子 | 装什么 | 什么时候有 |
|---|---|---|
| **Models** | 训练好的权重 + config | 第一个策略训出来之后 |
| **Datasets** | LeRobot dataset v2（parquet + `meta/info.json` + 视频） | 遥操作采完第一批 |
| **Spaces** | Gradio 在线 demo | 有模型之后，给人点开就能跑 |

**发布门槛**——不看"训练跑通了"，看在同一个任务、同一套指标下的对比：

| 指标 | 现在手写 IK | 训练策略要赢在哪 |
|---|---|---|
| 同轴度 | 1.05 mm | 不劣于基线，或收敛更快 |
| 流程耗时 | 20.5 s | 明显更快 |
| 泛化 | 工件起始位姿写死 | **换位姿仍能成功** ← 这一条才算成果 |

前两条赢了只是复现，第三条赢了才是"模型比代码强"。

发布顺序不能反：**标定 → 采集 → 训练 → 权重上传 → 在线 demo**。标定没做好，采出来的数据集整批是废的。

> Code stays on GitHub. Weights, datasets and the interactive demo will go to Hugging Face once a trained policy actually beats the hand-written IK on *generalization* — not merely on speed or repeatability.

---

## 语言 / Language

正文用中文，**关键结论中英双写**（首屏定位、7 项指标、已知限制）。

不做全文双语的原因很实际：这个仓库每天迭代，逐句翻译会拖慢速度，而机翻质量的英文比没有英文更劝退。哪一节需要英文，开 issue 说一声。

Body text is Chinese; **key results are written in both languages**. Full bilingual docs would slow down daily iteration — if you need a section in English, open an issue and I'll add it.

---

## 相关工作 / Prior art

开工前查过一轮，**大方向都有人做过**，所以这里不装成首创；我们专注的是下面最后一格那个没人给判据的问题。

| 方向 | 已有的东西 | 和这里的关系 |
|---|---|---|
| V 槽夹圆柱、几何自定心 | [US9808936](https://patents.google.com/patent/US9808936)、Goldberg 的 [v-grip 论文](https://goldberg.berkeley.edu/pubs/gopal-icra02.pdf) | 给出槽深约束 `H ≤ R/sinθ`：90° / Φ20 → **14.14 mm 是上限**。我们曾误当成下限加到 16 mm，实测反而更差，已改回 8 mm |
| 形封闭理论 | [Multi-Fingered Robotic Grasping: A Primer](https://robotics.ucmerced.edu/sites/g/files/ufvvjh1576/f/page/documents/graspingprimer.pdf) | 3D 无摩擦形封闭需 ≥7 点接触。4 条线接触且轴向无约束 → **必须补挡边**，这不是调参能绕过的 |
| SO-101 仿真 + LeRobot 采集 | [tuul-ai/so101_sim](https://github.com/tuul-ai/so101_sim)、[makolon/lerobot_sim](https://github.com/makolon/lerobot_sim) | 环境、多相机、数据集生成比我们从零搭的完整；训练阶段应基于它们 |
| CNC 上下料工作站 | [intrinsic-ai/intrinsic-omts](https://github.com/intrinsic-ai/intrinsic-omts)（2026-09 开源，Apache-2.0） | 工业答案是 ROS 2 + FoundationPose + MoveIt2；本项目是低成本臂的接触级验证，定位不同 |
| 大模型设计爪型 | [Lang2Morph](https://alphaxiv.org/abs/2509.18937)、清华 RoboTwin 博士论文（2025） | 验证「参数化 + 仿真验证闭环」可行；Lang2Morph 把 50+ 参数缩到每指 4 个，正是 `_v_plates()` 的思路 |
| 通用 benchmark / 现成夹爪 | [robosuite](https://github.com/ARISE-Initiative/robosuite)、[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 换爪之前先翻现成 MJCF，别从 box 拼起 |

**空白**：现有项目判断「抓住了没有」，多用 task-level attachment / kinematic lock、渲染能跑通，或端到端成功率倒推。本项目 `src/check_grip_hold.py` 给的是**低摩擦 + 定量下拉 + 抬起跟走 + 逐条接触力**的可复现判据，配上 7 项装夹指标。这是值得拿出去的部分。

> 完整调研记录、槽深扫描表和否决结论见 [`HANDOFF.md`](HANDOFF.md) §18。

---

## 第三方代码与资产

- `models/panda/` — [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，Apache-2.0
- `models/_tmp_so101/lerobot_env_so101-0.4.0/` — [LeRobot](https://github.com/huggingface/lerobot) 的 SO-101 环境包，Apache-2.0
- `models/simready/` — 工作台 / 夹具 / 工件资产

详见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。

## License

代码 MIT，见 [`LICENSE`](LICENSE)。第三方部分遵循各自协议。

## 引用

```bibtex
@misc{mujoco-robot-cell,
  title  = {mujoco-robot-cell: a MuJoCo robot cell for lathe tending and manipulation},
  author = {penguinkinggod},
  year   = {2026},
  note   = {First station: SO-101 seats a Φ20 cylinder in a 3-jaw chuck, verified by 7 physics checks},
  url    = {https://github.com/penguinkinggod/mujoco-robot-cell}
}
```
