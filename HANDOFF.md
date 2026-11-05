# 交接文档 — SO-101 机床上下料仿真

> 最后更新：2026-11-05（阶段一完成 + 本地 git 基线 + 一键回归，见下方状态更新）
> 交接原因（历史）：夹具自定心连续 8 次失败，暂停交回；该问题已由路线 B 解决。
> 读者：**不写代码**，靠指挥 AI 推进。

---

## ✅ 状态更新（2026-11-05，路线 B 已执行完）

**§1 第 3 件那个决定已经做了：走路线 B，放弃自定心。阶段一全部验收通过。**

| 项 | 结果 |
|---|---|
| `diag_slot_alone.py` | **13/13 SEATED** → `fixture works` |
| `demo_09_try_versions.py` | **3/3**（抬升 92.8 mm / 搬运误差 1.7 mm / 入槽且漂移 0.0 mm）|
| `demo_09_sequence.gif` | 已生成，150 帧 / 9 秒 |
| `fixture_profile.png` | 已按新槽重画，可直接看 y-z 剖面 |
| `verify_all.py` | **7/7 PASS**（约 18 秒），一条命令验全局，解析真实数字而非只看退出码 |
| 本地 git 基线 | 首次提交已完成，**未连 GitHub**；`git status` / `git log` 可查 |
| `diag_chuck_health.py` | **PASS**：偏心 0.00 mm / 倾斜 0.00° / 6 接触点 / 送入位姿 IK 9.07 mm |
| 卡盘位置 | `LATHE_Y` 0.24 → **0.12**；可达性实测 13.40 → 9.07 mm（原来够不着） |
| **三爪真同步** | **已实现**：1 个丝杠执行器 + 2 条 `mjEQ_JOINT` 联动约束。位移差 **4.24 → 0.118 mm**、各爪力比 4× → **1.57**、20 N 拉不动。判据已写进 `verify_all.py`（位移差 ≤ 0.2 mm、力比 ≤ 2.0） |
| **联动约束语义** | **已验证**：`add_equality().data` 默认 `[0,1,0,0,0,...]` 即 `q2=q1`，编译后 `eq obj 7→8 / 7→9`；空载扫伺服三爪真跟随（4.000/4.009/4.009 mm）。结构 + 行为双重断言已写进 `diag_chuck_health.py` |
| `diag_feed_chuck.py` | **PASS**（2026-11-05）：机器人送料 → 夹紧 → 退出全闭环，同轴 0.89 mm / 倾斜 5.9° / 11 接触 / 18.7 N / 20 N 拉 0.62 mm，已挂 `verify_all.CHECKS`。**含两层临时简化**：D 铣扁是粗糙碰撞代理、搬运段用运动学锁（假设夹爪完美形封闭），详见文件头与 §13 |

**夹具最终参数（和 §10 的建议有一处不同）**：槽宽 **24 mm**、槽深 **10 mm**（原建议 3 mm）。
3 mm 壁顶比轴心低 7 mm，那个高度上圆柱截面半宽只有 7.45 mm，够不到 12 mm 的半槽宽，工件会在平底上一直滚（实测 2/13）。壁加到与轴心齐平后，圆柱被限制在 ±2 mm 内，13/13。仍是**直壁平底、零自定心**。

**验收口径也改了**（原「≥10/13」测的是 90 mm 自由落体 + 0~60 mm 偏移的自定心，路线 B 下物理上不可能）：
现在是"机器人实际放置"工况 —— 落料高 5 mm、y 偏移 0~2.4 mm 共 13 点，判据 `|dy| ≤ 2.5 mm 且 |dz| ≤ 1.5 mm`。补了 z 判据，避免"骑在槽沿上"被误判成功。

**顺带修掉 3 个原文没写的隐藏 bug**（跑起来才暴露）：
1. `shaft_flat` 尺寸轴写反，"铣平面"其实是一块 80 mm 高的竖板，会穿透桌面。已改成沿轴长的扁条，并设为纯视觉 —— box 角点会凸出圆柱包络约 3 mm，先着地就会把轴顶偏。
2. `GRIP_CLOSE = -0.08` 让 pad 中心距只剩 8.7 mm，比 Φ20 轴还窄，夹爪把轴挤飞、抬升 0.0 mm。按 `diag_grasp_geometry.py` 实测曲线（间距 = 14.8 + 75.8×ctrl）改成 **0.15**，抬升 92.8 mm。
3. `demo_09` 每一步都重新标定"抓取偏移"，把下降途中的滑动算了进去，放置点越抬越高；改回夹紧后只标定一次。同时修正：`lift` 取 `trace[4]`（④b 抬起）而非 `trace[3]`（还没离地）、⑥ 必须夹着下降到位再松爪、搬运判据按 §2 用 ⑤ 阶段的 xy 误差（≤5 mm）而不是最终落点。

> ⚠️ 项目已迁到 `D:\codex\projects\robot-sim-first-step`。旧目录 `D:\robot-sim-first-step` 完整保留作备份，**别在旧目录跑脚本**。下面 §8 / §9 / §10 的夹具描述是历史记录，当前实现以本段为准。

---

## 0. 一句话现状

**场景能跑、机器人能抓、能抬起来、能搬运。卡在最后一步：把工件放进夹具。**

| 能力 | 状态 | 证据 |
|---|---|---|
| 场景搭建（真实 SimReady 资产 + 真 SO-101） | ✅ | `scene_full.py` 编译，ngeom=42 nmesh=15 |
| 逆运动学（IK）精度 | ✅ | 5/5 测试点，残差 < 0.3 mm |
| 抓取工件 | ✅ | `demo_08_sequence.png`，抬升 117.7 mm |
| 搬运到夹具上方 | ✅ | 同上 |
| **工件放进夹具** | ✅ 已解决（路线 B） | 放置判据通过，松开后漂移 0.0 mm |
| 严格三判据（抓/搬/放） | ✅ | **3/3**，`demo_09_result.png` + `demo_09_sequence.gif` |

---

## 1. 你要干什么

你是**指挥**，不是打字。下面每一件事都是"对 AI 说一句话 + 用眼睛验收"。

### 第 1 件：先看懂现在有什么（10 分钟，不动代码）

```powershell
Set-Location D:\codex\projects\robot-sim-first-step
$env:MUJOCO_GL="glfw"; $env:PYTHONUTF8="1"
$py = "D:\robot-env\Scripts\python.exe"
& $py scene_full.py
```

然后打开 `D:\codex\projects\robot-sim-first-step\scene_full.png`，看三张小图。

**你要看出这三样东西**：
1. 左边全景里，白色机械臂（6 个关节）站在桌边，中间有个灰色台钳
2. 中间特写里，夹具是**两根银色小圆柱**并排，中间有个 V 形凹口
3. 右边是工件：一根 Φ20 的银灰色短轴，上面铣平了一小块

看不懂就问 AI："`scene_full.png` 里哪个是夹具，哪个是工件，指给我。"

### 第 2 件：亲手复现失败（5 分钟）

```powershell
& $py diag_slot_alone.py
```

看最后一行。
- 如果写 `fixture works` → 已经好了，跳到第 4 件
- 如果写 `fixture still fails` → 正常，这就是要你解决的问题

**这一步的意义**：你要亲眼看到"2/13"这个数字，而不是听我说。这是你自己判断 AI 有没有在糊弄你的唯一手段。

### 第 3 件：做决定（最重要，AI 替不了你）

文档 §10 给了三条路线。**我推荐路线 B：放弃自定心。**

当时要对 AI 说的原话（**历史记录，已执行完，别再照着敲**）：

> 按 HANDOFF.md 的路线 B 走。把 scene_full.py 里的双托辊夹具换成一块带浅槽的方台：
> 槽宽 24 mm（工件 Φ20，留 4 mm 容错），槽深 3 mm。
> 改完先跑 diag_slot_alone.py，把里面的判据从 ±6 mm 放宽到 ±8 mm，
> 要看到 ≥ 10/13，再跑 demo_09_try_versions.py。

实际落地与这段指令有**三处不同**，以顶部状态更新和 §2 为准：槽深改成 **10 mm**（3 mm 壁顶够不到圆柱，实测只有 2/13）；判据改成 `|dy| ≤ 2.5 mm 且 |dz| ≤ 1.5 mm`（不是放宽到 ±8 mm，反而收紧并补了 z 判据）；落料工况改成"轻放 5 mm 高、偏移 0~2.4 mm"，结果 **13/13**。

AI 如果说"我建议继续调托辊"，你就说：

> 已经调了 8 代了，收益递减。走路线 B。

**这个决定为什么该你做**：这不是技术问题，是**需求问题**。自定心是给"手把工件扔进去"的场景用的。你有 0.3 mm 精度的机器人，压根不需要夹具帮你纠偏。AI 容易陷在"把当前方案修好"的惯性里，跳不出来。

### 第 4 件：验收（要看得见运动）

```powershell
& $py demo_09_try_versions.py
```

要看到 `3/3`，并生成 `demo_09_result.png`（四视图）。

然后对 AI 说：

> 把 demo_09 的整段序列导成 GIF，我要能看见抓→抬→搬→放→松 的完整动作。

GIF 生成后自己播一遍，**用眼睛确认三件事**：
1. 夹爪真的夹住了轴（不是穿模）
2. 轴真的被抬离桌面
3. 轴在夹具上方松开后真的落进槽里，没弹飞

### 第 5 件（之后）：往下推

✅ 本地 `git init` + 首次提交已完成（2026-11-05，**未连 GitHub**）。改任何场景前先跑：

```powershell
& "D:\robot-env\Scripts\python.exe" verify_all.py          # 全量，约 15 秒
& "D:\robot-env\Scripts\python.exe" verify_all.py --quick  # 只查语法/编译，1 秒
```

两条场景线**已经合并**（`add_lathe(spec)`），机器人送料装夹**已经跑通**（`diag_feed_chuck.py` PASS，见 §13）。当前唯一未解决的物理缺口是**夹爪抓持姿态**：搬运段暂时用运动学锁假设完美形封闭，真实夹爪在抓取阶段就把轴搓转 41.5°。下一件事是修它，然后删锁。

---

## 2. 做到什么算完成

### 阶段一 — ✅ 已完成（2026-11-05）

| 判据 | 怎么量 | 通过线 |
|---|---|---|
| 夹具能用 | `diag_slot_alone.py` 输出 | **13/13 SEATED** |
| 三判据全过 | `demo_09_try_versions.py` 输出 | **3/3** |
| 看得见运动 | `demo_09_sequence.gif` | 150 帧 / 9 秒，动作连续无跳变 |
| 没穿模 | 自己看 GIF | 轴始终在夹爪两片之间 |
| 整体没退化 | `verify_all.py` 输出 | **7/7 PASS** |

**量化定义**（别用"差不多"，以下为路线 B 后的当前口径）：
- **夹具**：落料高 5 mm、y 偏移 0~2.4 mm 共 13 点；每点 `|dy| ≤ 2.5 mm 且 |dz| ≤ 1.5 mm`（补了 z 判据，避免"骑在槽沿上"被误判）
- **抓取成功**：轴心抬升 ≥ 50 mm 且夹爪与轴保持接触
- **搬运成功**：到达夹具上方（⑤ 阶段）时轴心 xy 误差 ≤ 5 mm
- **放置成功**：落进 24 mm 槽宽、z 在槽内，松爪后漂移 ≤ 10 mm

> 老口径「≥10/13 CENTERED、90 mm 自由落体 + 0~60 mm 偏移」测的是自定心，路线 B 下物理上不成立，已废弃。

### 阶段二（下一步）

- [x] **合并两条场景线**：已抽成 `scene_machining.add_lathe(spec)`，由 `scene_full.build_scene(with_lathe=True)` 调用；默认关闭，`demo_09` 3/3 与 7/7 回归均不退化
- [x] **机器人送料装夹闭环**：`diag_feed_chuck.py` PASS，判据同轴 ≤ 2 mm / 倾斜 ≤ 10° / 6 接触 / 重力不打滑 / 20 N 拉不出，实测 **0.89 mm / 5.9° / 11 接触 / 0.01 mm / 0.62 mm**，已挂 `verify_all.CHECKS`
- [ ] **修抓取姿态 + 删运动学锁**：送料流程里搬运段仍是运动学锁，假设夹爪完美形封闭；真实抓取阶段轴已倾斜 41.5°（无锁时 98.2°）。`demo_09` 的 3/3 从不检查姿态，这个隐藏缺陷还没真正修好
- [ ] **精修 D 铣扁几何**：现在只是把 box 碰撞打开，角点凸出 D20 包络 3.1 mm；后续换成不凸出的真实 D 形并全局同步给 `scene_full.py`（须重跑 `diag_slot_alone.py` / `demo_09`）
- ~~三爪同步~~ **已完成**：根因是三个爪的 box 没按 120° 旋转（有效径向半厚 9.0/12.3/12.3 mm），不是驱动问题；修几何 + 单丝杠联动后位移差 0.118 mm，见 §13
- reset 循环 + 20 次随机初始位姿 + 成功率统计

### 阶段三（月级，对齐你的职业目标）

- 抓取策略从硬编码 IK 换学习方法
- `mujoco-mjx` GPU 并行加速
- 开源发布。**稀缺的原创件是数控车床 + 卡盘，不是抓取代码**（§12 有理由）

---

## 3. 你能在里面学到什么

零基础没关系，下面每个知识点都**挂在你能看见的画面上**，不用先学理论。

### 阶段一会逼你搞懂的四件事

| 你会看到的现象 | 背后的知识点 | 学到什么程度算够 |
|---|---|---|
| 夹具偏移 5 mm 就弹飞 | **碰撞检测**：两个物体的"位掩码"决定碰不碰 | 能解释为什么"夹具永远不和机器人碰"是故意设的，不是 bug（§7） |
| 斜面 45° 但工件卡住不动 | **摩擦角**：`atan(μ)` 必须小于斜面角 | 能自己算出 μ=1.2 时摩擦角 50.2° > 45°，所以卡住（§8 规律一） |
| 机器人手伸太远就抖 | **臂展与精度**：末端误差随伸展放大 | 能解释台钳为什么必须从 x=0.30 挪到 x=0.20 |
| 明明算好了位置却差 284 mm | **逆运动学**：要从末端位置反推 6 个关节角 | 知道"算位置"和"算雅可比"是两件事，而且顺序有讲究（§11） |

### 阶段二会接触的

- **坐标系与四元数**：为什么台钳要"绕 Z 转 90°"而不是"绕 Y 转 90°"——我当初就是靠**渲染四个方向用眼睛挑**才对，不是靠推理
- **自由度与约束**：三爪卡盘怎么用一个关节驱动三个爪（现已落地：1 个 position servo + 2 条 `mjEQ_JOINT`，`scene_machining.py` 有完整实测注释）
- **状态机**：抓→抬→搬→放→松 是一个状态序列，每步有进入条件和退出判据

### 阶段三会接触的

- **域随机化**：每次重置随机化工件位置、质量、摩擦，让策略不脆弱
- **成功率统计**：跑 100 次，看 92 次成功，这比"我看着像是对的"可靠得多
- **GPU 并行**：为什么 4096 个环境同时跑比 1 个环境跑 4096 次快 100 倍

### 一个反直觉但很重要的认知（我从这 8 轮失败里学到的）

**"把当前方案修好"和"这个方案对不对"是两个问题。**

我连续 8 代都在优化夹具形状：直角台阶→斜面→纯 V→梯形→阶梯→托辊。
每代都有实测数据，每代都真的比上一代好一点（1/3 → 9/15 → ...）。
但**整条路可能就不该走**——机器人有 0.3 mm 精度，不需要夹具自定心。

这个判断 AI 往往给不出来，因为它被"完成当前任务"的目标锁住了。**必须你来问："这个功能真的需要吗？"**

---

## 4. 文件位置总表

### 项目根

```
D:\codex\projects\robot-sim-first-step\
```

### 要动的核心文件

| 完整路径 | 干什么 | 什么时候动 |
|---|---|---|
| `D:\codex\projects\robot-sim-first-step\scene_full.py` | **场景总装**：碰撞组、夹具、台钳、工件、渲染 | 改夹具（路线 B 就改这里） |
| `D:\codex\projects\robot-sim-first-step\simready_loader.py` | SimReady GLB→OBJ→MuJoCo 加载器 | 加新资产时 |
| `D:\codex\projects\robot-sim-first-step\demo_08_pick_and_load.py` | `Arm` 类：IK、`move_to()`、抓取序列 | 改动作时 |
| `D:\codex\projects\robot-sim-first-step\demo_09_try_versions.py` | 严格三判据评分，出四视图 | 验收时跑 |
| `D:\codex\projects\robot-sim-first-step\cnfont.py` | matplotlib 中文字体 | 图里中文变方框时 |

### 诊断脚本（排查用，别删）

| 完整路径 | 查什么 |
|---|---|
| `D:\codex\projects\robot-sim-first-step\diag_slot_alone.py` | **纯夹具落料自定心率（当前主战场）** |
| `D:\codex\projects\robot-sim-first-step\diag_fixture_profile.py` | 夹具 y-z 剖面图，看斜面朝向对不对 |
| `D:\codex\projects\robot-sim-first-step\diag_grasp_geometry.py` | 夹爪 pad 位置、开度 vs ctrl 曲线 |
| `D:\codex\projects\robot-sim-first-step\diag_grasp_pose.py` | 扫描 `wrist_roll` 找正确抓取姿态 |
| `D:\codex\projects\robot-sim-first-step\diag_vise_structure.py` | 台钳各子部件包围盒 |
| `D:\codex\projects\robot-sim-first-step\diag_vise_orient.py` | 四种朝向渲染对比 |
| `D:\codex\projects\robot-sim-first-step\diag_shaft_slip.py` | 逐帧滑移监控 + 接触对名字 |
| `D:\codex\projects\robot-sim-first-step\diag_scene_full_reach.py` | 可达性检查 |

### 会生成的图（用眼睛验收这些）

| 完整路径 | 内容 |
|---|---|
| `D:\codex\projects\robot-sim-first-step\scene_full.png` | 场景三视图 |
| `D:\codex\projects\robot-sim-first-step\demo_08_sequence.png` | 抓取序列帧 |
| `D:\codex\projects\robot-sim-first-step\demo_09_result.png` | 三判据结果四视图 |
| `D:\codex\projects\robot-sim-first-step\fixture_profile.png` | 夹具剖面 |
| `D:\codex\projects\robot-sim-first-step\vise_orient.png` | 台钳四种朝向对比 |

### 3D 数据

| 完整路径 | 内容 |
|---|---|
| `D:\codex\projects\robot-sim-first-step\models\simready\manifest.jsonl` | 2,595 条 SimReady 资产清单 |
| `D:\codex\projects\robot-sim-first-step\models\simready\obj\` | GLB 转出的 OBJ |
| `D:\codex\projects\robot-sim-first-step\models\simready\mjcf_meshes\` | MuJoCo 用网格 |
| `D:\codex\projects\robot-sim-first-step\models\_tmp_so101\lerobot_env_so101-0.4.0\lerobot_env_so101\assets\so101_new_calib.xml` | **SO-101 机器人定义** |
| `D:\codex\projects\robot-sim-first-step\models\_tmp_so101\lerobot_env_so101-0.4.0\lerobot_env_so101\assets\assets\` | 13 个 STL（注意多一层 `assets/`） |

> ⚠️ MuJoCo 的 `meshdir` 是**全局唯一**的。所有生成的 OBJ 必须写进上面这个 `assets\assets\` 目录，否则加载报错。

### Python 环境

| 完整路径 | 内容 |
|---|---|
| `D:\robot-env\Scripts\python.exe` | **解释器（所有命令用这个）** |
| `D:\robot-env\Lib\site-packages\` | mujoco 3.17.0 / numpy / matplotlib / opencv / pybullet / mediapipe / imageio / trimesh 5.2.0 / pygltflib / scipy 1.18.1 |

### 文档

| 完整路径 | 内容 |
|---|---|
| `D:\codex\projects\robot-sim-first-step\HANDOFF.md` | **本文档** |
| `D:\codex\projects\robot-sim-first-step\docs\README.md` | 项目说明 |
| `D:\codex\projects\robot-sim-first-step\docs\01` ~ `07` | 分阶段笔记 |

---

## 5. 怎么跑

### 每次运行必须先设这三行

```powershell
Set-Location D:\codex\projects\robot-sim-first-step
$env:MUJOCO_GL="glfw"
$env:PYTHONUTF8="1"
$py = "D:\robot-env\Scripts\python.exe"
```

不设会怎样：
- `MUJOCO_GL` 不设 → 渲染崩
- `PYTHONUTF8` 不设 → 中文输出乱码

### 常用命令

```powershell
& $py scene_full.py              # 建场景 + 三视图 -> scene_full.png
& $py diag_slot_alone.py         # 纯夹具落料测试  <- 当前卡点
& $py diag_fixture_profile.py    # 夹具剖面图      -> fixture_profile.png
& $py demo_08_pick_and_load.py   # 抓取序列        -> demo_08_sequence.png
& $py demo_09_try_versions.py    # 三判据评分      -> demo_09_result.png
```

### 过滤噪音

MuJoCo / matplotlib 会刷一堆 warning，加后缀清爽：

```powershell
| Where-Object { $_ -notmatch 'Warning|Glyph|plt\.|CategoryInfo|FullyQualified|所在位置|^\+' }
```

---

## 6. 已验证的事实（冻结，别重新测）

### SO-101 机器人

| 项 | 值 |
|---|---|
| 关节 | 6 个：`shoulder_pan` `shoulder_lift` `elbow_flex` `wrist_flex` `wrist_roll` `gripper` |
| gripper ctrl 语义 | **增大 = 张开**（反直觉，别搞反） |
| ctrl 0 → pad 间隙 | 14.8 mm |
| ctrl 1.74 → pad 间隙 | 128 mm（超过物理极限被夹住） |
| 最大开口 | 94.4 mm |
| pad 相对 `gripperframe` 偏移 | `[-0.0004, 0.0274, 0.0029]` |
| **正确抓取姿态** | `wrist_roll = 1.55 rad`（pad 轴 `[-0.065, 0.996, 0.068]`，与 x 轴 86.3°，与水平 3.9°） |
| 关节力矩限 | `forcerange="-3.35 3.35"` |
| 安全负载 | 600 g |
| **pad 中心到腕部底面** | **仅 8.2 mm**（关键限制，见 §7） |
| 腕部宽度 | 65 mm |

### 工件

```
D 形轴：Φ20 × 80 mm，铣了一个 4 mm 平面防止滚动
质量 0.197 kg
初始位置 [0.26, -0.13, 0.010]
```

### 场景常量

```python
TABLE_TOP_Z = 0.0
TABLE_SIZE  = (0.60, 0.50, 0.024)

CT_ROBOT   = (1, 3)   # 机器人
CT_WORK    = (2, 3)   # 工件
CT_FIXTURE = (4, 2)   # 夹具：只和工件碰
CT_STATIC  = (4, 3)   # 台面/地面

VISE_POS   = [0.20, 0.17, 0.0]         # 台钳位置（不能放 0.30）
vise_quat  = [cos45°, 0, 0, sin45°]    # 绕 Z +90°
```

**台钳为什么在 x=0.20 而不是 0.30**：SO-101 臂展只有约 40 cm。放 0.30 时整段搬运手臂完全伸直，末端精度崩掉，实测残差 160 mm。

### 碰撞判定规则

```
(contype_A & conaffinity_B) || (contype_B & conaffinity_A) != 0
```

代入验证：

```
robot(1,3) vs fixture(4,2) = (1&2)|(4&3) = 0|0 = 0   不碰  ← 故意
robot(1,3) vs table(4,3)   = (1&3)|(4&3) = 1|0 = 1   碰
work(2,3)  vs fixture(4,2) = (2&2)|(4&3) = 2|0 = 2   碰
work(2,3)  vs table(4,3)   = (2&3)|(4&3) = 2|0 = 2   碰
```

---

## 7. 为什么要有碰撞组（踩过最贵的坑）

**实测**：pad 中心到腕部底面只有 **8.2 mm**，腕部宽 **65 mm**。pad 一旦进钳口，腕部也进去了；腕部比任何钳口都宽，直接顶在夹具上表面。

症状：下压到 z≈0.1377 就卡死，残差从 11 mm 暴涨到 127 mm。

**这是机械臂的结构限制，调参数救不回来。** 真实夹具会做成让开腕部的形状。仿真里用碰撞位掩码达到同样效果：**夹具永远不和机器人碰撞**。

副作用：落料诊断如果只把关节归零、不禁用机器人碰撞，夹爪会托住工件，测出来的全是假数据。`diag_slot_alone.py` 已修好（会打印 `disabled collision on 8 robot geoms`）。

---

## 8. 夹具演进史（8 次失败，全有实测数据）

当前实现：**双托辊 V 形块**

```python
ROLL_D     = 0.016       # 托辊直径 Φ16
ROLL_ANGLE = 38°         # 接触角（从竖直方向量）
ROLL_L     = 0.130       # 托辊长度（沿 x）
FIXTURE_X  = 0.16
FIXTURE_Y  = 0.075
```

几何解：

```
工件半径 Rw = 10 mm，托辊半径 rr = 8 mm
两轴距离 = 18 mm
托辊轴心 z = 8 mm（辊底刚好贴台面）
工件轴心 z = 8 + 18·cos38° = 22.2 mm
托辊轴心 y 偏移 = 18·sin38° = 11.1 mm
```

### 失败链

| # | 方案 | 结果 | 死因 |
|---|---|---|---|
| 1 | 钳口沿 x 分开 | 夹不住 | 沿 x 分开 = 和轴平行，等于从两头推轴 |
| 2 | 窄槽 25 mm + 直角台阶 | 1/3 | 偏移 5 mm 就弹开 |
| 3 | 窄槽 + 32° 斜面 | 0/15 | 偏移 4 mm 卡在钳口顶沿 |
| 4 | 纯 V 形，45° 尖底 | **9/15** | 偏移 10~25 mm 卡在斜面上。尖底点接触，法线朝侧上方，水平分力不够 |
| 5 | 梯形槽（V 底带平台） | 2/11 | 工件悬在 z=20 不落。底部支撑板先接住了它 |
| 6 | 删掉支撑板 | 2/11 | 同上，仍没落到斜面 |
| 7 | box 阶梯，10 级 | 4/11 | 骑在台阶棱上 |
| 8 | box 阶梯，24 级，35° | 6/11 | 仍卡在 z=11~20 |
| 9 | 双托辊 | **2/13** | 见 §9 |

### 两条通用规律

**规律一：摩擦角必须小于斜面角。**
`atan(μ) < 斜面角`，否则工件静止卡在斜面上不往下滑。
实例：斜面 45°，μ=1.2 → `atan(1.2)=50.2° > 45°` → 卡住。改 μ=0.15（8.5°）后明显改善。

**规律二：任何水平棱都是卡点。**
阶梯近似斜面永远不行，因为台阶棱是水平线，工件会停在棱上。加到 24 级只是把卡点变密。
**曲面（圆柱）的接触法线连续变化，没有卡点。** 这是第 9 代换托辊的原因，也是真实机床 V 形块的真实结构。

---

## 9. 当前卡点与诊断

### 实测数据

`diag_slot_alone.py`（机器人碰撞已禁用，工件从夹具中心上方 90 mm 自由落下）：

```
drop offset  +0 mm -> y   -0.0 mm  z  +13.1 mm  CENTERED
drop offset  +5 mm -> y  -12.1 mm  z  +16.0 mm  OFF
drop offset +10 mm -> y   -0.3 mm  z  +13.5 mm  CENTERED
drop offset +15 mm -> y  +24.4 mm  z  +19.9 mm  OFF
drop offset +20 mm -> y  +25.5 mm  z  +11.9 mm  OFF
drop offset +25 mm -> y  +25.5 mm  z  +11.9 mm  OFF
drop offset +30 mm -> y  +14.8 mm  z  +18.5 mm  OFF
drop offset +35 mm -> y   +9.7 mm  z  -12.5 mm  OFF
drop offset +40 mm -> y  +39.7 mm  z  +63.8 mm  OFF
drop offset +45 mm -> y  +45.8 mm  z  +63.8 mm  OFF
drop offset +50 mm -> y +131.2 mm  z  -12.5 mm  OFF
drop offset +55 mm -> y  +57.1 mm  z   +7.8 mm  OFF
drop offset +60 mm -> y  +60.0 mm  z   +7.8 mm  OFF

self-centering: 2 / 13
```

### 症状判读

- `z = +63.8 mm`、`y = +131.2 mm` —— **被弹飞了**，不是滑落。这是硬碰撞反弹的特征。
- 只有 |偏移| ≤ 10 mm 能进，**有效窗口约 ±10 mm**，比理论开口小得多。
- 90 mm 自由落体末速 1.33 m/s，这个速度远高于真实"机器人缓慢放入"的工况。**这一批数据本身就偏悲观。**

### 我的判断

1. **先别继续调夹具。** 已烧 8 轮，收益递减。
2. 关键战略问题：**机器人抓放任务真的需要自定心吗？**
   自定心是给手动上料用的。机器人有 < 0.3 mm 的 IK 精度，只要放得准，**一块简单平台就够**。真实产线靠机器人精度 + 视觉，不靠夹具自纠。
   我在这条路上耗太久，方向可能一开始就选重了。

---

## 10. 三条路线，建议 B

### 路线 A：继续调双托辊（不推荐）

- 落料高度从 90 mm 降到 3~5 mm，看是否还弹（当前数据全是大高度自由落体，不真实）
- 给托辊设 `solref` / `solimp` 软化接触（MuJoCo 默认接触很硬，圆柱-圆柱容易弹）
- 增大托辊直径、减小接触角，撑大有效窗口

风险：即使调到 12/13，也只解决"手动上料"场景，对机器人任务无实际价值。

### 路线 B：放弃自定心，换简单平台（推荐）

理由：机器人 IK 精度 0.3 mm，工件 Φ20，夹爪开度 94 mm，**放准完全做得到**。平台只需"支撑住、别掉下去"。

做法：
1. 夹具换成带浅槽的方台：槽宽 24 mm（比 Φ20 大 4 mm 容错），槽深 3 mm
2. 判据从 ±6 mm 放宽到 ±8 mm
3. 跑 `demo_09_try_versions.py` 拿 3/3

工作量：改 `scene_full.py` 里夹具那约 30 行 + 重跑两个脚本。

### 路线 C：加视觉闭环（长期正解，成本高）

腕部装相机，实测工件位置后修正。真实产线做法，但要接渲染→图像→位姿估计整条链，工作量是 B 的 10 倍。

---

## 11. MuJoCo 踩坑清单

| 坑 | 症状 | 解法 |
|---|---|---|
| **`meshdir` 全局唯一** | `Error opening file 'vise_sr_col.obj'` | 所有生成的 OBJ 必须写进 SO-101 的 mesh 目录 |
| **`mj_kinematics` 不算雅可比** | `mj_jacSite` 全返回 0，IK 残差 284 mm | 必须先调 `mj_forward` |
| **mesh 编译时自动居中** | 按原始坐标算的 `pos` 补偿全错位 | 读回 `model.mesh_vert` 实测范围反推，或干脆不用 mesh |
| **自定义小 mesh 当碰撞体会失效** | 工件**直接穿过**楔块掉到台面，接触列表里完全没有 `vjaw` | 用 `mjGEOM_BOX` / `CYLINDER` 拼 |
| **凸包会填平凹腔** | 台钳钳口槽被填成实心块，工件被弹飞 | 凸包必然填凹腔，不可调；凹形夹具必须自建碰撞体 |
| **摩擦角 > 斜面角就卡住** | 工件静止停在斜面上 | `μ < tan(斜面角)` |
| **水平棱都是卡点** | 阶梯近似斜面永远卡 | 改用曲面 |
| **位置伺服要额外收敛步** | 负载下残差 100+ mm | 插值结束后补一段 settling steps |
| **IK 里锁 `wrist_roll` 必须在求解循环内** | 姿态被覆盖 | 别在循环外强制赋值 |

### PowerShell 专属坑（血的教训）

> **绝对不要用 PowerShell 的 `-replace` 改含中文的 .py 文件。**
> 会把多字节 CJK 打乱，导致 `def _mask(g, *CT_ROBOT)` 这类签名被破坏、文件语法错误。
> 因此重写了三遍 `scene_full.py`。

正确写法：

```powershell
# 用 Python 打补丁
& $py _patch.py

# 或写文件时显式 UTF-8 无 BOM
[System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding $false))
```

`Set-Content -Encoding utf8NoBOM` 在这个 PowerShell 版本上不支持。

---

## 12. 资产与网络

### SimReady 资产库

```
D:\codex\projects\robot-sim-first-step\models\simready\manifest.jsonl   2,595 条
```

分类：food 531、kitchenware 179、box 154、handTool 97、equipment 59、factory 15（全是螺丝螺母）

**关键结论：库里没有车床、没有 CNC、没有卡盘、没有主轴**（已验证 0 命中）。

所以"二轴数控车 + 三爪卡盘"必须自己建。**这是这个项目真正稀缺的原创部分，不是资产。** 也是你开源发布时唯一有壁垒的东西。

已于 2026-11-05 14:40 在 `scene_machining.py` 自建完成（床身 + 主轴 + 三爪卡盘，`diag_chuck_grip.py` 已验证夹紧）。与阶段一 `scene_full.py` 的合并也已完成（`add_lathe(spec)`），机器人送料装夹已跑通，见 §13。

### 网络

可用：

```
huggingface.co                    ✅
api.github.com                    ✅
mirrors.aliyun.com/pypi/simple    ✅
gcore.jsdelivr.net                ✅
registry.npmmirror.com            ✅
```

被墙：

```
simready.com            ❌
developer.nvidia.com    ❌
pypi.org                ❌（SSL）
github.com 裸域         ❌（用 api.github.com 代替）
```

### 已放弃的方案

**Isaac Sim / Isaac Lab**：需 torch 2.11，微调要 40 GB 显存，本机 RTX 5070 Ti Laptop 只有 12 GB，驱动 591.86。不值得。
**替代**：`mujoco-mjx` 是更便宜的 GPU 并行路线。

---

## 13. 待办

### 立刻（路线 B）— ✅ 2026-11-05 全部完成

- [x] 夹具换简单平台 + 浅槽 → `diag_slot_alone.py` **13/13**
- [x] `demo_09_try_versions.py` **3/3**
- [x] 整段序列 GIF → `demo_09_sequence.gif`

### 之后（按优先级）

- [x] `git init` + 首次提交（**仅本地**，未连 GitHub）
- [x] `verify_all.py` 一键回归（7/7 PASS）
- [x] **卡盘工位体检**：新增 `diag_chuck_health.py`，物理夹紧 + 机器人可达性一次量完
- [x] **卡盘位置修正**：`LATHE_Y` 0.24 → 0.12（可达性实测 13.40 → 9.08 mm）
- [x] **合并两条场景线（2026-11-05）**：`scene_machining.add_lathe(spec)` 已抽出，`scene_full.build_scene(with_lathe=True, lathe_origin=...)` 调用；默认关闭，当时 `verify_all` 6/6（现 7/7）与 `demo_09` 3/3 均不退化。合并场景 `nu=7, neq=2, jaw_act=6`；卡爪用 `CT_FIXTURE=(4,2)` 只跟工件碰、不跟夹爪碰，主轴箱/盘体纯视觉
- [x] **机器人送料装夹（2026-11-05）**：`diag_feed_chuck.py` 从 KNOWN FAIL 转 **PASS** 并挂进 `verify_all.CHECKS`（7/7）。实测同轴 0.89 mm / 倾斜 5.9° / 11 接触 / 三爪法向力 18.7 N / 重力下沉 0.01 mm / 20 N 拉 0.62 mm。三个决定性修复：① **把 D 铣扁做成真碰撞体**（用户选的 a，box 先用着，形状待精修）——对照组保持纯视觉时同轴 1134 mm、轴被弹飞，证明它是必要条件；下降阶段倾斜 23.6° → 4.1°；② **夹紧阶段补摩擦求解器参数**：`scene_full.py` 从未设 `spec.option`，而 `scene_machining.py` 有 `noslip_iterations=10 / impratio=5 / ELLIPTIC`；从一开始就套会毁掉按默认参数调好的抓取（轴飞到 x=-34 m），只在夹紧阶段写 `model.opt` 后法向力 8.4 → 18.7 N（独立卡盘场景实测 18.2 N）；③ **退出时保持卡爪夹紧**：原 move helper 每步写 `CHUCK_OPEN_CTRL`，边退出边松爪，轴直接掉台面（同轴 50.85 mm）。**已排除**：单纯加大夹紧行程（-0.011/-0.013/-0.015，力到 21.4 N 但 20 N 拉仍 4713~35460 mm）、全局对齐求解器、±拉拔方向、flat 只碰 pad 的掩码。**更正上一轮记录**：本轮实测两 pad 连线为 `[0.64, 0.768, 0.006]`，z 分量只有 0.006（几乎水平），旧记录 0.068 是 pad 轴不是连线，"两片 pad 不等高"并非根因；真正问题是连线与轴成 ~50° 斜夹、轴吊在 pad 下方靠摩擦悬着
- [x] **三爪真同步（2026-11-05 补完）**：`scene_machining.py` 现在是 1 个丝杠执行器 `jaw_act` + 2 条 `mjEQ_JOINT` 联动约束（`nu=7, neq=2`）。**真正的根因不是驱动架构，是几何**：三个爪的 box 从未按 120° 旋转，沿各自径向的有效半宽是 9.0 / 12.3 / 12.3 mm，三个相同电机当然停在不同位置。证据链：① 只改几何、驱动一行不动 → 位移差 4.24 → 0.23 mm，力 `[7.1,23.8,25.8]` → `[19.1,18.4,19.4]` N；② 再换单丝杠 + 联动 → **0.118 mm**、力比 1.57、偏心 0.00 mm、20 N 拉 0.10 mm；③ 早先"3 伺服 + 约束"12 组全灭（q 冲到 202 mm）是因为 3 个伺服和约束互相打架，不是真同步不成立。残余 0.1 mm 是软约束在接触载荷下的弹性柔度，真实卡盘丝杠也不是绝对刚体。判据已纳入体检与 `verify_all.py`: `SPREAD_LIMIT=0.2 mm`、`FORCE_RATIO_LIMIT=2.0`
- [ ] **删运动学锁 + 精修形状（当前第一优先）**：搬运段目前每步把轴 qpos/quat 写回 `pad_center - offset / Q0`，等于假设夹爪完美形封闭；拉拔测试前已解除，`FEED_METRIC` 输出 `physics_after_clamp=1`，`verify_all.check_feed` 会验它，防止有人把锁延长到拉拔阶段造假。真实抓取仍倾斜 41.5°（无锁 98.2°），`demo_09` 的 3/3 没查姿态。修法：不凸出包络的真实 D 形铣扁 / 调平 pad / 从下方托住；同时给 `demo_09` 补姿态判据。修好后删锁，把送料升为全物理，再出 GIF
- [ ] 接 sw-mc 参数 JSON → 生成台阶轴 + 用 MasterCAM 粗/精车刀路驱动仿真刀具（不做真材料去除）
- [ ] reset 循环 + 域随机化 + 成功率统计

### 更远（职业目标：机器人仿真工程师 / AI 应用）

- [ ] 抓取策略从硬编码 IK 换学习方法
- [ ] `mujoco-mjx` GPU 并行加速训练
- [ ] 开源发布：稀缺原创件是**数控车床 + 卡盘模型**。
  公开前先做三件事：查清 `models/simready/`（119 MB 第三方 OBJ）和 SO-101 资产的再分发条款并补进 `LICENSE`；清理一次性 `demo_*` / `diag_*` 实验脚本；确认 `github.com` 裸域不通、需走代理或 API（§12）。

---

## 14. 给接手人的一句话

阶段一（抓 → 抬 → 搬 → 放进浅槽）、两条场景线合并、机器人送料 → 卡盘夹紧 → 退出，都已跑通，`verify_all.py` 7/7 PASS。

别再调夹具，也别再从零建车床。下一件真正值钱的事是**修夹爪抓持姿态、删掉搬运段的运动学锁**——它是目前唯一被假设掉的物理层。然后才轮到同一份 sw-mc 工艺数据驱动 CAD/CAM/仿真。

这个项目的教训不是"某个参数调错了"，而是**代码和文档会悄悄分叉**：跑起来之前别信文档里的数字。
