# 现在在做什么

给下一个接手的人或 agent 看的现场记录。只写"此刻"的状态，不写历史；换手前改这里，不要另开日志。
最后更新：2026-11-07

## 当前目标（用户 2026-11-07 重新确认）

固定一根棒料，**机械臂自己**完成一次可靠送料：抓取 → 提起 → 对轴 → 插入卡盘 → 夹紧 → 松爪退出。
随机来料、流水线、训练明确暂不碰。

用户要的是**交接过程的姿态**：松爪 ↔ 卡爪接手那一瞬不能"往左上怼一下 / 放掉料自己偏 /
再夹紧又动一下 / 找轴心吨吨吨几下"。不能只看最终余量 PASS 就收尾。

**验收必须分层，不能混成一个**（用户 2026-11-07 定的规矩）：
- **卡盘基准测试** = 只摆料、不用机械臂、不用对中力场。只证明卡盘本身，
  **不能记作机械臂送料通过**。
- **机械臂送料** = 必须跑 `src/diag_feed_chuck.py --v-jaw --lift --no-lock` 全物理路径，
  由机械臂实际抓料、送入、交接才算数。
- 直接放料不算作弊，前提是不冒充送料成功；**用隐藏力场替机械臂对中才算不严谨**。

## 状态：上一版定案作废，默认已回退

上一轮把"交接位移 4.77 → 1.05 mm"当作成果，用户否掉了。拆开对照证实：
把棒料摆回主轴轴线的主要是 `seat_rod()` 里那根 `qfrc_applied` 数学弹簧
（`SEAT_RADIAL_K=5000`，目标直接取 `axis[1]/axis[2]` **理论轴线坐标**），
等于替机械臂完成了最后对中。关掉它、只留实体导套，导套反以 483 N 把料别到
14.2° FAIL。所以"加实体导套"这话夸大了，功劳归错。

**代码默认已全部回退为关**：`FEED_BUSHING=0`、`BUSHING_STIFF=0`、
`HANDOVER_FIRST=0`、`SEAT_RADIAL_K=0`、`SEAT_FADE=0`。开关全保留，仅供对照，
**不要再把 `SEAT_RADIAL_K` 设回默认**。主路径现在输出的是诚实的无作弊基线：

`夹紧前同轴 0.06 → 三爪咬合 4.20 → 夹紧瞬态 z +2.85 → 交接位移 4.77 mm →
最终同轴 0.87 / 倾斜 0.3° / 余量 1.05 mm / 拉拔 0.00 / 力 166.0 N`。
`FEED_METRIC PASS` 只说明最终值在判据内，**过程姿态仍未达标**。

## 阶段 A：卡盘基准测试 —— 完成（本轮）

命令：`python src/diag_chuck_seat.py`（日志 `_build/chuck_baseline.log`）

脚本 `src/diag_chuck_seat.py` 已重写：只摆料、不驱动机械臂、不建任何指向轴线的
弹簧/挡块/导套；给定插入深度 + 初始偏心 + 初始倾角，闭合三爪后测同轴、倾斜、
接触力、20 N 拉拔。三爪咬合后恢复重力并**显式清零** `qfrc_applied` 与
`xfrc_applied`，再静置和拉拔；全程统计与棒料接触的几何名。

**24 组全部 OK**（插入 25 / 47 mm × 初倾 0/1/2° × 初偏 0/1/2/4 mm，沿 y）：

- 25 mm 插入：头端偏 0.26~1.63 mm、倾斜 ≤3.39°、中心偏 ≤1.17 mm。
- 47 mm 插入：头端偏 0.17~0.84 mm、倾斜 ≤1.39°、中心偏 ≤0.35 mm，明显更稳。
- 初偏 4 mm 也能被拉回轴线；每组 30 个接触、三爪力 19~85 N、20 N 拉拔 ≤0.021 mm。
- **`外力max=0.0`、`臂碰=0`**（与棒料接触的只有 15 个 `jaw_geom_*`），不是假通过。
- 对照「开重力 + 纯竖直 mg 托力」（无径向项）：偏心沿 z 或 yz 时 4 mm → 倾斜
  3.58~4.59°、头端偏 1.52~2.33 mm，比水平偏心差。竖直偏心与重力耦合会恶化姿态。

诚实边界：夹紧期默认**关重力**（卡爪张开内径 44 mm、棒料 Ø20，不托住会在丝杠
走完前掉台面）。关重力是全局物理设置、不参与对中；另跑了 mg 托力对照互相印证。

### 阶段 A 给出的决定

**卡盘自己能把偏料夹正 → 卡盘侧不用改，不需要导套/挡块/软爪改造，也不必先做路径优化。**
问题不在"最终夹不夹得正"，而在夹紧过程中机械臂夹爪还夹着尾段，三爪第一次碰料
把料从 0.06 推到 4.20 mm。

**插入深度加深这条路当前走不通**（无作弊配置下扫 `FEED_INSERT_MM`）：

| 插入 mm | 25 | 30 | 35 | 40 | 47 | 55 |
|---|---|---|---|---|---|---|
| 夹紧前同轴 mm | 0.06 | 0.08 | 3.60 | 9.20 | 20.12 | 29.04 |
| IK 残差 mm | 2.5 | 2.5 | 6.4 | 13.4 | 26.8 | 41.8 |
| 交接位移 mm | 4.77 | 7.87 | 6.62 | 18.50 | 21.98 | 26.47 |

卡盘基准里 47 mm 最好，但机械臂够不到。要用它必须**先改工位/夹爪站位**，
不是调路径算法能解决的。

## 阶段 B：机械臂无作弊基线 —— 未达标

就是上面回退后的默认输出：交接位移 4.77 mm、首次接触推挤 4.20 mm、夹紧瞬态
+2.85 mm。这才是下一步要解决的真问题，且必须由机械臂侧的动作解决，
不许再引入"知道理论轴线在哪"的力场。

## 下一步（按用户认可的顺序）

1. 先让机械臂用**简单路径**把料送到位，不先做路径优化（阶段 A 已证明卡盘够用）。
2. 交接期只允许机械臂自己的动作（松爪时机、柔顺让位、夹爪站位）与真实工装几何，
   **禁止**以理论轴线为目标的 `qfrc_applied` 径向弹簧。
3. 若需要 47 mm 插入深度，先改工位/夹爪站位解决可达性，再谈交接。
4. 每一步都分开验收：卡盘基准归卡盘，机械臂送料归机械臂。

## 已否掉（别再重试）

夹紧力 150→300 N、卡爪相位（固定 90/210/330）、丝杆联动、软爪几何/包角、
预收托圈、两段位移闭合、夹爪半开、受载自镗 6 档、低压预夹 5 档、关重力、
速度前馈/速度误差阻尼/柔顺期阻尼/夹紧前收敛等待、
单独实体导套（软/硬/半环/加长/后移/入爪）、单独先松爪交接、
**以理论轴线为目标的径向对中弹簧（作弊，已被用户否掉）**。

卡盘基准里"关重力"是允许的理想化，和这里否掉的"关重力"不是同一件事：
后者曾作为送料交接的修复手段试过，阴性。

## 实现要点（改代码前必读）

- 导套几何仍在代码里但默认不装：12 块切向 box 环，内孔 R10+gap，
  卡爪口外 `x=[jaw_front−20, −6] mm`。掩码 `(0,2)` 只碰工件不碰机器人
  （用卡爪的 `CT_FIXTURE(4,2)` 会撞机器人本体）。
- **坑 1：碰撞掩码在编译期烘焙。** 构建期 (0,0) 再运行时改 `geom_conaffinity`
  完全不产生接触；构建期就是 (0,2) 则立刻 ncon=12。
- **坑 2：常驻闭合环横在搬运路径上**，棒料撞外壁掉桌面。真机中心架送料时
  本就张开、到位才合上；实现靠改 `model.geom_pos` 瞬移，没有驱动/时序/质量。
- **坑 3：`tuned = bool(bushing_ids) and v_jaw and lift and not carry_lock`。**
  `FEED_BUSHING=0` 会把整组交接开关一起短路，所以"有弹簧无导套"不是有效对照。
  想让"先松爪"独立于导套生效，得先解耦这个门控。
- `FEED_ARM_KV=5.0` 仍只在送料场景注入，**全局 `ARM_SERVO_KV` 必须保持 2.6**
  （改成 5 会让 demo_09 掉到 8/9）。

## 怎么复现

```bash
python verify_all.py                                    # 9/9
python src/diag_chuck_seat.py                           # 阶段 A 卡盘基准
python src/diag_feed_chuck.py --v-jaw --lift --no-lock  # 阶段 B 无作弊基线（默认）
# 复现已作废的旧定案（仅作对照，六个开关要一起开）
FEED_BUSHING=1 FEED_BUSHING_GAP=0.2 FEED_BUSHING_STIFF=1 \
FEED_HANDOVER_FIRST=1 FEED_SEAT_RADIAL_K=5000 FEED_SEAT_FADE=400 \
  python src/diag_feed_chuck.py --v-jaw --lift --no-lock
FEED_INSERT_MM=47 python src/diag_feed_chuck.py --v-jaw --lift --no-lock # 够不到
```

看 `夹紧前`、`三爪咬合`、`夹紧瞬态`、`交接位移`、`解锁静置`、`FEED_METRIC*`。
卡盘基准看 `外力max`（必须 0）、`臂碰`（必须 0）、`20N拉mm`。

## 文献 / 论坛出处

- A 交接期较劲：机械社区《关于机床上下料的问题》
  http://mp.weixin.qq.com/s?__biz=MzA5ODYyNjQxMQ==&mid=2668557070&idx=1&sn=f3038a239ac8e6dec9a446925814f347
  FANUC 软浮动专利 DE69837741T2 https://patents.google.com/patent/DE69837741T2/en
  EVST《Robot Lathe Tending: Chuck Handoff and Interlocks》
  https://www.evsint.com/robot-lathe-tending-chuck-handoff-interlocks/
- B hunting / 欠阻尼：Loughborough 综述
  https://repository.lboro.ac.uk/articles/journal_contribution/Motion-control_techniques_of_today_and_tomorrow_A_review_and_discussion_of_the_challenges_of_controlled_motion/14047004/1/files/26506868.pdf
  Industrial Monitor Direct《Robot Force Mode Seating in CNC Chuck: Fixing Wobble》
  https://industrialmonitordirect.com/blogs/knowledgebase/why-does-the-robot-wobble-when-force-seating-a-part-in-a-chuck
- C 卡盘侧：Practical Machinist《3 jaw question for lathe》
  https://www.practicalmachinist.com/forum/threads/3-jaw-question-for-lathe.265817/
  MentorCNC 软爪受载自镗 https://www.mentorcnc.com/en/cnc-lathe-training-lesson-5/
  浮动动力卡盘专利 CN121535233A https://eureka.patsnap.com/patent/CN121535233A

## 同步状态

`verify_all.py` **9/9 通过**（回退默认后重跑）。推送需用会话代理端口，
全局配置里的 7897 是坏的：
`git -c http.proxy=$HTTP_PROXY -c https.proxy=$HTTPS_PROXY push origin main`
