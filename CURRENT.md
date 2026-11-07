# 现在在做什么

给下一个接手的人或 agent 看的现场记录。只写"此刻"的状态，不写历史；换手前改这里，不要另开日志。
最后更新：2026-11-07

## 当前目标：交接瞬间的姿态 —— 已达成

用户看到的五个现象，都发生在**机械臂松爪 ↔ 卡盘卡爪接手**这一瞬间：
往左上怼一下 / 放掉料自己往下偏 / 再夹紧又动一下 / 找轴心吨吨吨几下 / 最终偏心太大。
上一轮曾把验收收窄成"余量偏心 ≤ 2 mm、偏移第一刀能切掉"，用户明确推翻：
**要的是过程本身不出这几下，不是成品精度。别再只看最终值判 PASS。**

本轮（第三轮）按过程姿态解决，`--v-jaw --lift --no-lock` 主路径已达标。

## 本轮定案：实体导套 + 先松爪交接

默认路径实测（不带任何环境变量），对照上一轮默认：

| 指标 | 旧默认 | 新默认 |
|---|---|---|
| 三爪首次接触推挤（咬合同轴） | 4.20 mm | **0.53 mm** |
| 夹紧瞬态 z 变化 | +2.85 mm | **−0.05 mm**（峰谷 −0.14~+0.91） |
| 原地松爪 同轴 / 轴向偏 | 2.69 mm / 1.8° | **0.53 mm / 1.8°** |
| 交接位移（z 峰谷） | 4.77 mm | **1.05 mm** |
| 最终同轴 / 倾斜 | 0.87 mm / 0.3° | **0.53 mm / 1.8°** |
| 余量偏心（限 2.0） | 1.05 mm | 1.79 mm |
| 20 N 拉拔 / 重力下沉 | 0.00 / 0.00 mm | 0.00 / 0.01 mm |
| 三爪接触 / 力 | 30 / 166.0 N | 30 / 165.3 N |

倾斜 0.3°→1.8°、余量 1.05→1.79 是代价，都在判据内（10° / 2.0 mm）。
**夹爪先松时同轴 0.00 mm、不下沉；三爪闭合时料已在轴线上，所以首次接触不再推挤。**
咬合后导套自己卸载：`解锁静置 导套=0/0.0N`，拉拔/下沉是卡爪自己的力，不是假通过。
无数值警告。

### 默认行为（不带任何环境变量）

`FEED_BUSHING=1`、`BUSHING_GAP=0.2 mm`、`BUSHING_STIFF=1`、
`HANDOVER_FIRST=1`、`SEAT_RADIAL_K=5000`、`SEAT_FADE=400`。
**只对「真实交接的全物理 V 槽路径」生效**（`tuned = 有导套几何 and v_jaw and lift and not carry_lock`）：

- `送料·全物理(V槽)` → 新定案，交接位移 1.05 mm。
- `机械臂送料装夹(历史基线·穿模)` `--passthrough` → carry_lock=True，走旧路径。
- `送料·官方场景(SO-101 env)` → `build_model(official_world=True)` 不装导套，旧交接（同轴 0.87 / 0.3°）。
  官方 env 是另一套世界（自有 timestep/相机/几何），导套按主场景 `LATHE_ORIGIN` 标定，放进去会把料别歪（实测 3.20 mm / 7.2°、余量 8.17 FAIL）。

`FEED_ARM_KV=5.0` 仍只在送料场景注入，**全局 `ARM_SERVO_KV` 必须保持 2.6**（改了 demo_09 掉到 8/9）。

### 为什么不是"单独加一个导套"

CURRENT.md 上一轮推断"缺的是实体导套几何"，**实测只答对一半**：

- 单独加实体导套是**阴性**：软导套被三爪挤穿（只顶回 9~17 N，咬合仍 4.3 mm）；
  硬导套与三爪抢料（峰值 250 N，松爪 7.24 mm / 8.6°、余量 7.28 FAIL）。
- 扫过 gap 0.05~0.8、len 14~40、front −20~25、lower 半环、跨卡爪口/入爪：
  单独用最好也只是勉强 PASS，过程姿态不如基线；加长、后移、放进爪内一律更差。
- **只有配合「先松夹爪 → 端面顶正 → 三爪闭合 → 径向托力 quintic 渐变卸载」才成立。**
  单独 `HANDOVER_FIRST` 无导套也 FAIL（余量 3.62）；两者是组合效应。

### 实现要点（改代码前必读）

- 导套 = 12 块切向 box 拼的多边形环，内孔 = 棒料 R10 + gap，
  卡爪口外 `x=[jaw_front−20, −6] mm`，常驻几何**从不撤销**。
- 掩码 `(contype=0, conaffinity=2)`：只碰工件 (2,3)，不碰机器人/机床。
  用卡爪的 `CT_FIXTURE(4,2)` 会撞机器人本体（4 & 12 = 4），第一次试就崩。
- **坑 1：掩码在编译期烘焙。** 构建期 (0,0) 再运行时改 `model.geom_conaffinity`
  完全不产生接触（探针 ncon=0）；构建期就是 (0,2) 则同样摆位立刻 ncon=12。
- **坑 2：常驻闭合环横在搬运路径上**，棒料撞外壁掉桌面（夹紧前 51 mm / 74.7°）。
  按真机中心架做法：构建期就带 (0,2)，初始沿径向**张开 40 mm**，进给到位由
  `close_bushing()` 把 `model.geom_pos` 移回闭合位（位置每步从 model 重算，可以改）。
- **坑 3：必须开硬接触**（`solref 0.001/1` + 陡 `solimp`），软接触顶不住；
  但硬接触单独用会抢料，必须与 `HANDOVER_FIRST` 同开。
- 日志里的「卸载托力」指撤销 `SEAT_RADIAL_K` 那层**力等效托力**，不是撤实体导套。

## 还没解决 / 别自己扩

- 倾斜与余量略有回退，要压回旧水平得动卡爪几何或加大插入深度（35/45/55 mm
  之前因机械臂够不到 FAIL，需先改工位/夹爪站位）——用户没要求，不要自己开。
- 定案组合**没有**移植到官方 env 场景；那两套仍是独立对照。
- 随机来料、流水线、训练：明确暂不碰。

## 已否掉（别再重试）

夹紧力 150→300 N、卡爪相位（固定 90/210/330）、丝杆联动、软爪几何/包角、
预收托圈、两段位移闭合、夹爪半开、受载自镗 6 档、低压预夹 5 档、
关重力、速度前馈/速度误差阻尼/柔顺期阻尼/夹紧前收敛等待、
单独实体导套（含软/硬/半环/加长/后移/入爪）、单独先松爪交接。

## 怎么复现

```bash
python verify_all.py                                   # 9/9
python src/diag_feed_chuck.py --v-jaw --lift --no-lock # 默认即定案
# 旧默认对照（四个开关要一起关）
FEED_BUSHING=0 FEED_HANDOVER_FIRST=0 FEED_SEAT_RADIAL_K=0 FEED_SEAT_FADE=0 \
  python src/diag_feed_chuck.py --v-jaw --lift --no-lock
FEED_BORE_PROBE=1 python src/diag_feed_chuck.py --v-jaw --lift --no-lock
```

看 `导套闭合`、`夹爪先松`、`三爪咬合`、`夹紧瞬态`、`卸载托力`、`原地松爪`、
`交接位移`、`解锁静置`、`FEED_METRIC*`。关闭定案也可以用 `FEED_BUSHING=0` 单开关
（主路径会退回旧行为，但 `HANDOVER_FIRST` 仍开着，要完整对照按上面四个一起关）。

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

`verify_all.py` **9/9 通过**。推送需用会话代理端口，全局配置里的 7897 是坏的：
`git -c http.proxy=$HTTP_PROXY -c https.proxy=$HTTPS_PROXY push origin main`
