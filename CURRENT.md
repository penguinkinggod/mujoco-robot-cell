# 现在在做什么

给下一个接手的人或 agent 看的现场记录。只写"此刻"的状态，不写历史；换手前改这里，不要另开日志。
最后更新：2026-11-07（阶段 B 第一轮）

## 当前目标

固定一根棒料，**机械臂自己**完成一次可靠送料：抓取 → 提起 → 对轴 → 插入卡盘 → 夹紧 → 松爪退出。
随机来料、流水线、训练明确暂不碰。

用户要的是**交接过程的姿态**：松爪 ↔ 卡爪接手那一瞬不能"往左上怼一下 / 放掉料自己偏 /
再夹紧又动一下 / 找轴心吨吨吨几下"。不能只看最终余量 PASS 就收尾。

**验收必须分层**：卡盘基准（`diag_chuck_seat.py`，只证明卡盘）≠ 机械臂送料
（`diag_feed_chuck.py --v-jaw --lift --no-lock`，必须机械臂实际抓料送入）。
用隐藏力场替机械臂对中不算数。

## 状态：阶段 B 第一轮已落地为新默认

主路径现在输出（不带任何环境变量）：

`夹紧前 0.06 → 三爪咬合 3.06 → 夹紧瞬态 z +2.20 → 交接位移 3.75 mm →
原地松爪 2.33 mm/1.4° → 最终 0.74 mm/0.7° → 余量 0.92 mm → 拉拔 0.02 mm → 力 166.2 N`
`FEED_METRIC PASS`，**交接期棒料 qfrc 峰值 = 0.000 N**。

| 指标 | 回退后的诚实基线 | 现默认 |
|---|---|---|
| 三爪首次接触推挤 | 4.20 mm | **3.06** |
| 夹紧瞬态 z | +2.85 mm | **+2.20** |
| 交接全程位移 | 4.77 mm | **3.75** |
| 原地松爪 同轴/倾斜 | 2.69 / 1.8° | **2.33 / 1.4°** |
| 最终 同轴/倾斜 | 0.87 / 0.3° | 0.74 / 0.7° |
| 余量偏心 | 1.05 mm | **0.92** |
| 棒料外力峰值 | 1.933 N（mg 托力） | **0.000** |

## 本轮最重要的发现：4.20 mm 不是三爪推的

加了逐帧跟踪（`FEED_CLAMP_TRACE=1`），证据很直接：
**卡爪还没碰到料（接触=0）的前 300 步，料已经从 −0.04 漂到 +3.29 mm，
而且和夹爪同步移动（爪 −2.43 → +3.05）。**

所以「三爪第一次碰料把料怼出去 4.2 mm」这个归因是错的。真正的链条是：

1. 进给到位时料在轴线上 0.06 mm；
2. 夹紧斜坡前 65% 行程卡爪还在空走（张开内径 44 mm vs 料 Ø20）；
3. 这段时间手臂在 0.2 kg 悬臂负载下持续变形 + 追进给终点 2.5 mm 的 IK 残差，
   夹爪带着料漂了 3.3 mm；
4. 三爪真正接触时料已经偏了，于是"咬合同轴"直接继承了这个偏移。

顺带查到 mg 托力（`CLAMP_LIFT`，旧默认 1.933 N）是漂移源之一：它纯 +z，
但料躺在**斜 V 槽**里，往上顶就沿斜面滑，实测把料顶出 3.3 mm 横向位移。

## 新默认做了什么（全部是机械臂自身动作 + 真实工装几何）

1. **终点锚定** `FEED_CLAMP_ANCHOR=1`：进给到位后把位置伺服目标一次性同步到
   实际关节角，不再追一个它到不了的旧目标。纯伺服动作，无外力。
2. **去掉棒料托力** `FEED_CLAMP_LIFT=0`：不再给棒料加 1.933 N 的 mg 托力。
   交接期棒料 qfrc 峰值因此为 0，这是"没有隐藏力场"的直接证据。
3. **下半环托料架** `FEED_BUSHING=1`：卡爪口外 `x=[−20,−6] mm` 六块 box 拼成
   下半圆，内孔 R11.6（间隙 1.6 mm），掩码 `(0,2)` 只碰工件。送料时张开 40 mm
   避开搬运路径，**进给到位才合上**，之后一路不撤销。
4. 场景分流：以上只给真实交接路径（`tuned = 有导套 ∧ v_jaw ∧ lift ∧ 非 carry_lock`）。
   历史基线·穿模、官方 env 场景保持旧行为，作为独立对照，回归仍 9/9。

为什么间隙取 1.6 mm：1.46~2.0 mm 连续 PASS，<1.44 开始出现 0.02 mm 级的
接触离散化跳变（1.38/1.40/1.44 会 FAIL）。托架只在下垂超过间隙时才接触，
不给料任何"轴线目标"；咬合与解锁静置时实测 `导套=0/0.0N`，不是被托出来的假通过。

## 诚实边界（不要当成已解决）

- **改善幅度有限**：交接位移 4.77 → 3.75 mm（−21%），肉眼仍能看出一下。
- **物理下限约 3.4~3.8 mm**：这一轮把机械臂侧能做的都试了（锚定 / 柔顺 gain
  0.001~1 / 负载重力补偿 / 交接期 kp ×2~×8 / 夹持力 −0.10~−0.18 / 常驻托架），
  没有一组突破 3.4 mm。根因是 SO-101 舵机 kp=17.8 太软，挂 80 mm 料、
  55 mm 悬空，手臂本身就会变形；提高 kp 反而引入 hunting（kp2/kp3 掉到 5.9/6.1）。
- **托架间隙 <1.44 mm 不可用**：与三爪形成过约束（间隙 1.2 mm 时最终 3.9°、
  余量 4.6 FAIL），且结果对间隙极敏感。
- **最终倾斜 0.3° → 0.7°**：略退，仍在判据内（10°）。

## 已否掉（别再重试）

- **常驻托架**（`FEED_BUSHING_ALWAYS=1`）：前段极好（咬合 0.87、交接 2.67），
  但三爪**夹空**——实测 `三爪咬合 接触=0 力=0.0N、导套=2/107.7N`，料被托架
  卡在卡爪口外，后段全崩（余量 3.3~4.9 FAIL）。托架退回又会炸松爪段（10 mm）。
- **零力柔顺**（`FEED_ARM_GCOMP=1` + `FEED_CLAMP_FLOAT≥0.1`）：夹爪托不住料，
  掉 12~16 mm。机器人重力补偿只补本体，不含负载。
- **负载重力补偿**（`FEED_ARM_PAYLOAD`）：`J^T·(0,0,+mg)` 加在关节上思路合规、
  棒料仍零外力，但刚性伺服把它吃掉、柔顺时又稳不住，指标无改善。
- **交接期提高伺服刚度**（`FEED_ARM_KP=2/3/5/8`）：kp2/kp3 直接 FAIL（5.9/6.1 mm）。
- 夹持力加大（−0.14/−0.16/−0.18）：料被"咬死"跟着手臂走，余量劣化到 4.3~5.0。
- 历史已否：夹紧力 150→300 N、卡爪相位、丝杆联动、软爪几何/包角、预收托圈、
  两段位移闭合、夹爪半开、受载自镗 6 档、低压预夹 5 档、关重力、
  速度前馈/阻尼、单独实体导套、单独先松爪、
  **以理论轴线为目标的径向对中弹簧（作弊，用户否掉）**。

## 下一步（按优先级）

1. **改夹爪站位/工位**：夹持点离卡爪口 55 mm 是柔度的主因。夹到更靠近卡盘处
   （或让卡爪口更近），能直接减小负载力矩；这条路之前因 IK 够不到 47 mm 插入
   而搁置，需要先动工位几何，不是调路径。
2. 若坚持 3.4 mm 以下：真机水平对应的是**更高刚度机械臂**或**更靠近的托料点**，
   在 SO-101 + 当前站位下已到极限。
3. 不要再用"调托架间隙"压指标——它是接触离散化的偶然结果，不稳健。

## 实现要点（改代码前必读）

- `tuned = bool(bushing_ids) and v_jaw and lift and not carry_lock`。
  `_lift = CLAMP_LIFT if tuned else 1.0`，所以历史基线/官方场景仍是旧托力行为。
- `FEED_BUSHING=0` 会把整组交接开关一起短路（`tuned` 依赖导套存在）；
  想做"无导套"对照就直接不传 `--v-jaw/--lift`，或临时改 `tuned`。
- **坑 1：碰撞掩码编译期烘焙。** 构建期 `(0,0)` 再改 `geom_conaffinity` 无接触；
  构建期 `(0,2)` 则立刻 ncon=12。移动几何靠改 `model.geom_pos`（每步重算）。
- **坑 2：常驻闭合环横在搬运路径上**，料撞外壁掉桌面。所以走"张开 40 mm →
  到位合上"。
- `FEED_ARM_KV=5.0` 仍只给送料场景，**全局 `ARM_SERVO_KV` 保持 2.6**。
- 新增开关：`FEED_CLAMP_ANCHOR`（默认 1）、`FEED_CLAMP_TRACE`（默认 0）、
  `FEED_ARM_GCOMP`/`FEED_ARM_GCOMP_GAIN`/`FEED_ARM_PAYLOAD`/`FEED_ARM_KP`
  （默认全关/1）、`FEED_GRIP_CLOSE`（覆盖交接期夹持值）、
  `FEED_BUSHING_ALWAYS`/`FEED_BUSHING_RELEASE`（默认 0）。

## 怎么复现

```bash
python verify_all.py                                    # 9/9
python src/diag_chuck_seat.py                           # 卡盘基准（24 组）
python src/diag_feed_chuck.py --v-jaw --lift --no-lock  # 机械臂送料（新默认）
FEED_CLAMP_TRACE=1 python src/diag_feed_chuck.py --v-jaw --lift --no-lock  # 逐帧跟踪
# 对照：旧默认（无托架、有 mg 托力）
FEED_BUSHING=0 FEED_CLAMP_ANCHOR=0 FEED_CLAMP_LIFT=1 \
  python src/diag_feed_chuck.py --v-jaw --lift --no-lock
```

看 `夹紧前`、`三爪咬合`、`交接外力`、`夹紧瞬态`、`交接位移`、`原地松爪`、`FEED_METRIC*`。
`交接外力 棒料 qfrc 峰值` 必须为 0.000，否则说明又引入了作用在料上的力。

## 文献 / 论坛出处

- 交接期较劲：机械社区《关于机床上下料的问题》
  http://mp.weixin.qq.com/s?__biz=MzA5ODYyNjQxMQ==&mid=2668557070&idx=1&sn=f3038a239ac8e6dec9a446925814f347
  FANUC 软浮动专利 DE69837741T2 https://patents.google.com/patent/DE69837741T2/en
  EVST《Robot Lathe Tending: Chuck Handoff and Interlocks》
  https://www.evsint.com/robot-lathe-tending-chuck-handoff-interlocks/
- hunting / 欠阻尼：Loughborough 综述
  https://repository.lboro.ac.uk/articles/journal_contribution/Motion-control_techniques_of_today_and_tomorrow_A_review_and_discussion_of_the_challenges_of_controlled_motion/14047004/1/files/26506868.pdf
- 卡盘侧：Practical Machinist《3 jaw question for lathe》
  https://www.practicalmachinist.com/forum/threads/3-jaw-question-for-lathe.265817/
  MentorCNC 软爪受载自镗 https://www.mentorcnc.com/en/cnc-lathe-training-lesson-5/

## 同步状态

`verify_all.py` **9/9 通过**（新默认下重跑）。推送需用会话代理端口，
全局配置里的 7897 是坏的：
`git -c http.proxy=$HTTP_PROXY -c https.proxy=$HTTPS_PROXY push origin main`
