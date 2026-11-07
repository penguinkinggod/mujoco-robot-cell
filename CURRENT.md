# 现在在做什么

给下一个接手的人或 agent 看的现场记录。只写"此刻"的状态，不写历史；换手前改这里，不要另开日志。
最后更新：2026-11-07

## 当前目标：交接瞬间的姿态

用户看到的五个现象，都发生在**机械臂松爪 ↔ 卡盘卡爪接手**这一瞬间：
往左上怼一下 / 放掉料自己往下偏 / 再夹紧又动一下 / 找轴心吨吨吨几下 / 最终偏心太大。
上一轮曾把验收收窄成"余量偏心 ≤ 2 mm、偏移第一刀能切掉"，用户明确推翻：
**要的是过程本身不出这几下，不是成品精度。别再只看最终值判 PASS。**

## 本轮结论（用户要求"两条都做"：受载自镗+分段压力、伺服阻尼）

### ✅ 已定案并设为默认：「吨吨吨」= 位置伺服欠阻尼 hunting

文献 B（IEEE IE Magazine 2020 / Loughborough 综述）：接触时环境阻尼不足 →
力控在接触点振荡、被弹回、再接触，即 hunting；解法是**加阻尼，别加比例增益**。
Industrial Monitor Direct 补：Rx/Ry 柔顺、TCP 不在受力点时零件会在孔里来回摇摆，
要锁倾转柔顺 + 夹紧前留足 dwell 让料坐实。

- 改法：`src/scene_full.py` 的 `build_scene()` 新增 `arm_kv` 参数；
  `src/diag_feed_chuck.py` 用 `FEED_ARM_KV=5.0` 单独给送料场景注入。
  **全局 `ARM_SERVO_KV` 必须保持 2.6** —— demo_09「搬运到槽上方 xy<5 mm」
  在阻尼加大后会从 4.0 涨到 5.4~6.6 mm，回归掉到 8/9。
- 实测（默认路径，不带任何环境变量）：

  | 指标 | 改前 | 改后 |
  |---|---|---|
  | 末端抖动 RMS / 最大 | 0.046 / 0.878 mm | **0.031 / 0.377 mm** |
  | 进给 3 轮重新起步次数 | 2/2/1（共 5 次） | **1/1/1（共 3 次）** |
  | 最终同轴 / 倾斜 | 1.26 mm / 0.7° | **0.87 mm / 0.3°** |
  | 余量偏心 | 1.55 mm | **1.05 mm** |
  | 20 N 拉拔 / 重力下沉 | 0.00 / 0.26 mm | **0.00 / 0.00 mm** |

- KV 扫描 2.6/3.5/4/5/6/8：5.0 是最佳点。6 起 grasp_tilt 涨到 1.7°，
  8 直接 27.2°（抓取时料被拽歪），且三爪拧紧起步回升到 8 次。
- **坑**：`ARM_LEAD` 刻意仍用全局 2.6/70，**不要**"顺手统一"成 5/70 —
  上面这组有效数据就是在「伺服 kv=5 + 前馈 2.6/70」下测的，改前馈是未验证变量。

### ❌ 两条阴性（开关保留、默认关闭，别再重试）

1. **软爪受载自镗**（文献 C "Never bore jaws unloaded"）：
   `SM_JAW_BORE_Q` 扫 -0.016/-0.0155/-0.015/-0.0145/-0.014/-0.0135 六档，
   三爪咬合同轴始终 4.33~4.44 mm、夹紧瞬态 z 始终 +2.77~3.02 mm。
   原因：力控执行器会自适应几何平移，镗爪位平移被补偿掉；现有圆弧软爪
   三爪同步差只有 0.087 mm，没有可修的几何误差。
2. **低压预夹 → 端面顶正 → 高压终夹**（`FEED_PRECLAMP=1`）：
   `PRECLAMP_CTRL` 扫 -0.012/-0.015/-0.017/-0.020/-0.024，夹紧瞬态仍 +2.74~3.11 mm。
   **决定性证据：`预夹贴料` 打印时同轴就已经 4.34 mm** —— 推挤发生在三爪
   **第一次碰到料**那一刻，与压力/位移分段无关，所以所有分段方案都打不中。
   -0.020/-0.024 只把最终同轴改善到 0.67/0.70 mm，代价是把"怼一下"提前；
   -0.015 直接 FAIL（余量 3.29 mm），边界很敏感。

### 诊断数据（用 `FEED_BORE_PROBE=1` 复现）

夹紧稳定时三个滑座 `q=[-0.01448, -0.01441, -0.01439] m`，比设计镗爪位
`-0.015` 还**外 0.52 mm**、同步差 0.087 mm。即爪还没收到位就靠偏心料先
接触了某一爪 —— 这是"夹紧又动一下"的几何来源，但改镗爪位修不掉（见上）。

### 补充实验：渐变撤销径向支撑（有进展，但仍 FAIL）

上一轮 `HANDOVER_FIRST + SEAT_RADIAL_K=5000` 的夹紧瞬态只有 z −0.42 mm
（默认 +2.81），但一步撤销支撑会弹飞到 10.27 mm / 13.3°。新增
`FEED_SEAT_FADE` 按 quintic 渐变卸载 400/800 步：

- FADE=400：弹飞消失（最终 2.38 mm）、交接位移 **3.00 mm**（默认 4.77）；
  配合 KV=5 时三爪咬合 **0.63 mm**、夹紧瞬态 **z −0.19 mm** —— 过程姿态最好的一组。
- 仍 FAIL：撤掉力等效支撑后，**25 mm 插入 / 55 mm 悬空**那一段自己跷起来
  （原地松爪轴向偏 7.2°、最终倾斜 2.2°、余量 3.87 mm）。
- **结论：缺的是实体导套/中心架几何（不可撤销的径向约束），不是可撤销的力。**

### "吨吨吨"其它尝试（也否掉）

`FEED_ARM_LEAD` 速度前馈、`FEED_ARM_DAMP` 速度误差阻尼 → 抖动反而略增、
grasp_tilt 涨到 2.2~3.3°；`FEED_FLOAT_DAMP` 柔顺期阻尼 → 几乎无变化；
`FEED_CONVERGE` 夹紧前收敛等待 → 等 32 步料就从 0.08 漂到 1.27 mm、
交接位移 4.77→5.55。**有效的是全局阻尼档位，不是这些局部补偿。**

## 根因收敛（写死，别再发散）

1. "往左上怼 / 夹紧又动" = 三爪**首次接触**就把料从同轴 0.08 mm 推到
   4.3~4.5 mm。已排除：夹紧力 150→300 N、卡爪相位、丝杆联动、软爪几何、
   预收托圈、两段位移闭合、夹爪半开、受载自镗、分段压力、关重力（顶偏照旧）。
2. "放掉料自己偏 / 下沉又回来" = 悬空段没有实体径向支撑，力等效支撑一撤就跷。
3. "吨吨吨" = 伺服欠阻尼 hunting → **本轮已解决**。

## 下一步（只剩这一条，别自己扩）

**实体托料导套几何**：在卡爪口外加一个不撤销的下半圆/整圆导套，给 55 mm
悬空段真实径向约束；配合 `FEED_HANDOVER_FIRST=1 + FEED_SEAT_RADIAL_K=5000
+ FEED_SEAT_FADE=400` 那组过程姿态最好的交接顺序。
或者加大插入深度（之前 35/45/55 mm 因机械臂够不到 FAIL，需先改工位/夹爪站位）。

## 怎么复现

```bash
python verify_all.py                                  # 9/9
python src/diag_feed_chuck.py --v-jaw --lift --no-lock   # 默认路径，看抖动/同轴
FEED_BORE_PROBE=1 python src/diag_feed_chuck.py --v-jaw --lift --no-lock
SM_JAW_BORE_Q=-0.0145 python src/diag_feed_chuck.py --v-jaw --lift --no-lock
FEED_PRECLAMP=1 FEED_PRECLAMP_CTRL=-0.020 python src/diag_feed_chuck.py --v-jaw --lift --no-lock
FEED_HANDOVER_FIRST=1 FEED_SEAT_RADIAL_K=5000 FEED_SEAT_FADE=400 \
  python src/diag_feed_chuck.py --v-jaw --lift --no-lock
```

看 `夹紧前`、`夹爪先松`、`预夹贴料`、`三爪咬合`、`夹紧瞬态`、`受载爪位`、
`撤销导套`、`交接位移`、`末端抖动幅度`、`分段起步/静止`、`FEED_METRIC*`。

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

`verify_all.py` **9/9 通过**。默认行为已改（送料场景阻尼 5.0），其它实验
开关默认全关闭。推送需用会话代理端口，全局配置里的 7897 是坏的：
`git -c http.proxy=$HTTP_PROXY -c https.proxy=$HTTPS_PROXY push origin main`
