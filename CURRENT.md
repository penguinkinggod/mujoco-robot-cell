# 现在在做什么

给下一个接手的人或 agent 看的现场记录。只写"此刻"的状态，不写历史；换手前改这里，不要另开日志。
最后更新：2026-11-07

## 当前目标：交接瞬间的姿态，不是成品余量

用户 2026-11-07 推翻了上一轮"偏移可以切掉、可以无视"的收尾判断。
他看到的四个现象都发生在**机械臂松爪 ↔ 卡盘卡爪接手**这一瞬间：

1. 机械臂放开、卡爪接手时，料被**往左上怼一下**；
2. 放掉料会**自己往下偏移**；
3. 再夹紧**又动一下**；
4. 机械臂找轴心时**吨吨吨几下**；
5. 最终**偏心太大**。

上一轮把验收标准收窄成"整根料最坏偏心 ≤ 单边余量"（`stock_ecc_mm=1.55 ≤ 2.0`），
那管的是第一刀能否见光，**管不到过程姿态**。用户要的是过程本身不出这几下。

## 文献 / 论坛调研结论（2026-11-07 检索）

### A. 交接期"两边都夹紧 = 较劲"（最直接对口）

- **机械社区《关于机床上下料的问题》**（社友讨论）：原帖就写「机器人送入，然后
  卡盘夹紧，然后机器人再松开。造成两者交替时，**有一个两者都夹紧的状态，较劲**。
  也是定位精度不高的原因」。解法：①「放进零件后，**松开夹爪但不移动手臂**，
  等卡盘夹紧后再移走」；②「气爪端面有**弹簧定位**，松开后端面不要移动，靠弹簧
  把工件顶在主轴卡爪定位面上，**三爪锁紧过程会自动找正**」；③ 短轴用弹簧推料机构，
  长轴加尾座顶紧；④「FANUC **软浮动**功能可以解决机器人同机床夹具别劲儿」；
  ⑤ 夹爪内孔精磨、示教用直角坐标系保证 6 轴中轴线与卡盘中轴线平行；
  ⑥ 每次装夹后程序校验是否装正。
  http://mp.weixin.qq.com/s?__biz=MzA5ODYyNjQxMQ==&mid=2668557070&idx=1&sn=f3038a239ac8e6dec9a446925814f347
- **FANUC 软浮动专利 DE69837741T2**：检测到接触后开启 soft-floating，机器人被动
  沿外力方向移动，让工件贴合卡盘定位面；**卡爪此时仍在退回位、尚未闭合**。
  https://patents.google.com/patent/DE69837741T2/en
- **Fanuc LR Mate 虎钳交接案例**：机器人释放后夹具夹在未就位料上 →「爪可能把
  毛坯夹得**倾斜或旋转**」；夹爪仍夹持时夹具闭合 →「不对中对腕部造成负载…
  仅当必须保持方向时才用，且必须配合**柔顺性**」。解法是推料器在夹紧闭合前一直压住、
  并就位信号互锁夹紧指令。
  https://industrialmonitordirect.com/tr/blogs/knowledgebase/should-a-fanuc-lr-mate-release-the-part-before-vise-clamp
- **EVST《Robot Lathe Tending: Chuck Handoff and Interlocks》**：「blank can appear
  centered in the jaws while remaining off the locating face because of …
  **spring in the handoff**」；要求 prove seating at the locating face，并明确
  **不要用主轴/卡盘把不确定的料拉正**。
  https://www.evsint.com/robot-lathe-tending-chuck-handoff-interlocks/

### B. "吨吨吨" = 力控 hunting / 欠阻尼，不是没对准

- **IEEE Industrial Electronics Magazine 2020 / Loughborough 综述**：接触时环境阻尼
  不足 → 力控在接触点振荡、被弹回、再接触，反复即 **hunting**；标准解法是插入
  **速度反馈环**，并用阻抗控制按任务阶段调度刚度（自由空间高刚度、接触段低刚度）。
  https://repository.lboro.ac.uk/articles/journal_contribution/Motion-control_techniques_of_today_and_tomorrow_A_review_and_discussion_of_the_challenges_of_controlled_motion/14047004/1/files/26506868.pdf
- **Industrial Monitor Direct《Robot Force Mode Seating in CNC Chuck: Fixing Wobble》**：
  Rx/Ry 柔顺且 TCP 在法兰（不在受力点）时，推力线横向偏移产生力矩，机器人倾斜去
  归零力矩 → 零件边缘在孔内**来回摇摆**而非轴向推进；Z 推力太低推不动摩擦，
  倾斜轴继续 hunting。解法：**TCP 移到接触点** + **锁住 Rx/Ry 倾转柔顺** +
  提高 Z 推力 + **夹紧前留足 dwell 让料坐实**（clamping before the part has
  slipped to full depth locks a proud part into the jaws）+ 夹爪要有让料微滑的 slack。
  https://industrialmonitordirect.com/blogs/knowledgebase/why-does-the-robot-wobble-when-force-seating-a-part-in-a-chuck
- **Stiffness Scheduling**：振荡是**欠阻尼**，先加微分/阻尼，不要加比例增益；
  接触段关掉或钳制积分项防 windup；控制器须在一个控制周期内收敛。
  https://lillytechsystems.com/ai-physical-ai/control-basics-for-ai/stiffness-scheduling.html
- 学术侧主动柔顺 RCC：Mobedi et al., RO-MAN 2020,
  DOI 10.1109/RO-MAN47096.2020.9223515，peg-in-hole 有间隙/角度偏差时在线调节
  约束方向柔顺度。https://zenodo.org/record/4020930

### C. 夹紧时料被推动 / 偏心放大（卡盘侧）

- **Practical Machinist《3 jaw question for lathe》**：磨损卡盘会向外让、
  **把料推离挡块**；爪尖碰到台阶圆角也会顶出。车间解法：橡胶锤轻敲、
  **装在刀塔上的弹簧推料器在循环中把料重新顶正**（推着料 + M 代码触发卡盘动作）、
  尾座顶紧两下、爪口做微锥度。
  https://www.practicalmachinist.com/forum/threads/3-jaw-question-for-lathe.265817/
- **MentorCNC 车床装夹教程**：软爪必须**在夹紧压力下自镗**（boring jaws under
  load，用镗爪环/spider 预加载），"Never bore jaws unloaded"；未镗软爪是线接触，
  卡盘 0.02–0.03 mm 跳动会放大到工件 0.1–0.2 mm。
  https://www.mentorcnc.com/en/cnc-lathe-training-lesson-5/
- **浮动动力卡盘专利 CN121535233A**：浮动桥自适应均衡各爪夹紧力，明确针对
  「受力不均导致工件**变形或位移**」。https://eureka.patsnap.com/patent/CN121535233A
- 长轴/悬伸必须支撑：中心架、跟刀架、尾座活顶尖、扇形爪/弹簧夹头（360° 接触）。
  EVS 曲轴上下料案例也是「三爪夹一端 + 远端支撑」，且"让上料台决定位置，
  机器人只重复抓取"。https://www.evsrobot.com/?p=7639

## 本轮实测（代码里已留对照开关，默认关闭）

按 A 实现 `FEED_HANDOVER_FIRST=1`：进给到位 → **夹爪先完全松开**（手臂原地、
soft floating）→ 端面轴向弹簧定位（等效挡块+推料器）+ 径向弹性支撑（等效导套）
→ 停留 dwell → 三爪闭合 → 撤支撑 → 退出。

| 对照 | 夹爪先松时同轴 | 夹紧瞬态 | 结果 |
|---|---|---|---|
| 旧顺序（当前默认） | — | z +2.81 mm，y/z (+3.37,+2.81) | PASS，交接位移 4.99 mm |
| 先松爪，无径向支撑 | 4.7~8.0 mm（料被弹走，**正是"往左上怼"**） | z −6.09 / +5.50 | FAIL |
| 先松爪 + 径向支撑 k=200 | **0.00 mm** | z −4.07 | FAIL（余量 2.22） |
| 先松爪 + 径向支撑 k=1000 | **0.00 mm** | z −4.66 | FAIL |
| 先松爪 + 径向支撑 k=5000 | **0.00 mm** | z −0.42，交接位移 **3.30 mm** | 撤支撑后料飞了：10.27 mm / 13.3° |

**结论（重要）**：
- "往左上怼"的根因已定位 = **夹爪张开瞬间料失去径向约束被弹开**。
  加径向支撑后该瞬间同轴从 7.5 mm 变 **0.00 mm**，是文献 A/C 的直接验证。
- 但**三爪闭合仍把料往下推约 4 mm**，撤销硬支撑还会弹飞 → 还缺真机的
  **软爪受载自镗**和/或**低压预夹→高压终夹**（文献 C）。纯力控参数已到头，
  别再扫 `FEED_CLAMP_FLOAT` / `SEAT_*` 增益。

## 已否掉，别再试

夹紧力 150→300 N、卡爪相位 30/90/150/210、预收托圈、两段位移闭合、夹爪半开、
纯柔顺增益扫描（本轮 6 组全 FAIL）、关重力对照（顶偏与重力无关）。

## 下一步（按文献优先级，别自己扩）

1. **软爪在受载状态下自镗**：`SM.JAW_ARC_Q0` / 圆弧半径按实际夹紧力下的爪面
   位置标定，而不是理想 `SHAFT_D/2`。对应"夹紧又动一下"。
2. **低压预夹 → 停留/顶正 → 高压终夹**两段**压力**（不是两段位移，之前试的
   `CLAMP_TWOSTAGE` 是位移斜坡，已否掉），对应 Practical Machinist 的
   "弹簧推料器 + M 代码重新顶正"。
3. **实体径向导套几何**（不是力等效），撤掉时不会储能弹飞。
4. "吨吨吨"单列处理：给位置伺服加阻尼/速度前馈、夹紧前保证伺服已收敛，
   别再加比例增益。

## 怎么复现

```bash
python verify_all.py                                  # 默认路径 9/9
python src/diag_feed_chuck.py --v-jaw --lift --no-lock
FEED_HANDOVER_FIRST=1 FEED_SEAT_RADIAL_K=200 python src/diag_feed_chuck.py --v-jaw --lift --no-lock
```

看 `夹紧前`、`夹爪先松`、`三爪咬合`、`夹紧瞬态`、`交接位移`、
`FEED_METRIC`、`FEED_METRIC_HANDOVER`、`FEED_METRIC_MACHINING`。

## 同步状态

默认行为未变，`verify_all.py` 9/9。推送需用会话代理端口：
`git -c http.proxy=$HTTP_PROXY -c https.proxy=$HTTPS_PROXY push origin main`
（全局配置里写死的 7897 是坏的）。
