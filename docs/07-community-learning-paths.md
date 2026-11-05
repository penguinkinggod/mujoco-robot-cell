# 07 · 社区里的学习路径（别人怎么走的）

> 这一篇不写代码，只回答一个问题：**别人是怎么规划这条路的，我有没有走偏。**
>
> 所有链接都是 2026-11 实测可访问的。标注「⚠️仅摘要」的，是站点反爬、只看到搜索摘要，不作为强证据。

## 先给结论

**你没走偏，而且已经跨过了最劝退的两道坎。**

社区路径里最劝退的两件事是：①环境搭不起来 ②FK/IK 没有直觉。你现在跑通了 Panda 抓取 + 可达性检查，这两件事都过了。

**你当前唯一的真缺口是「控制器」。** 这是下面几乎所有路线都共同强调、而你现在完全没碰的一块。

## 一、中文社区三条硬证据

### 1. 古月居《MuJoCo 机器人强化学习仿真入门》

[课程页](https://www.guyuehome.com/detail?id=1825501975181848577) · 古月居官方（国内 ROS 教育头部社区）

10 节课的顺序，本身就是一份教学法声明：

```
1  课程简介
2  环境搭建
3  Python 和 XML 基础
4  建立机器人模型
5  控制器接口
6  搭建 PID 模块
7  轨迹跟踪
8  强化学习基础      ← RL 在这里
9  Gym 实例
10 机器人 RL 实例
```

**强化学习排在第 8 节。前 7 节全是建模 + 控制器 + PID + 轨迹跟踪。**

这条证据的价值在于：它是一个**卖课的**，如果 RL 好卖，它会把 RL 放前面。它没有。因为把 RL 放前面，学生根本跑不通。

> 对照你现在的进度：你在第 4–5 节之间。**第 6、7 节（PID + 轨迹跟踪）你还没做。**

### 2. Datawhale《动手学具身智能》

[文档站](https://datawhalechina.github.io/dive-into-embodied-ai/) · [GitHub](https://github.com/datawhalechina/dive-into-embodied-ai/blob/master/README.md)

Datawhale 是国内最规范的开源 AI 学习社区（《蘑菇书》那个团队）。这个项目的结构是：

```
具身导论
理论基础
  ├─ 大脑：强化学习决策 / VLA / World-Model
  ├─ 小脑：RL控制 / 控制器(PID-LQR-MPC-阻抗) / 运动规划
  ├─ 感知：视觉VLM / 标定与sim2real
  └─ 工程底座：仿真工具 / ROS2 / 数据工程
项目实战（6 个仿真实战项目）
```

**两个关键事实：**

**① 6 个仿真实战项目，100% 用 MuJoCo / mjlab，一个 Isaac 都没有。**
四足机器人、Pupper、Quaddle、MicroDuck、MuJoCo 机械臂 + DDPG、ACT 双臂。

这是对「要不要上 Isaac Sim」最有力的回答——**国内最规范的学生向课程，集体选择了 MuJoCo。**

**② 它把「控制器（PID/LQR/MPC/阻抗）」和「运动规划」跟 RL 并列放在「小脑」下面，不是放在 RL 后面。**

意思是：控制器不是学 RL 的前置，控制器**本身就是一条独立的主线**。

它还有三个交互式 playground，可以拖鼠标直接玩：

- [FK Playground](https://datawhalechina.github.io/dive-into-embodied-ai/cs123/fk-playground) —— 拖关节看齐次变换矩阵怎么变
- [IK Playground](https://datawhalechina.github.io/dive-into-embodied-ai/cs123/ik-playground) —— 拖目标点看 DLS 迭代，**把目标拖出可达范围，看它怎么"贴边追"**
- [PD Playground](https://datawhalechina.github.io/dive-into-embodied-ai/cs123/pd-playground) —— 调 kp/kd 看超调和稳态误差

> 那个 IK Playground 的「拖出甜甜圈外圈」正好是你上次问的「为什么会错」的可视化版本。**建议现在就去玩五分钟。**

### 3. Xbotics《Roadmap to Embodied AI》

[GitHub](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) · Xbotics 社区 · CC BY 4.0

结构：具身综述 → 学习路线（**10 条 4–8 周路线**）→ 基础学习 → 经典算法 → 前沿 → 仿真 → 开源实物 → 人物图谱 → 公司图谱。

它比前两个强的地方是**有验收标准**。每条路线都要求给出「前置 → 周计划 → 里程碑 → 验收」。

它的求职 Roadmap 里有一句话值得抄下来：

> 8 周内交付 1 个可演示的 Demo；把 Demo 写成「问题 → 方法 → 指标 → 链接」四段式项目描述。

**这正是你 README 里那张实测数字表在做的事。** 你已经在对的格式上了。

缺点也说清楚：10 条路线容易让人选择瘫痪，而且偏就业导向。

### 4. 一条要打折的：《机器人仿真完整知识树》

[博客园](https://www.cnblogs.com/dokibook/p/22475106) · 作者 luke0366

**这篇文章自己标了「AI解答摘抄」。** 我读了全文，确认：

- 结构最全（四大层级，从数学基础到工业仿真）
- 但堆了 10+ 个仿真器（RobotStudio / RoboGuide / KUKA Sim / ADAMS / Tecnomatix），**一个可跑代码都没有**
- 分阶段的月数没有任何依据

**只当目录用，别当路径用。**

它有一个判断是对的：把「坐标变换 / 齐次矩阵 / 四元数」列为数学基础必学，并注明「重点，解决姿态歧义」。

## 二、英文社区的官方顺序

### MuJoCo 官方 notebook 的顺序

官方文档的 tutorial 链是有顺序的：

```
introductory  →  Model Editing  →  rollout  →  LQR  →  least-squares  →  MJX  →  differentiable physics
```

这个顺序说明官方的判断：**先搞懂 mjModel / mjData 与模型编辑，再谈控制与学习。**

> 你现在做的事（`MjSpec` 建模型、`mj_step` 推进、`qfrc_bias` 读重力）正好对应前两篇。
> 而「LQR」那一篇——单腿平衡人形——就是控制器。

### MuJoCo Playground（要走 RL 时的入口）

[GitHub](https://github.com/google-deepmind/mujoco_playground)

门槛写得很明确：**Python ≥3.10、CUDA 12 的 NVIDIA GPU、JAX。**

你的 RTX 5070 Ti Laptop 12 GB 够，但要装 JAX CUDA 版。教程顺序：DM Control 入门 → Locomotion → Manipulation → Vision。

**别现在装。** 这是走 RL 路线时才需要的。

### LeRobot（要走模仿学习/真机时的入口）

[GitHub](https://github.com/huggingface/lerobot)

流程：装 → 组装校准 SO-101 → **录约 50 条示教** → `lerobot-train --policy.type=act` → `lerobot-eval`。

**注意它要真机械臂。** 纯仿真阶段用不上。

## 三、社区公认「必须先学」vs「可以跳过」

### 必须先学

| 内容 | 谁说的 |
|---|---|
| **坐标变换 / 齐次矩阵 / 四元数** | 博客园知识树列为必学并标「重点，解决姿态歧义」；Xbotics 基础学习第一位就是「坐标系」；Datawhale 把「坐标系、外参误差放大」写进 sim2real 章节 |
| **mjModel / mjData 分离 + 广义坐标** | MuJoCo 官方 overview 专设「Clarifications」解释「习惯游戏引擎的人会觉得广义坐标反直觉」 |
| **FK / IK + 雅可比** | ETH Robot Dynamics 课程结构：旋转表示 → 串联链 → 雅可比 → 动力学 → 控制 |
| **Python + NumPy** | 全部路线的前提 |

**这四条你都在做了。** 坐标变换你在 demo_02/demo_04 里天天用；mjModel/mjData 你在 `reachability.py` 里用得很熟；雅可比你在 IK 里手写过。

### 可以跳过（至少现在）

| 内容 | 为什么跳过 |
|---|---|
| **ROS2 + Gazebo** | 那是移动机器人/导航栈的路。你走的是机械臂抓取主线，不同支。等要做移动操作再回补 |
| **Isaac Sim / Isaac Lab** | 门槛是 CUDA 12 + NVIDIA GPU + 一堆新概念。**Datawhale 6 个实战项目全用 MuJoCo 就是最好的反证** |
| **工业离线仿真**（RobotStudio / KUKA Sim / ADAMS） | 跟具身智能主线无关，纯干扰。你的 SolidWorks/Mastercam 背景反而容易在这里被吸走注意力 |
| **「6 个月 / 12 周」这类时间表** | 各家给的月数互相矛盾且无来源。用「4–8 周一条线 + 一个可演示 Demo」当验收单位更实际 |
| **付费课** | 你现在缺的不是课程，是把 Panda 从「能抓」推到「能按轨迹抓」的那几十行代码 |

### 新手三个坑（有实证）

**坑 1：一上来学强化学习。**
最硬的证据就是古月居的课程编排——RL 是第 8 节。前 7 节全是建模和控制。**因为 RL 跑不通的时候，你分不清是算法问题还是环境问题。**

**坑 2：一上来装 Isaac Sim。**
Playground 官方门槛写明 CUDA 12 + NVIDIA GPU；Datawhale 全用 MuJoCo。

**坑 3：只看路线不动手。**
Xbotics 求职 Roadmap 明写要「8 周内交付 1 个可演示 Demo」。

**坑 4（中文特有）：把同一篇文章当成多方共识。**
中文侧有大量「一键分发」账号，同一份内容同步到 CSDN / 知乎 / 掘金 / 百家号。你会以为「大家都这么说」，其实是同一个人说了四遍。**看作者，不看平台数。**

## 四、中文社区活跃入口

| 入口 | 状态 |
|---|---|
| [古月居](https://www.guyuehome.com/) · [社区](https://www.guyuehome.com/community) | 实测在线。「ROS学习」1364 人 / 816 问答，但「具身智能」板块只有 6 人 / 8 问答。**规模真实但具身板块很小** |
| [古月学院](https://class.guyuehome.com/page/1628905?navIndex=0) | ROS2 入门 21 讲、Gazebo 移动机器人仿真等，付费 |
| [深蓝学院](https://www.shenlanxueyuan.com/courselist/ar) | 课程真实在售，但「剩余名额」是营销话术，无法核实 |
| [Datawhale](https://datawhalechina.github.io/dive-into-embodied-ai/) | 有组队学习机制，但 README 明写「最新一期报名入口：施工中」。**当前无法核实有在跑的学习营** |
| [Xbotics](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) | 入口是 GitHub Issue（AMA 模板） |
| B站 同济子豪兄 | LeRobot 官方 README 点名收录「LeRobot + SO-ARM101 中文教程-同济子豪兄」，**这是最可靠的背书** |

**QQ 群 / 微信群 / Discord：没有找到可核实的公开入口。** 不编。

## 五、对你现在的判断

### 你在哪

对照古月居 10 节大纲，你在**第 4–5 节之间**：建模会了（`MjSpec`），控制器接口摸到了（`ctrl` + PD），但**没有系统做过 PID 和轨迹跟踪**。

### 该做的三件事（按顺序）

**① 补引擎语义，不补新工具**（半天到一天）

去读官方 notebook 的 Model Editing + rollout 两篇。你现在是「能跑但说不清为什么」。Playground 和 LeRobot 全建立在这层之上。

具体自测：能不能不看代码说出 `mj_forward` 和 `mj_step` 的区别？`qpos` 和 `qvel` 为什么长度可能不一样？`qfrc_bias` 里到底装了什么？

**② 接控制器——这是你唯一的真缺口**（这是重点）

古月居第 5–7 节做的事，也是 Datawhale「小脑」那一列、Xbotics「规划控制」线共同强调的。

具体做什么：**让 Panda 的末端走一条直线，或一个圆。**

这件事看起来简单，但它是分水岭：

- 现在你的做法是「算一个目标关节角 → 喂给 ctrl → 等它到」
- 轨迹跟踪要求「每一步都重新算 → 每步都修正 → 末端沿指定路径走」

**这正好也回答了你上次问的「为什么会错」——闭环。** 开环执行必然累积误差，闭环才能压住它。

**③ 然后才选学习方向**（二选一，别同时开）

- 想走 RL → MuJoCo Playground 的 `PandaPickCube`（你的 12 GB 显卡够）
- 想走模仿学习/真机 → LeRobot + SO-101

### 一句话总结

> **社区的路和你的路是同一条。你已经在正确的路上，只是停在了「能抓」这一步，下一步是「能按轨迹抓」。**

## 附：这份调研的可靠性

- 中文侧 8 条里，古月居、Xbotics、Datawhale 三条是**全文核实**的硬证据
- 知乎两条 + 坐标变换专文，抓取被反爬拦截，**只看到搜索摘要，已降权**
- B站课程页和 CSDN 那篇，已核实但内容质量低，**已降权**
- 英文侧全部 fetch 成功
- **本机网络限制**：GitHub / readthedocs / HuggingFace 直连返回 000，只有 `*.github.io` 和国内站点可达。所以部分链接是通过搜索摘要 + 镜像核实的，已在正文标注
