# mujoco-robot-cell

**一个 MuJoCo 机器人工作站，一站一站往上搭。**
*A MuJoCo robot cell, built one station at a time.*

数字截至 2026-11-06。

参数化分析了夹爪夹不同直径的圆柱，照着结果改了一版：不换爪，能夹的料从 Φ20 细到 Φ18；另备一付按 Φ15 镗的爪，Φ15 到 Φ25 全过。
这一站还是车床上下料，SO-101 夹住棒料尾段，把自由端沿 +x 送进三爪卡盘 25 mm；搬运段没有
运动学锁，卡盘那 163 N 是量出来的接触力。

> A parametric sweep over jaws and stock diameter. With the revised jaws, Φ18 stock is held
> without changing jaws (was Φ20). The station is still lathe tending: an SO-101 grips the
> tail of a bar and feeds the free end 25 mm into a 3-jaw chuck; carry is physics, not a lock.

![送料全过程](https://github.com/penguinkinggod/mujoco-robot-cell/releases/download/media-v1/now_feed.gif)

同轴 0.25 mm · 倾斜 0.2° · 30 点接触 · 20 N 拉拔 0.00 mm · 三爪 163.4 N · 末端抖动 0.045 mm。
`verify_all.py` 9/9，约 25 秒，纯 CPU，不需要显卡。

## 30 秒跑起来

```bash
git clone https://github.com/penguinkinggod/mujoco-robot-cell.git
cd mujoco-robot-cell

uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

python verify_all.py
```

Windows / Linux / macOS 都能跑。实测版本 Python 3.12.15、mujoco 3.17.0、numpy 2.5.4、
scipy 1.18.1、matplotlib 3.11.2、imageio 2.38.1。

想直接看画面要起两个进程，缺一个窗口就不动：

```bash
python src/live_bridge.py view        # 常驻 3D 窗口，负责画
python src/live_bridge.py run --loop  # 跑仿真并推帧，负责演
```

窗口不热重载。改了场景要关掉重开，不然看到的还是旧场景。

## 这次做到了什么

这一轮只做了一件事：把「换一根料还夹不夹得住」扫清楚。五档直径，两种爪。

不换爪，还是原来那套 Φ20 的弧面：Φ18 到 Φ25 都能夹，同轴 0.14~0.27 mm、倾斜 0.35~0.52°、
20 N 拉拔 0.00 mm、30 个接触点。Φ15 不行。Φ20 的弧面半径 10.2 mm，Φ15 的料半径 7.5 mm，
中间空着 2.7 mm，三爪得往里多走这一截才够得着；而这一截行程又把爪的端点送进彼此的缝里，
三爪互相顶到 1600 N。所以这是行程不够，不是包角不够，缩包角治不了。

换爪就通了。按该直径重新镗一套爪，Φ15 到 Φ25 全过：同轴 0.11~0.21 mm、倾斜 0.40~0.49°、
拉拔 0.00 mm、接触点 30 个。

| 直径 | 不换爪 | 换爪 |
|---|---|---|
| Φ15 | 夹不住（爪互顶 1600 N） | 过 |
| Φ18 | 过 | 过 |
| Φ20 | 过 | 过 |
| Φ22 | 过 | 过 |
| Φ25 | 过 | 过 |

所以这套卡盘天生的通用范围是 Φ18~Φ25；要夹 Φ15，只能另备一付细料爪。

顺手把爪内侧从 104° 缩到了 88°（半包角 44°）。不缩的话 Φ18 以下三爪先互相顶上，Φ15 时
互顶力 1274 N，重新镗爪也照样顶。缩过头也不行：42° 互顶是没了，主线三爪力比却从 1.6 涨到 4.24；
46° 又回到 845 N。44° 是两头都能过、还留 2° 余量的那个点，代价只是每爪少了 16° 接触面。

重跑：`python src/diag_jaw_face_sweep.py --full`，结果落 `_build/jaw_face_sweep_full.csv`。
全部读数和扫的过程在 [06-sweeps.md](docs/cell/06-sweeps.md)。

## 已经做到了什么

九项检查一次跑完，任何一项不过，退出码就不是 0。各验什么看
[07-checks.md](docs/cell/07-checks.md)，逐日时间线看 [08-log.md](docs/cell/08-log.md)，
目录和 demo_00–07 看 [09-layout.md](docs/cell/09-layout.md)。近期三件：

- 全物理送料（V 槽 + 无锁 + 卡盘外壳实体碰撞）：同轴 0.25 mm、倾斜 0.2°、30 点接触、163.4 N。
- 同一流程跑在官方 SO-101 tabletop 世界里，四个机位都在，数字一样。
- 换爪之后 Φ15~Φ25 五档全过；不换爪从 Φ18 起。就是这一轮做的。

只想单独复现全物理送料那一项：

```bash
python src/diag_feed_chuck.py --v-jaw --lift --no-lock
```

## 下一步

1. 先修摆位，把力的绝对值做成能引用的数。变径扫描为了只动直径这一个变量，把工件直接摆进
   卡盘，它还同时顶着卡盘内壁，法向力 1~4 kN、各档都一样。A/B 对照会抵消，但那张表里的牛
   不是真机夹紧力。改成悬臂装夹重扫一遍，绝对力才敢往报告里写。
2. 把参数扫描扩成一份能引用的报告。直径这维有 A/B 对照了，爪型和摩擦两维还没扫。顺带把
   不换爪夹 Φ15 的 2.7 mm 缺口定死：备一付细料爪，或者接受换爪，别再调包角试。
3. 相机标定加视觉引导抓取，位姿从图像来，不再手给。
4. 手势遥操作，顺带录 `(观测, 动作)`；那批序列出来之后才是 BC / ACT。

四个工位的长线计划在 [13-roadmap.md](docs/cell/13-roadmap.md)。

## 为什么做这个

想有个能逐日迭代的地方，不是又一份一次性 demo。每天学的东西有地方落，指标能复现，失败也
记下来。做久了，才可能拿出「泛化」上的东西：换个位姿也能成，而不只是复现固定路线。

仿真基础设施别人已经做完了，这部分不重做。intrinsic-omts 把 CNC 上下料的参考设计开源了，
LeRobot 自带 SO-101 的 MuJoCo 环境和三个机位，抓取到底成不成立在文献里也是标准项。
这里只留三件别人不做的：具体工位（三爪卡盘加轴向进给，基准任务都是桌面方块）、
几何归因（这个爪在这个料上为什么会失败）、能一键跑的判据。

为什么用 MuJoCo 不用 Isaac Sim 看 [01-why.md](docs/cell/01-why.md)，相关工作的对照表看
[12-related.md](docs/cell/12-related.md)。

## 目前的局限

挑五条。完整十条在 [10-limits.md](docs/cell/10-limits.md)。

1. 变径扫描的力只有相对意义。工件还同时顶着卡盘内壁 1~4 kN，各档一样，别拿去跟手拧卡盘的
   kN 比。首屏那个 163 N 是送料末态量的，只数爪与料之间的接触，两个数别对着减。
2. 163 N 在物理上算轻的。真机手拧 Φ100 三爪每爪 3~5 kN。现在封顶在 150 N 是为了不让爪压穿
   卡盘，不是夹不动的上限。
3. 默认路径的搬运段还在用 `kinematic_lock`。全物理那条要加 `--v-jaw --lift --no-lock`。
   默认值没换，因为几项旧判据建在旧数字上，换了会打断历史对比。
4. 摩擦系数偏乐观。爪 1.8、工件 1.2，磨光的车削件没这么高，还没按真实材料标定。
5. 没有 sim-to-real，没有自动避障；视觉、遥操作、模仿学习都还没开始。

## 踩过的坑

拣五个典型的，每条都带数字。全部十九条按抓取几何、卡盘夹紧、判据测量、工程工具分了四组，
在 [11-pitfalls.md](docs/cell/11-pitfalls.md)。

- 假夹紧力比没有力更危险。方盒爪闭合时互相穿插，量出 36 N，看着能过判据；换成弧爪，真力
  只剩 3.4 N。也就是说之前一直没夹住，只是数字好看。
- `td.qpos[:] = 0.0` 把带 freejoint 的工件挪到了原点，于是所有槽口都是对着「在原点的一个
  幽灵工件」算的。在这一行没修之前得出的槽深结论全部作废。
- 四元数转角会把「绕自身轴滚转」算成倾斜。读到 77° 差点去修一个不存在的问题，卡盘一夹就
  回 0°。判据得用 `arccos(|axis·x|)`。
- 力限是个够不着的天花板。`forcerange` 只在位移误差乘 `kp` 超过它时才生效；目标只比实际停位
  深 4.2 mm 时出力恒定 84.5 N，把上限从 150 抬到 1200 N 一个数都不变。
- 伺服没有阻尼项。`so101_new_calib.xml` 里 `Kd = 0`，阶跃超调 45.6%、2.8 Hz 振铃；
  轨迹再光滑，伺服自己会弹。

## 参考与声明

用了谁的模型、什么协议、怎么引用，都在这一节。相关工作的完整对照表在
[12-related.md](docs/cell/12-related.md)。

### 第三方代码与资产

- `models/panda/`：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，Apache-2.0
- `models/_tmp_so101/lerobot_env_so101-0.4.0/`：[LeRobot](https://github.com/huggingface/lerobot) 的 SO-101 环境包，Apache-2.0
- `models/simready/`：工作台 / 夹具 / 工件资产

详见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。

### License

代码 MIT，见 [`LICENSE`](LICENSE)。第三方部分遵循各自协议。

### 引用

```bibtex
@misc{mujoco-robot-cell,
  title  = {mujoco-robot-cell: a MuJoCo robot cell for lathe tending and manipulation},
  author = {penguinkinggod},
  year   = {2026},
  note   = {First station: a SO-101 grips a Phi20 cylinder by its tail, feeds the free end
            25 mm into a 3-jaw chuck with bored arc jaws, and releases without a kinematic
            lock -- 0.25 mm coaxiality, 30 real contacts, 163 N measured grip},
  url    = {https://github.com/penguinkinggod/mujoco-robot-cell}
}
```

### 语言 / Language

正文用中文，关键结论（首屏定位、核心指标、已知限制）中英双写。不做全文双语是因为这个仓库
每天在改，逐句翻译会拖慢速度，而机翻质量的英文比没有英文更劝退。哪一节需要英文，开 issue 说一声。

Body text is Chinese; key results are written in both languages. Full bilingual docs would slow
down daily iteration. If you need a section in English, open an issue.
