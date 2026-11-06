# mujoco-robot-cell

**一个 MuJoCo 机器人工作站，一站一站往上搭。**
*A MuJoCo robot cell, built one station at a time.*

数字截至 2026-11-06。

参数化分析了夹爪夹不同直径的圆柱，照着结果改了一版：不换爪，能夹的料从 Φ20 细到 Φ18。
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

这一轮扫了「换一根料还夹不夹得住」，五档直径、两种爪。顺手把卡盘从实心圆柱改成带
Φ30 通孔的环，扫描摆位也换成跟主线一致：插入 25 mm、外留 55 mm。改完才发现旧那版数据
是假阳性，工件插进实心卡盘 35 mm，被穿透罚力顶着，看着夹住了，其实是卡盘实体在撑。

新摆位下十组重测：

| 直径 | 不换爪（Φ20 爪，44°） | 按径另镗爪（44°） |
|---|---|---|
| Φ15 | 挂（互顶 48 对） | 挂（互顶 42 对） |
| Φ18 / 20 / 22 / 25 | 过 | 过 |

八组过：同轴 0.21~0.34 mm、倾斜 0.03~0.13°、30 点接触、20 N 拉拔 ≤0.03 mm、
三爪力 139~256 N。卡盘内壁接触力从 1~4 kN 降到 0，这些牛现在可以当数引用。

Φ15 两组都挂在爪互顶上，不是夹不住。半包角 44° 是按 Φ20 定的，料细到 15 mm，
相邻弧片的角向缝不够，三爪先顶到彼此；把包角缩到 42° 以下，Φ15 定制爪就过
（160 N、同轴 0.40 mm），但主线三爪力比会挂，所以没改全局。

重跑：`python src/diag_jaw_face_sweep.py --full`，数据落 `_build/jaw_face_sweep_full.csv`。
旧摆位那张表和包角扫描的过程留在 [06-sweeps.md](docs/cell/06-sweeps.md)，已标作废。

## 已经做到了什么

九项检查一次跑完，任何一项不过，退出码就不是 0。各验什么看
[07-checks.md](docs/cell/07-checks.md)，逐日时间线看 [08-log.md](docs/cell/08-log.md)，
目录和 demo_00–07 看 [09-layout.md](docs/cell/09-layout.md)。近期三件：

- 全物理送料（V 槽 + 无锁 + 卡盘外壳实体碰撞）：同轴 0.25 mm、倾斜 0.2°、30 点接触、163.4 N。
- 同一流程跑在官方 SO-101 tabletop 世界里，四个机位都在，数字一样。
- 卡盘改成带 Φ30 通孔的环，扫描力从相对值变成能引用的数；不换爪 Φ18~Φ25 全过。就是这一轮做的。

只想单独复现全物理送料那一项：

```bash
python src/diag_feed_chuck.py --v-jaw --lift --no-lock
```

## 下一步

1. 把 44° 的选择依据在新摆位下重验一遍。包角那张表（45° 是临界、42° 力比 4.24）也是
   旧摆位测的；主线力比那一列来自 `diag_chuck_health`，与摆位无关，互顶那一列得重扫。
2. 把参数扫描扩成一份能引用的报告。直径这维有 A/B 对照了，爪型和摩擦两维还没扫。
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

1. 卡盘内孔是 24 片 box 拼出来的近似圆，不是真圆环。挡住穿透够用，但孔壁有棱，
   别拿它做孔壁摩擦那种接触级研究。
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
