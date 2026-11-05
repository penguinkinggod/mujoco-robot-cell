# 第三方代码与资产

本仓库的代码为 MIT（见 `LICENSE`），但以下内容来自第三方，遵循各自的许可证。
**如果你要再分发本仓库，请同时保留本文件和对应的原始许可证。**

---

## 1. Franka Panda 模型

| | |
|---|---|
| 路径 | `models/panda/` |
| 来源 | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Google DeepMind） |
| 许可证 | **Apache-2.0** |
| 原始许可文件 | `models/panda/LICENSE` |

**本仓库做过的修改**：`link2.stl`（碰撞网格）用 `link1.stl` 顶替，因为下载镜像对 >50 MB 的包做了限制。视觉网格全部为原版，物理碰撞体存在这一处近似。详见 `HANDOFF.md`。

---

## 2. LeRobot SO-101 环境包

| | |
|---|---|
| 路径 | `models/_tmp_so101/lerobot_env_so101-0.4.0/` |
| 来源 | [LeRobot](https://github.com/huggingface/lerobot)（Hugging Face），版本 0.4.0 |
| 许可证 | **Apache-2.0** |
| 原始许可文件 | `models/_tmp_so101/lerobot_env_so101-0.4.0/LICENSE` |

包含 SO-101 机械臂的 MJCF 模型、Gym 环境封装（`pick_env.py` / `pick_place_env.py` / `wrappers.py`）和行为克隆策略（`policy/bc_model.py`）。

**本仓库未修改该包的任何文件。** 目录名中的 `_tmp_` 是历史遗留，它不含任何临时内容——`scene_full.py` 直接引用其中的 MJCF 与 STL，不能删除。

> 注意：这个包是第三方 vendored 代码，不是本仓库的原创部分。若需更新版本，请从上游重新获取，不要就地修改。

---

## 3. SimReady 工业资产

| | |
|---|---|
| 路径 | `models/simready/` |
| 资产 | `drivetrain_axle_shaft_a01`、`drivetrain_axle_shaft_d01`、`equipment_bench_vise_a01`、`equipment_bench_vise_heavy_b01`、`equipment_machine_vise_c01`、`tool_clamp_c01` 等 |
| 许可证 | **CC-BY-4.0**（依据仓库内 `models/simready/manifest.jsonl` 中每个资产的 `license` 字段） |
| 原始作者 | ⚠️ **待补充** —— 资产清单中只记录了 `license` 字段，未记录作者与下载地址 |

CC-BY-4.0 允许再分发和修改，**但要求署名原始作者并标明是否修改**。

当前清单缺少作者信息，属于待补项。如果你知道这批资产的原始下载来源，请补进本节；在此之前，本仓库对这些资产的使用仅限于本地仿真，不建议以此仓库作为资产的上游分发源。

**本仓库做过的修改**：OBJ → MJCF 网格转换（`models/simready/mjcf_meshes/`），以及物理属性从 `.phys.json` 读取后写入 MJCF。

---

## 4. 本仓库原创部分

除上述之外的所有代码（`demo_*.py`、`scene_*.py`、`diag_*.py`、`verify_all.py`、`live_bridge.py`、`looks.py`、`reachability.py` 等）与文档为原创，MIT 许可。

---

## 汇总

| 内容 | 许可证 | 可再分发 |
|---|---|---|
| 本仓库代码与文档 | MIT | ✅ |
| `models/panda/` | Apache-2.0 | ✅ 需保留 LICENSE |
| `models/_tmp_so101/` | Apache-2.0 | ✅ 需保留 LICENSE |
| `models/simready/` | CC-BY-4.0 | ✅ 需署名，**作者待补** |
