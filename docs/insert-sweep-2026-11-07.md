# 插入深度四档复验

日期：2026-11-07  
任务卡：P0.3 复验（已完成）  
判据：`feed-criteria-v1`  
工件：Ø20 × 80 mm 圆柱棒料  
夹具：90° V 槽，槽深 8 mm，槽长 24 mm，夹持站位 −5 mm  
路径：完整机械臂送料，`--v-jaw --lift --no-lock`

## 复现命令

```powershell
$env:PYTHONIOENCODING = "utf-8"
& D:\robot-env\Scripts\python.exe src\diag_feed_insert_sweep.py `
  --inserts 7.75 8 10 25 `
  --output _build\handoff_insert_sweep.csv
```

本次产物：

- `_build/handoff_insert_sweep.csv`
- `_build/handoff_insert_sweep_cases/d7.75-706e4436.json`
- `_build/handoff_insert_sweep_cases/d8-dbd807ed.json`
- `_build/handoff_insert_sweep_cases/d10-30957fcd.json`
- `_build/handoff_insert_sweep_cases/d25-c8c07718.json`

## 结果

| 插入深度 | 总体 | 硬失败 | 质量判定 | 同轴 mm | 倾角 ° | 交接峰谷 mm | 余量偏心 mm | 20 N 拉拔 mm |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 7.75 | FAIL | PASS | FAIL：`stock_eccentricity` | 1.28 | 1.30 | 1.55 | 2.19 | 0.02 |
| 8 | FAIL | PASS | FAIL：`stock_eccentricity` | 1.30 | 1.40 | 1.54 | 2.28 | 0.02 |
| 10 | PASS | PASS | PASS | 0.66 | 0.47 | 1.53 | 0.92 | 0.04 |
| 25 | PASS | PASS | PASS | 0.64 | 0.23 | 1.58 | 0.70 | 0.00 |

本次固定初态四档中，最浅通过档位是 **10 mm**。这不是生产标准，也不是任意尺寸的通用夹持深度。

## 阶段定位

7.75 mm 和 8 mm 的 `feed_complete`、`three_jaw_clamped`、`gripper_released_in_place` 仍低于余量偏心上限；首次超过 2.0 mm 出现在**轴向退出**之后：

- 7.75 mm：`after_axial_retreat` 余量偏心约 2.28 mm。
- 8 mm：`after_axial_retreat` 余量偏心约 2.35 mm。
- 10 mm：`after_axial_retreat` 余量偏心约 0.81 mm。
- 25 mm：`after_axial_retreat` 余量偏心约 0.72 mm。

这说明当前差异首先出现在退出段，而不是进给完成或三爪首次夹紧阶段。它还不能单独证明具体根因；后续应做 `FEED_RETREAT_AXIAL=0/1` 和松爪后不移动的诊断对照，并报告接触对与姿态变化。

## 与旧记录的关系

旧文档曾记录“8 mm 通过”。那是另一轮代码和结果，不能覆盖本次恢复 P0 后的复验。当前应以本报告和对应 JSON 为准：**8 mm 在当前 `feed-criteria-v1` 下 FAIL，10 mm 是本次四档的最浅 PASS。**

本次没有修改阈值、没有调整夹具、没有加入来料扰动，也没有训练模型。
