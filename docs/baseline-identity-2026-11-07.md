# P0.4 基准身份记录

日期：2026-11-07  
任务卡：P0.4  
仓库：`D:\codex\projects\robot-sim-first-step`

## 代码身份

- `HEAD`：`fa14f31c656a774e3627907a730e4389725063db`
- 工作树：**脏**，本轮没有 commit 或 push
- 关键文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `src/diag_feed_chuck.py` | `183960B3017F7BEFBA74E695F6566D0FE7A44DA4C42B70A3B79E7C6685DD1816` |
| `src/diag_feed_insert_sweep.py` | `BF4EBAD5C04C219AC51A043BDF74D9E42E1F1D0FDB593F5F420E9A37A40F2461` |
| `docs/metric-dictionary.md` | `77347B50DAF8844578A7EACD54AE91B4E09B4D3901EE56C54A4117A4048E786E` |
| `docs/insert-sweep-2026-11-07.md` | `76744D81690EA38A9E3FF806E080388AB1DEA08AE73E68A0FD302C819E0ABFD2` |

## 运行环境

- OS：Windows 11
- Python：3.12.15
- MuJoCo：3.17.0
- 解释器：`.venv\Scripts\python.exe`
- 终端编码：`PYTHONIOENCODING=utf-8`

## 基准复现命令

```powershell
$env:PYTHONIOENCODING = "utf-8"
& .venv\Scripts\python.exe verify_all.py
& .venv\Scripts\python.exe src\diag_feed_chuck.py --v-jaw --lift --no-lock
```

本次原始输出：

- `_build/baseline_verify_2026-11-07.txt`
- `_build/baseline_feed_2026-11-07.txt`

结果：`verify_all.py` **9/9**；全物理 V 槽送料 `FEED_METRIC PASS`，同轴约 `0.64 mm`、倾角约 `0.2°`、交接位移约 `1.58 mm`、余量偏心约 `0.70/2.0 mm`、20 N 拉拔约 `0.00 mm`。

## 8 mm 视觉核验

复现配置：`FEED_INSERT_MM=8`，`--v-jaw --lift --no-lock`。证据：

- `_build/now_stages.png`
- `_build/now_views.png`
- `_build/now_feed.gif`
- `_build/handoff_d8_visual.json`
- `_build/frame_check/`（抽帧）

视觉结论：未看到棒料被挤飞、三爪在抓取失败后继续闭合、或机械臂明显穿过工件；棒料最终留在卡盘中，机械臂正常退出。  
数值结论：该 case 仍为 `overall_pass=false`，`quality_failures=["stock_eccentricity"]`，余量偏心约 `2.276 mm`，超过当前 `2.0 mm` 项目阈值；因此不能写成 8 mm 通过。

## 边界

这份记录只冻结“本地脏工作树 + 当前环境 + 当前证据”的身份，不把 `HEAD` 误称为已提交的完整基线。后续若修改代码，应重新生成哈希和输出，不覆盖本目录中的原始日志。
