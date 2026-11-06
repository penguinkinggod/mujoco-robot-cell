"""gif_slim.py -- 把逐帧渲染出来的 GIF 压到能放 README 的体积。

为什么需要
----------
show_current.py 是「每仿真帧渲一格」，25 fps 下 14.5 s 的流程就是 425 帧。
直接 mimsave 出来 20 MB，GitHub 首屏加载会卡。

但里面大部分帧是同一个像素：进给段有 9.2 s 末端几乎不动，闭环在等误差收敛。
这些帧留着只是让观看者干等。所以先按「和上一帧的差」去重，合并掉的时长
累加到前一个保留帧上，总时长不变、动作一帧不少。

然后统一调色板。PIL 逐帧各自 quantize 会每帧带一份 768 字节调色板，
而且帧间差分失效；先用代表帧算出一张全局 128 色调色板，所有帧映射到它，
optimize=True 才能把「这帧和上帧相同的块」真正省掉。

用法
----
    python src/gif_slim.py in.gif out.gif               # 默认 128 色
    python src/gif_slim.py in.gif out.gif --colors 96 --diff 8
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageSequence


def load_frames(path):
    """读帧，连同每帧停留时长一起读。

    只取像素、丢掉 duration 的话，拿已经压过的 GIF 再压一遍会把所有停顿
    抹平成 1/fps，整段被放快：14.3 s 的流程会变成 9.3 s。
    """
    im = Image.open(path)
    frames, holds = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(np.asarray(f.convert("RGB")))
        holds.append(float(im.info.get("duration") or 0) / 1000.0)
    if not any(holds):
        holds = None
    return frames, holds


def dedupe(frames, fps, diff_thresh, probe=(80, 60), max_hold=None, src_holds=None):
    """丢掉和上一帧几乎相同的帧，把时长还给上一帧。

    probe     比较用的缩略图尺寸，不是输出尺寸；只为判断「动了没有」。
    max_hold  单帧最长停留秒数。静止段留下来就是一个 9 秒的定格，
              观感上是卡住；超过这个值就截断（画面不变，只是不再等）。
    src_holds 每帧原本的停留时长；None 表示按 1/fps 均匀。重新压一个
              已经带时序的 GIF 时必须传，否则停顿会被抹平。
    """
    kept, holds = [], []
    prev = None
    merged = 0

    def step(i):
        return src_holds[i] if src_holds else 1.0 / fps

    for i, f in enumerate(frames):
        small = np.asarray(
            Image.fromarray(f).resize(probe, Image.Resampling.BILINEAR)
        ).astype(np.int16)
        if prev is not None and np.abs(small - prev).mean() < diff_thresh:
            merged += 1
            hold = holds[-1] + step(i)
            holds[-1] = min(hold, max_hold) if max_hold else hold
            prev = small
            continue
        kept.append(f)
        holds.append(step(i))
        prev = small
    return kept, holds, merged


def dip_fade(frames, seconds, fps):
    """末尾淡出到黑、开头从黑淡入，循环接缝落在纯黑上。

    为什么不直接叠化末帧和首帧：这两帧姿态差得太远（末帧是机械臂退回、
    三爪夹住料；首帧是机械臂在取料位），叠化出来是两套手臂同时出现在
    画面里，看起来像渲染故障。淡到黑只有一个画面在变暗变亮，不会重影。

    代价是每次循环有一次约 0.5 s 的黑场，观感上像转场；好处是黑帧几乎
    不占体积（LZW 之后几百字节），且接缝严格连续。
    """
    n = max(2, int(round(seconds * fps)))
    dt = 1.0 / fps
    a = frames[0].astype(np.float32)
    b = frames[-1].astype(np.float32)
    fade_in = [np.clip(a * (k / n), 0, 255).astype(np.uint8)
               for k in range(1, n)]
    fade_out = [np.clip(b * (1.0 - k / n), 0, 255).astype(np.uint8)
                for k in range(1, n + 1)]
    return fade_in, fade_out, dt


def global_palette(frames, colors, samples=12):
    """从若干等距代表帧里算一张调色板，所有帧共用。"""
    picks = [frames[int(i * (len(frames) - 1) / max(1, samples - 1))]
             for i in range(samples)]
    h, w = frames[0].shape[:2]
    sheet = Image.new("RGB", (w, h * len(picks)))
    for i, f in enumerate(picks):
        sheet.paste(Image.fromarray(f), (0, i * h))
    return sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def build(frames, holds, path, colors):
    pal_src = global_palette(frames, colors)
    out = [Image.fromarray(f).quantize(palette=pal_src,
                                      dither=Image.Dither.NONE)
           for f in frames]
    out[0].save(path, save_all=True, append_images=out[1:],
                duration=[max(20, int(round(h * 1000))) for h in holds],
                loop=0, optimize=True, disposal=1)


def slim_frames(frames, path, fps=25.0, colors=128, diff=0.12, max_hold=1.2,
                fade=0.0, fade_fps=12.0, end_hold=0.6, src_holds=None):
    """给 show_current.py 直接调：输入 ndarray 列表，输出瘦身后的 GIF。

    fade      末尾淡出 / 开头淡入的时长（秒），0 表示不做，保持硬切。
    end_hold  末帧定格上限（秒）。循环播放时末帧停太久会像卡住。
    src_holds 每帧原本的时长；从已有 GIF 二次处理时必须传，否则停顿会被抹平。
    """
    kept, holds, merged = dedupe(frames, fps, diff, max_hold=max_hold or None,
                                 src_holds=src_holds)
    n_fade = 0
    if fade > 0 and len(kept) > 2:
        holds[-1] = min(holds[-1], end_hold)
        fi, fo, dt = dip_fade(kept, fade, fade_fps)
        kept = fi + kept + fo
        holds = [dt] * len(fi) + holds + [dt] * len(fo)
        n_fade = len(fi) + len(fo)
    build(kept, holds, path, colors)
    return len(frames), len(kept), merged, n_fade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--diff", type=float, default=6.0,
                    help="缩略图平均通道差低于此值视为没动")
    ap.add_argument("--max-hold", type=float, default=1.2,
                    help="单帧最长停留秒数，0 表示不限制")
    ap.add_argument("--fade", type=float, default=0.0,
                    help="末尾淡出 / 开头淡入的秒数，让无限循环不留硬切（0 = 不做）")
    ap.add_argument("--fade-fps", type=float, default=12.0,
                    help="淡入淡出段的帧率，低一点更省体积")
    ap.add_argument("--end-hold", type=float, default=0.6,
                    help="末帧定格上限，只在做淡出时生效")
    a = ap.parse_args()

    frames, src_holds = load_frames(a.src)
    n_all, n_kept, merged, n_fade = slim_frames(
        frames, a.dst, fps=a.fps, colors=a.colors, diff=a.diff,
        max_hold=a.max_hold or None, fade=a.fade,
        fade_fps=a.fade_fps, end_hold=a.end_hold, src_holds=src_holds)

    src_mb = os.path.getsize(a.src) / 1048576
    dst_mb = os.path.getsize(a.dst) / 1048576
    print("%d 帧 -> %d 帧（合并 %d，淡入淡出 %d）  %d 色  %.2f -> %.2f MB"
          % (len(frames), n_kept, merged, n_fade, a.colors, src_mb, dst_mb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
