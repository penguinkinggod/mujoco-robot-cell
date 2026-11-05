"""verify_all.py -- one command to prove the project has not regressed.

Why this exists
---------------
A script can print "OK" and still be wrong. In this project we already hit
several cases where the docs said "works" while the code raised AttributeError,
or where a physics script exited 0 with a lift of 0.0 mm. So every check below
parses a **number out of the real output**. Exit code alone is never trusted.

Run:
    D:\\robot-env\\Scripts\\python.exe verify_all.py
    D:\\robot-env\\Scripts\\python.exe verify_all.py --quick
    D:\\robot-env\\Scripts\\python.exe verify_all.py --verbose

--quick    only syntax/compile checks (seconds). Use while editing.
--verbose  print the full output of every child script.

Exit code: 0 = all passed, 1 = at least one failed.
"""
import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ)
ENV["MUJOCO_GL"] = "glfw"
ENV["PYTHONUTF8"] = "1"

CORE_FILES = [
    "scene_full.py",
    "scene_machining.py",
    "diag_slot_alone.py",
    "diag_fixture_profile.py",
    "diag_chuck_grip.py",
    "demo_08_pick_and_load.py",
    "demo_09_try_versions.py",
    "verify_all.py",
    "diag_chuck_health.py",
]

MACHINING_CODE = (
    "import scene_machining as S; "
    "m,d,ids=S.make_scene(); "
    "print('COMPILE_OK ngeom=%d nbody=%d nq=%d' % (m.ngeom,m.nbody,m.nq))"
)


def check_compile(out, err, rc):
    if rc != 0 or err.strip():
        return False, "py_compile failed"
    return True, "%d files" % len(CORE_FILES)


def check_slot(out, err, rc):
    m = re.search(r"seated:\s*(\d+)\s*/\s*(\d+)", out)
    if not m:
        return False, "no 'seated: N / M' line"
    ok, total = int(m.group(1)), int(m.group(2))
    good = rc == 0 and ok == total and total >= 13 and "fixture works" in out
    return good, "%d/%d SEATED" % (ok, total)


def check_demo09(out, err, rc):
    m = re.search(r"得分\s*(\d+)\s*/\s*3", out)
    if not m:
        return False, "no '得分 N / 3' line"
    score = int(m.group(1))
    flags = ("✓ 抓取" in out, "✓ 搬运" in out, "✓ 放置" in out)
    good = rc == 0 and score == 3 and all(flags)
    return good, "%d/3 (抓%s 搬%s 放%s)" % (
        score, *["✓" if f else "✗" for f in flags])


def check_machining(out, err, rc):
    m = re.search(r"COMPILE_OK\s+ngeom=(\d+)\s+nbody=(\d+)\s+nq=(\d+)", out)
    if not m:
        return False, "no COMPILE_OK line"
    return rc == 0, "ngeom=%s nbody=%s nq=%s" % m.groups()


def check_chuck(out, err, rc):
    held = "夹住了，没有下滑" in out
    line20 = next((l for l in out.splitlines()
                   if l.strip().startswith("20 N")), "")
    pull_ok = "没动" in line20
    ended = out.strip().endswith("OK")
    good = rc == 0 and held and pull_ok and ended
    return good, "夹紧=%s 20N=%s" % (
        "OK" if held else "FAIL", "OK" if pull_ok else "FAIL")


def check_chuck_health(out, err, rc):
    m = re.search(
        r"CHUCK_HEALTH jaw_spread_mm=([\d.]+) ecc_mm=([\d.]+) "
        r"tilt_deg=([\d.]+) contacts=(\d+) reach_err_mm=([\d.]+)", out)
    if not m:
        return False, "no CHUCK_HEALTH line"
    ecc, tilt = float(m.group(2)), float(m.group(3))
    contacts, reach = int(m.group(4)), float(m.group(5))
    good = (rc == 0 and "CHUCK_HEALTH PASS" in out
            and ecc <= 1.0 and tilt <= 2.0
            and contacts >= 6 and reach <= 10.0)
    return good, "偏心%.2fmm 接触%d 可达%.1fmm" % (ecc, contacts, reach)


CHECKS = [
    dict(name="语法检查", cmd=[PY, "-m", "py_compile"] + CORE_FILES,
         check=check_compile, slow=False, hint="秒级"),
    dict(name="车床场景编译", cmd=[PY, "-c", MACHINING_CODE],
         check=check_machining, slow=False, hint="秒级"),
    dict(name="夹具落料 13/13", cmd=[PY, "diag_slot_alone.py"],
         check=check_slot, slow=True, hint="约 3 秒"),
    dict(name="卡盘夹紧", cmd=[PY, "diag_chuck_grip.py"],
         check=check_chuck, slow=True, hint="约 2 秒"),
    dict(name="卡盘工位体检", cmd=[PY, "diag_chuck_health.py"],
         check=check_chuck_health, slow=True, hint="约 5 秒"),
    dict(name="完整序列 3/3", cmd=[PY, "demo_09_try_versions.py"],
         check=check_demo09, slow=True, hint="约 9 秒"),
]


def run_one(item, verbose):
    t0 = time.monotonic()
    try:
        p = subprocess.run(item["cmd"], cwd=HERE, env=ENV,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=600)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = (e.stderr or "") + "\nTIMEOUT after 600 s"
        rc = -1
    dt = time.monotonic() - t0

    if verbose:
        print("-" * 60)
        print(out)
        if err.strip():
            print("[stderr]", err)
        print("-" * 60)

    ok, detail = item["check"](out, err, rc)
    print("  [%s] %-18s %6.1fs  %s" %
          ("PASS" if ok else "FAIL", item["name"], dt, detail))
    if not ok and not verbose:
        tail = (out + "\n" + err).strip().splitlines()[-25:]
        print("        " + "-" * 50)
        for line in tail:
            print("        " + line)
    return ok, dt


def main():
    ap = argparse.ArgumentParser(description="项目回归检查")
    ap.add_argument("--quick", action="store_true",
                    help="只跑语法/编译检查，跳过物理仿真")
    ap.add_argument("--verbose", action="store_true",
                    help="打印每个子脚本的完整输出")
    args = ap.parse_args()

    print("=" * 68)
    print("verify_all — 项目回归检查" + ("（quick）" if args.quick else ""))
    print("=" * 68)
    results = []
    t0 = time.monotonic()
    for item in CHECKS:
        if args.quick and item["slow"]:
            print("  [SKIP] %-18s          %s" % (item["name"], "--quick"))
            continue
        print("  [  ..] %-18s 运行中（%s）" % (item["name"], item["hint"]),
              flush=True)
        results.append(run_one(item, args.verbose))

    total = time.monotonic() - t0
    n_ok = sum(1 for ok, _ in results if ok)
    all_ok = n_ok == len(results) and len(results) > 0
    print("-" * 68)
    print("  %s  (%d/%d 通过, 用时 %.1fs)" %
          ("PASS — 项目没有退化" if all_ok else "FAIL — 有检查未通过",
           n_ok, len(results), total))
    print("=" * 68)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
