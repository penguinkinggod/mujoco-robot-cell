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
    "src/scene_full.py",
    "src/scene_machining.py",
    "src/diag_slot_alone.py",
    "src/diag_fixture_profile.py",
    "src/diag_chuck_grip.py",
    "src/demo_08_pick_and_load.py",
    "src/demo_09_try_versions.py",
    "verify_all.py",
    "src/diag_chuck_health.py",
    "src/gif_slim.py",
    "src/release_media.py",
    "src/diag_diameter_sweep.py",
    "src/diag_jaw_face_sweep.py",
    "src/diag_jaw_wrap.py",
    # Also a real CHECKS entry below (kept here for py_compile coverage).
    "src/diag_feed_chuck.py",
    "src/diag_feed_insert_sweep.py",
    "src/diag_contact_sensitivity.py",
    "tests/test_feed_failure_detection.py",
    "tests/test_feed_insert_sweep.py",
]

MACHINING_CODE = (
    "import sys; sys.path.insert(0, 'src'); import scene_machining as S; "
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
    spread, ecc, tilt = float(m.group(1)), float(m.group(2)), float(m.group(3))
    contacts, reach = int(m.group(4)), float(m.group(5))
    good = (rc == 0 and "CHUCK_HEALTH PASS" in out
            and ecc <= 1.0 and tilt <= 2.0
            and contacts >= 6 and reach <= 10.0
            and spread <= 0.2)   # true synchronisation must not regress
    return good, "位移差%.2fmm 偏心%.2fmm 接触%d 可达%.1fmm" % (
        spread, ecc, contacts, reach)


def check_feed(out, err, rc):
    """Robot feeding. NOTE: the carry uses a kinematic lock (an explicit,
    documented simplification). We assert physics_after_clamp=1 so nobody
    can quietly extend the lock through the pull-out test and fake a pass,
    and match 'carry=kinematic_lock' literally so silently deleting the
    lock also fails the parse instead of being reported as full physics.
    """
    m = re.search(
        r"FEED_METRIC coax_mm=([\d.]+) tilt_deg=([\d.]+) "
        r"contacts=(\d+) slip_mm=([\d.]+) pull_mm=([\d.]+) "
        r"jaw_force_N=([\d.]+) grasp_tilt_deg=([\d.]+) "
        r"carry=kinematic_lock physics_after_clamp=(\d+)", out)
    if not m:
        return False, "no FEED_METRIC line"
    coax, tilt = float(m.group(1)), float(m.group(2))
    contacts = int(m.group(3))
    slip, pull = float(m.group(4)), float(m.group(5))
    force, unlocked = float(m.group(6)), int(m.group(8))
    good = (rc == 0 and "FEED_METRIC PASS" in out and unlocked == 1
            and coax <= 2.0 and tilt <= 10.0 and contacts >= 6
            and slip <= 2.0 and pull <= 2.0)
    return good, "同轴%.2fmm 倾斜%.1f° 接触%d 拉%.2fmm 力%.1fN" % (
        coax, tilt, contacts, pull, force)


def check_feed_physics(out, err, rc):
    """V jaws, and NOTHING pinned: form closure + friction carry the part.

    This check exists because the locked run above can pass while the gripper
    holds nothing at all -- which is exactly what happened for several
    sessions: 4 contacts and 105 N, yet the rod was wedged and 20 deg off
    axis. Here the V grooves must locate the rod by geometry and the contact
    must carry it across the cell on its own.
    """
    m = re.search(
        r"FEED_METRIC coax_mm=([\d.]+) tilt_deg=([\d.]+) "
        r"contacts=(\d+) slip_mm=([\d.]+) pull_mm=([\d.]+) "
        r"jaw_force_N=([\d.]+) grasp_tilt_deg=([\d.]+) "
        r"carry=physics physics_after_clamp=(\d+)", out)
    if not m:
        return False, "no FEED_METRIC line with carry=physics"
    coax, tilt = float(m.group(1)), float(m.group(2))
    contacts = int(m.group(3))
    slip, pull = float(m.group(4)), float(m.group(5))
    force, unlocked = float(m.group(6)), int(m.group(8))
    # 交接全程可见位移（夹紧+松爪+退出）。用户确认真机三爪就是这个分布，
    # 这段位移会被第一刀切掉、不影响成品，所以这里只作「别把料甩飞」的
    # 宽松护栏，不再是工艺质量目标。
    sw = re.search(r"FEED_METRIC_HANDOVER clamp_swing_mm=([\d.]+) "
                   r"guard=([\d.]+)", out)
    swing = float(sw.group(1)) if sw else None
    swing_guard = float(sw.group(2)) if sw else 12.0
    # 真正对应加工的判据：整根料最坏处偏心 ≤ 单边余量，第一刀才能见光。
    st = re.search(r"FEED_METRIC_MACHINING stock_ecc_mm=([\d.]+) "
                   r"allow_mm=([\d.]+)", out)
    stock = float(st.group(1)) if st else None
    stock_allow = float(st.group(2)) if st else 2.0
    good = (rc == 0 and "FEED_METRIC PASS" in out and unlocked == 1
            and coax <= 2.0 and tilt <= 10.0 and contacts >= 6
            and slip <= 2.0 and pull <= 2.0
            and swing is not None and swing <= swing_guard
            and stock is not None and stock <= stock_allow)
    return good, ("无锁同轴%.2fmm 倾斜%.1f° 接触%d 拉%.2fmm 力%.1fN "
                  "交接位移%s 余量偏心%s" % (
                      coax, tilt, contacts, pull, force,
                      "%.2fmm" % swing if swing is not None else "缺失",
                      "%.2f/%.1fmm" % (stock, stock_allow)
                      if stock is not None else "缺失"))


def check_env_cell(out, err, rc):
    """Same physics, but in the official lerobot/so101-tabletop-mujoco world.

    The scene migration is only real if the numbers survive it. This runs the
    identical V-jaw flow with official_world=True and requires the same
    coaxiality band and `carry=physics` -- plus all four cameras present, so a
    silently dropped camera (which would break the dataset/video path) fails
    the check instead of passing quietly.
    """
    m = re.search(
        r"FEED_METRIC coax_mm=([\d.]+) tilt_deg=([\d.]+) contacts=(\d+) "
        r"slip_mm=([\d.]+) pull_mm=([\d.]+) .*carry=physics", out)
    if not m:
        return False, "官方场景里没有 carry=physics 的 FEED_METRIC"
    ncam = re.search(r"CAMS (\d+)", out)
    if not ncam or int(ncam.group(1)) < 4:
        return False, "相机不全（应为 front/top/cell/wrist_cam 四个）"
    coax, tilt = float(m.group(1)), float(m.group(2))
    contacts = int(m.group(3))
    good = (rc == 0 and "FEED_METRIC PASS" in out
            and coax <= 2.0 and tilt <= 10.0 and contacts >= 6)
    return good, "官方场景 同轴%.2fmm 倾斜%.1f° 接触%d 相机%s个" % (
        coax, tilt, contacts, ncam.group(1))


def check_feed_failure_detection(out, err, rc):
    combined = out + "\n" + err
    m = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    passed = m is not None and "OK" in combined
    return rc == 0 and passed, (
        "%s failure-detection tests" % m.group(1) if m
        else "unittest output missing")


CHECKS = [
    dict(name="语法检查", cmd=[PY, "-m", "py_compile"] + CORE_FILES,
         check=check_compile, slow=False, hint="秒级"),
    dict(name="车床场景编译", cmd=[PY, "-c", MACHINING_CODE],
         check=check_machining, slow=False, hint="秒级"),
    dict(name="夹具落料 13/13", cmd=[PY, "src/diag_slot_alone.py"],
         check=check_slot, slow=True, hint="约 3 秒"),
    dict(name="卡盘夹紧", cmd=[PY, "src/diag_chuck_grip.py"],
         check=check_chuck, slow=True, hint="约 2 秒"),
    dict(name="卡盘工位体检", cmd=[PY, "src/diag_chuck_health.py"],
         check=check_chuck_health, slow=True, hint="约 5 秒"),
    dict(name="完整序列 3/3", cmd=[PY, "src/demo_09_try_versions.py"],
         check=check_demo09, slow=True, hint="约 9 秒"),
    dict(name="机械臂送料装夹(历史基线·穿模)",
         cmd=[PY, "src/diag_feed_chuck.py", "--passthrough"],
         check=check_feed, slow=True, hint="约 4 秒"),
    dict(name="送料·全物理(V槽)",
         cmd=[PY, "src/diag_feed_chuck.py", "--v-jaw", "--lift", "--no-lock"],
         check=check_feed_physics, slow=True, hint="约 20 秒"),
    dict(name="送料·官方场景(SO-101 env)",
         cmd=[PY, "src/env_cell.py", "--run", "--no-video"],
         check=check_env_cell, slow=True, hint="约 5 秒"),
    dict(name="送料失败检测单测",
         cmd=[PY, "-m", "unittest", "discover", "-s", "tests",
              "-p", "test_*.py"],
         check=check_feed_failure_detection, slow=False,
         hint="约 5 秒"),
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
