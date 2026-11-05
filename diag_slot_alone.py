"""diag_slot_alone.py -- fixture-only placement test, robot excluded.

Route B dropped self-centering. This now checks what actually matters:
when the ARM places the shaft within its real IK accuracy (0 .. 2.4 mm of
y offset) from 5 mm above the slot, does the platform catch it and hold it
seated? Large offsets and 90 mm free falls are gone -- those measured a
behaviour we deliberately no longer provide.

Pass: >= 10 of 13 drops end with |dy| <= 2.5 mm and |dz| <= 1.5 mm.

Run:
    D:\\robot-env\\Scripts\\python.exe diag_slot_alone.py
"""
import numpy as np
import mujoco

import scene_full as S


def main():
    model, data = S.make_scene()
    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]

    for n in ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll"]:
        a = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        data.qpos[a] = 0.0

    # REALLY exclude the robot. Parking the joints at zero was not enough:
    # at the home pose the gripper pads hover right above the fixture, and
    # robot(1,3) vs work(2,3) gives (2&3)|(1&3) = 2|1 = 2, so the pads do
    # collide with the workpiece. Measured: offsets 15-40 mm stalled at
    # z = +35 mm, i.e. the workpiece rested on a pad and never fell.
    # A fixture-only test must physically remove the robot from the
    # contact problem, so zero its collision bits.
    robot_geoms = []
    for gi in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if nm.startswith("vise_sr"):
            continue
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ""
        if ("so" in body.lower() or "gripper" in body.lower()
                or "pad" in nm.lower() or "sts" in body.lower()):
            robot_geoms.append(gi)
    for gi in robot_geoms:
        model.geom_contype[gi] = 0
        model.geom_conaffinity[gi] = 0

    mujoco.mj_forward(model, data)
    print("  disabled collision on %d robot geoms" % len(robot_geoms))

    jaw = S.jaw_center()
    geo = S.jaw_geometry()

    print("=" * 74)
    print("FIXTURE-ONLY PLACEMENT TEST (route B: no self-centering)")
    print("=" * 74)
    print()
    print("  slot center    : %s" % np.round(jaw, 4))
    print("  slot           : %.0f mm wide x %.0f mm deep"
          % (geo["slot_w"] * 1000, geo["slot_d"] * 1000))
    print("  slot floor z   : %.1f mm" % (geo["slot_bottom"] * 1000))
    print("  rest axis z    : %.1f mm" % (geo["rest_z"] * 1000))
    print("  workpiece      : D%.0f x %.0f mm, %.3f kg"
          % (S.SHAFT_D * 1000, S.SHAFT_L * 1000, S.SHAFT_MASS))
    print("  drop height    : 5 mm above rest axis (gentle robot placement)")
    print("  y offsets      : 0 .. 2.4 mm, 13 drops")
    print("  pass rule      : |dy| <= 2.5 mm AND |dz| <= 1.5 mm")
    print()

    results = []
    for k in range(13):
        dy_mm = k * 0.2
        start = np.array([jaw[0], jaw[1] + dy_mm / 1000.0, jaw[2] + 0.005])
        data.qpos[sq:sq + 3] = start
        data.qpos[sq + 3:sq + 7] = [0.7071068, 0.0, 0.7071068, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        for _ in range(1500):
            mujoco.mj_step(model, data)

        final = data.qpos[sq:sq + 3].copy()
        dy_final = (final[1] - jaw[1]) * 1000
        dz_final = (final[2] - jaw[2]) * 1000
        # dy alone is not enough: a shaft perched on the wall lip looks
        # centred but sits ~3 mm too high. Require the z seat as well.
        seated = abs(dy_final) <= 2.5 and abs(dz_final) <= 1.5
        results.append((dy_mm, dy_final, dz_final, seated))

        print("  drop offset %+4.1f mm -> y %+6.1f mm  z %+6.1f mm  %s"
              % (dy_mm, dy_final, dz_final,
                 "SEATED" if seated else "OFF"))

    n_ok = sum(1 for r in results if r[3])
    print()
    print("  seated: %d / %d" % (n_ok, len(results)))
    if n_ok >= 10:
        print("  fixture works. Safe to run the robot.")
    else:
        print("  fixture still fails. Fix it before running the robot.")
    print()
    print("OK")


if __name__ == "__main__":
    main()
