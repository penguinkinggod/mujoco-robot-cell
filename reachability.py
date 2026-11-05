"""可达性检查：在执行之前，先判断"这个目标能不能做到"。

为什么需要这个模块
------------------
IK 能解出一个关节角，不代表机器人能走到那个位置。实测过四种失败：

    1. 碰撞     —— 夹爪顶到桌面，被顶住停住（实测差 38 mm）
    2. 力矩饱和 —— j6 想要 325 Nm，上限只有 12 Nm，差 27 倍
    3. 关节限位 —— 目标角度超出 jnt_range
    4. 伺服稳态误差 —— 位置伺服扛重力必然有偏差，err ≈ 载荷 / kp

前三种是"物理上做不到"，第四种是"做到了但差一点"。
这个模块把它们分开报，让你知道该改任务还是该调参数。

用法
----
    from reachability import check_grasp_pose

    rep = check_grasp_pose(model, q_target, grasp_target_xyz)
    if not rep.ok:
        print(rep.summary())
    else:
        # 可以执行
        ...

    # 找最高的安全抓取高度
    z_safe = find_safe_grasp_height(model, xy=(0.55, 0.05))
"""
import numpy as np
import mujoco


# ---------------------------------------------------------------- 常量

# 夹爪几何容差：全开 80 mm，方块宽 43 mm -> 中心最多偏 18.5 mm
GRIPPER_FULL_OPEN = 0.080


class Report:
    """一次可达性检查的结果。"""

    def __init__(self):
        self.ok = False
        self.problems = []        # [(类别, 说明), ...]
        self.metrics = {}         # 各项实测数值

    def add(self, kind, msg):
        self.problems.append((kind, msg))

    def summary(self):
        if self.ok:
            return "可达 OK"
        lines = ["不可达："]
        for kind, msg in self.problems:
            lines.append("  [%s] %s" % (kind, msg))
        return "\n".join(lines)

    def table(self):
        """打印详细指标，调参时看这个。"""
        rows = []
        for k, v in self.metrics.items():
            if isinstance(v, float):
                rows.append((k, "%.6f" % v))
            else:
                rows.append((k, str(v)))
        w = max((len(r[0]) for r in rows), default=10)
        out = []
        for k, v in rows:
            out.append("  %-*s : %s" % (w, k, v))
        return "\n".join(out)


# ---------------------------------------------------------------- 工具

def _gripper_geoms(model, hand_id):
    """返回属于 hand 子树的所有 geom id。用几何体判断碰撞，比名字可靠。"""
    out = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if b == 0:
            continue
        p = b
        while p != 0:
            if p == hand_id:
                out.append(g)
                break
            p = model.body_parentid[p]
    return out


def _rot_error(R_cur, R_des):
    e = np.zeros(3)
    for i in range(3):
        e += np.cross(R_cur[:, i], R_des[:, i])
    return 0.5 * e


# ---------------------------------------------------------------- 主检查

def check_grasp_pose(model, q, grasp_target, hand_name="hand",
                     finger_offset=0.0584, table_z=0.0,
                     min_clearance=0.003, settle_steps=0,
                     expected_contacts=()):
    """检查一个抓取姿势能不能真的执行。

    参数
    ----
    model : MjModel
    q : array(7,)      7 个关节角（IK 解出来的）
    grasp_target : array(3,)  想让夹爪中心到达的位置
    hand_name : str    末端 body 名
    finger_offset : float  夹爪中心相对 hand 原点沿 z 轴的偏移
    table_z : float    桌面高度
    min_clearance : float  允许的最小离地间隙（米）
    settle_steps : int  可选：先跑几步仿真再检查（0 = 只用 mj_forward）
    expected_contacts : tuple of str
        允许接触的 geom 名（比如要抓的那个方块）。
        抓取时夹爪本来就要碰到目标物体，不该算"撞到异物"。

    返回
    ----
    Report
    """
    rep = Report()
    q = np.asarray(q, dtype=float)
    expected = set(expected_contacts)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
    gid = _gripper_geoms(model, hand_id)

    d = mujoco.MjData(model)
    d.qpos[:7] = q
    # 夹爪闭合到"夹住物体"的状态去检查，这才是真正执行时的姿态
    d.qpos[7:9] = [0.0, 0.0]
    mujoco.mj_forward(model, d)

    if settle_steps > 0:
        for _ in range(settle_steps):
            d.ctrl[:7] = q
            d.ctrl[7] = 0.0
            mujoco.mj_step(model, d)

    # ---- 1. 关节限位 ----
    jr = model.jnt_range[:7]
    margin = np.minimum(q - jr[:, 0], jr[:, 1] - q)
    rep.metrics["最小关节余量(rad)"] = float(margin.min())
    worst = int(np.argmin(margin))
    if margin.min() < 0:
        rep.add("关节超限",
                "j%d 目标 %.4f rad 超出范围 [%.4f, %.4f]"
                % (worst + 1, q[worst], jr[worst, 0], jr[worst, 1]))
    elif margin.min() < 0.02:
        rep.add("关节贴限位",
                "j%d 只剩 %.4f rad 余量，伺服误差会顶到限位" % (worst + 1, margin.min()))

    # ---- 2. 重力载荷 vs 力矩上限 ----
    loads = np.abs(d.qfrc_bias[:7])
    limits = model.actuator_forcerange[:7, 1]
    ratio = loads / np.maximum(limits, 1e-9)
    rep.metrics["最大力矩占用比"] = float(ratio.max())
    worst_t = int(np.argmax(ratio))
    if ratio.max() >= 0.99:
        rep.add("力矩饱和",
                "j%d 需要 %.1f Nm，上限 %.1f Nm，差 %.0f 倍"
                % (worst_t + 1, loads[worst_t], limits[worst_t],
                   loads[worst_t] / max(limits[worst_t], 1e-9)))
    elif ratio.max() > 0.7:
        rep.add("力矩接近饱和",
                "j%d 用掉 %.0f%% 力矩，动态时会饱和"
                % (worst_t + 1, ratio.max() * 100))

    # ---- 3. 碰撞：夹爪子树最低点 vs 桌面 ----
    if gid:
        zs = [d.geom_xpos[g][2] for g in gid]
        lowest = float(min(zs))
        rep.metrics["夹爪最低点z(m)"] = lowest
        clearance = lowest - table_z
        rep.metrics["离桌面间隙(m)"] = clearance
        if clearance < 0:
            rep.add("碰撞桌面",
                    "夹爪最低点在 z=%.4f，桌面在 %.4f，穿进去 %.1f mm"
                    % (lowest, table_z, -clearance * 1000))
        elif clearance < min_clearance:
            rep.add("离桌面太近",
                    "只剩 %.1f mm 间隙（要求 %.1f mm）"
                    % (clearance * 1000, min_clearance * 1000))

    # ---- 4. 实际接触检测（MuJoCo 自己算的）----
    # 注意：必须排除夹爪自身的左右指互相接触。
    # 夹爪闭合到 qpos=0 时两指尖贴在一起，会产生几十个接触点，
    # 那是"夹紧了"的正常现象，不是"撞到异物"。第一版没排除，导致误报。
    robot_gids = set(gid)
    bad_contacts = []
    self_contacts = 0
    for i in range(d.ncon):
        c = d.contact[i]
        g1_in = c.geom1 in robot_gids
        g2_in = c.geom2 in robot_gids
        if not (g1_in or g2_in):
            continue
        if g1_in and g2_in:
            self_contacts += 1          # 夹爪自己撞自己，忽略
            continue
        other = c.geom2 if g1_in else c.geom1
        other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or "geom%d" % other
        if other_name in expected:
            continue                    # 抓取目标本身，允许接触
        bad_contacts.append((other_name, float(c.dist)))
    rep.metrics["夹爪自接触数"] = self_contacts
    rep.metrics["夹爪与异物接触数"] = len(bad_contacts)
    if bad_contacts:
        names = sorted(set(n for n, _ in bad_contacts))
        rep.add("夹爪接触异物",
                "和 %s 有 %d 个接触点" % (",".join(names), len(bad_contacts)))

    # ---- 5. 伺服稳态误差预测 ----
    kp = model.actuator_gainprm[:7, 0]
    pred_err = loads / np.maximum(kp, 1e-9)
    rep.metrics["预测最大关节稳态误差(rad)"] = float(pred_err.max())
    if pred_err.max() > 0.05:
        rep.add("伺服误差偏大",
                "预测 %.4f rad（%.1f 度），末端会偏几厘米"
                % (pred_err.max(), np.degrees(pred_err.max())))

    # ---- 6. 实际到达位置 vs 目标 ----
    R = d.xmat[hand_id].reshape(3, 3)
    actual = d.xpos[hand_id] + finger_offset * R[:, 2]
    dist = float(np.linalg.norm(grasp_target - actual))
    rep.metrics["理想位置误差(m)"] = dist
    if dist > 0.005:
        rep.add("IK 残差大",
                "夹爪中心到目标 %.1f mm" % (dist * 1000))

    rep.ok = len(rep.problems) == 0
    return rep


def find_safe_grasp_height(model, xy, ik_fn, z_lo=0.005, z_hi=0.25,
                           hand_name="hand", finger_offset=0.0584,
                           table_z=0.0, q_init=None):
    """给定平面上一个点，二分查找"不撞桌子的最低抓取高度"。

    参数
    ----
    ik_fn : callable(target_xyz, q_init) -> (q, residual)
    q_init : 起始关节角（通常用 home）

    返回
    ----
    (z_safe, q_at_z_safe, report)
    """
    if q_init is None:
        q_init = np.zeros(7)

    def ok_at(z):
        tgt = np.array([xy[0], xy[1], z])
        q, _ = ik_fn(tgt, q_init)
        rep = check_grasp_pose(model, q, tgt, hand_name=hand_name,
                               finger_offset=finger_offset, table_z=table_z)
        # 只看碰撞和关节限位，力矩和伺服误差不算"不可达"
        blocking = [p for p in rep.problems
                    if p[0] in ("碰撞桌面", "离桌面太近", "关节超限",
                                "夹爪接触异物")]
        return (len(blocking) == 0), q, rep

    # 先确认高处可行
    ok_hi, q_hi, rep_hi = ok_at(z_hi)
    if not ok_hi:
        return None, None, rep_hi

    ok_lo, q_lo, rep_lo = ok_at(z_lo)
    if ok_lo:
        return z_lo, q_lo, rep_lo

    # 二分
    lo, hi = z_lo, z_hi
    best_z, best_q, best_rep = z_hi, q_hi, rep_hi
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        ok, q, rep = ok_at(mid)
        if ok:
            best_z, best_q, best_rep = mid, q, rep
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.001:
            break
    return best_z, best_q, best_rep
