"""Validate a produced USDZ against Apple's "Validating a Model for Motion Capture"
checklist, prove the skinning actually works (LBS test), and verify the things that
went wrong before:

  * every bone direction matches Apple's reference (was 6-20 deg off), except the two
    chains that deliberately keep the character's own shape: the hand (a robot paddle,
    not a hand) and the leg stance (a robot with bent knees and a shallower foot)
  * the foot stays level: the character's own ankle -> ball pitch and knee bend survive
    the fit, so the sole is not tilted off the ground (was 3-15 deg toes-up)
  * every joint that carries skinning has weight (was 39 joints at zero)
  * finger _3 / End have a real length (was 0 mm, tip verts all on _2)
  * spine / neck weights are continuous (was spine_1/3/5/7 and neck_1/3/4 at zero)
  * T-pose is a real T-pose (legs vertical, arms horizontal)

usage:
    python tools/validate_usdz.py                      # every .usdz in dist/
    python tools/validate_usdz.py dist/Foo_ARKit.usdz  # one file
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdSkel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import arkit_fit as fit

NO_SKIN_OK_BASE = {"root", "jaw_joint", "chin_joint", "nose_joint",
                   "left_toesEnd_joint", "right_toesEnd_joint"}
FINGERS = [("Thumb", "handThumb"), ("Index", "handIndex"), ("Mid", "handMid"),
           ("Ring", "handRing"), ("Pinky", "handPinky")]


def expected_hierarchy():
    """Apple's ARSkeletonDefinition.defaultBody3D: joint -> parent."""
    exp = {"root": None, "hips_joint": "root"}
    for k in range(1, 8):
        exp["spine_%d_joint" % k] = "hips_joint" if k == 1 else "spine_%d_joint" % (k - 1)
    for k in range(1, 5):
        exp["neck_%d_joint" % k] = "spine_7_joint" if k == 1 else "neck_%d_joint" % (k - 1)
    exp["head_joint"] = "neck_4_joint"
    exp.update({"jaw_joint": "head_joint", "chin_joint": "jaw_joint", "nose_joint": "head_joint"})
    for s in ("left", "right"):
        exp.update({
            s + "_eye_joint": "head_joint",
            s + "_eyeUpperLid_joint": s + "_eye_joint",
            s + "_eyeLowerLid_joint": s + "_eye_joint",
            s + "_eyeball_joint": s + "_eye_joint",
            s + "_shoulder_1_joint": "spine_7_joint",
            s + "_arm_joint": s + "_shoulder_1_joint",
            s + "_forearm_joint": s + "_arm_joint",
            s + "_hand_joint": s + "_forearm_joint",
            s + "_upLeg_joint": "hips_joint",
            s + "_leg_joint": s + "_upLeg_joint",
            s + "_foot_joint": s + "_leg_joint",
            s + "_toes_joint": s + "_foot_joint",
            s + "_toesEnd_joint": s + "_toes_joint",
        })
        for f, pre in FINGERS:
            chain = ([pre + "Start_joint", pre + "_1_joint", pre + "_2_joint", pre + "End_joint"]
                     if f == "Thumb" else
                     [pre + "Start_joint", pre + "_1_joint", pre + "_2_joint",
                      pre + "_3_joint", pre + "End_joint"])
            exp[s + "_" + chain[0]] = s + "_hand_joint"
            for a, b in zip(chain[1:], chain[:-1]):
                exp[s + "_" + a] = s + "_" + b
    return exp


def no_skin_ok():
    """joints Apple documents as optional / leaf and that may legitimately carry no skin"""
    out = set(NO_SKIN_OK_BASE)
    for s in ("left", "right"):
        for j in ("eye", "eyeUpperLid", "eyeLowerLid", "eyeball"):
            out.add("%s_%s_joint" % (s, j))
        for f in ("Thumb", "Index", "Mid", "Ring", "Pinky"):
            out.add("%s_hand%sEnd_joint" % (s, f))
    return out


def validate(usdz, skel_json=None):
    """returns (fails, warns) - empty `fails` means the asset is ARKit-ready"""
    fails, warns = [], []

    def check(cond, msg):
        print(("  OK   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    def warn(cond, msg):
        print(("  OK   " if cond else "  WARN ") + msg)
        if not cond:
            warns.append(msg)

    EXPECTED = expected_hierarchy()
    NO_SKIN_OK = no_skin_ok()

    print("== stage ==")
    print("      file:", usdz, "%.1f MB" % (os.path.getsize(usdz) / 1e6))
    stage = Usd.Stage.Open(usdz)
    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    dp = stage.GetDefaultPrim()
    check(up == "Y", "upAxis == Y (got %s)" % up)
    check(abs(mpu - 1.0) < 1e-6, "metersPerUnit == 1 (got %s)" % mpu)
    check(bool(dp) and dp.GetPath().IsRootPrimPath(),
          "defaultPrim set: %s" % (dp.GetPath() if dp else None))
    check(bool(dp) and dp.GetTypeName() == "SkelRoot",
          "defaultPrim is a SkelRoot (Apple's layout, got %s)"
          % (dp.GetTypeName() if dp else "-"))

    # usdchecker: usdSkelValidators:SkelBindingAPIValidator.InvalidSkelBindingAPIApply.
    # Without a SkelRoot above the skinned mesh RealityKit ignores the skinning entirely
    # and only draws the bind pose (the "T-pose that never moves" bug).
    print("== SkelRoot layout ==")
    roots = [p for p in stage.Traverse() if p.IsA(UsdSkel.Root)]
    check(bool(roots), "a SkelRoot exists: %s" % ([str(r.GetPath()) for r in roots] or "none"))
    unrooted = []
    for p in stage.Traverse():
        if not (p.HasAPI(UsdSkel.BindingAPI) or p.IsA(UsdSkel.Skeleton)) or p.IsA(UsdSkel.Root):
            continue
        anc = p.GetParent()
        while anc and not anc.IsA(UsdSkel.Root):
            anc = anc.GetParent()
        if not anc:
            unrooted.append(str(p.GetPath()))
    check(not unrooted, "every skinned prim / Skeleton sits under a SkelRoot"
          + ("" if not unrooted else " -> not rooted: %s" % unrooted))

    skels = [p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    check(len(skels) == 1, "exactly one Skeleton (%d)" % len(skels))
    check(len(meshes) >= 1, "mesh present (%d)" % len(meshes))
    if len(skels) != 1 or not meshes:
        fails.append("cannot continue without one skeleton and one mesh")
        return fails, warns

    skel = UsdSkel.Skeleton(skels[0])
    joints = list(skel.GetJointsAttr().Get())
    names = [str(j).split("/")[-1] for j in joints]

    print("== joints ==")
    check(len(names) == 91, "91 joints (got %d)" % len(names))
    check(set(names) == set(EXPECTED), "joint name set matches ARSkeletonDefinition.defaultBody3D")
    if set(names) != set(EXPECTED):
        print("     missing:", sorted(set(EXPECTED) - set(names)))
        print("     extra  :", sorted(set(names) - set(EXPECTED)))

    bind = skel.GetBindTransformsAttr().Get()
    rest = skel.GetRestTransformsAttr().Get()
    idx = {n: i for i, n in enumerate(names)}
    hier = {}
    for j in joints:
        j = str(j)
        parts = j.split("/")
        hier[parts[-1]] = parts[-2] if len(parts) > 1 else None
    bad = [(n, hier[n], EXPECTED.get(n)) for n in names if hier[n] != EXPECTED.get(n)]
    check(not bad, "parent hierarchy encoded in joint paths, matches Apple's table"
          + ("" if not bad else " -> %s" % bad[:6]))

    def bpos(n):
        return np.array(bind[idx[n]])[3, :3]

    check(np.linalg.norm(bpos("hips_joint")) < 1e-5,
          "hips_joint at the origin (ARBodyAnchor) : %s" % np.round(bpos("hips_joint"), 5))
    check(np.linalg.norm(bpos("spine_1_joint") - bpos("hips_joint")) < 1e-5,
          "spine_1_joint overlaps hips_joint (T-pose, Apple does the same)")

    _mpts = np.array(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get(), np.float32)
    height = float(_mpts[:, 1].max() - _mpts[:, 1].min())
    SC = height / 1.742
    print("      character height %.3f m (scale factor %.2f)" % (height, SC))

    err = 0.0
    for n in names:
        p = hier[n]
        lhs = np.array(bind[idx[n]])
        rhs = np.array(rest[idx[n]]) @ np.array(bind[idx[p]]) if p else np.array(rest[idx[n]])
        err = max(err, float(np.abs(lhs - rhs).max()))
    check(err < 1e-4, "bindTransforms == restTransform * parent_bind (err %.2e)" % err)

    print("== reference alignment (the point of the whole exercise) ==")
    ref = fit.Ref(skel_json or config.SKEL_JSON)
    rows = fit.compare({n: bpos(n) for n in names}, ref, height_scale=height / ref.height)
    # The hand is deliberately NOT snapped to the reference: Apple's reference hand is a
    # robot paddle whose finger directions point 10-40 deg away from a human hand's, and
    # forcing a character's fingers onto them tore the hand apart.  The eye offset follows
    # the character's own eyeballs.
    # The leg chain is deliberately not snapped either (arkit_fit.DEFAULT_SAGITTAL): only
    # its sideways splay follows the reference, the pitch is the character's own, because
    # the robot stands with a 15 deg knee bend and an ankle->ball line 12 deg shallower
    # than a human foot's.  Forcing those onto a human lifted the whole foot off the
    # ground and added a permanent bent knee.  Its own check is below.
    legs = set(n for s in ("left", "right")
               for n in (s + "_leg_joint", s + "_foot_joint", s + "_toes_joint",
                         s + "_toesEnd_joint"))
    body = [r for r in rows if "_hand" not in r[0] and "eye" not in r[0] and r[0] not in legs]
    leg_rows = [r for r in rows if r[0] in legs]
    ddir = max(body, key=lambda r: r[1])
    # leaf joints (finger tips) accumulate the length error of a whole chain, so only
    # joints that carry a real segment say something about the retarget quality
    inner = [r for r in body if not (r[0].endswith("End_joint")
                                     or r[0] in ("jaw_joint", "chin_joint", "nose_joint"))]
    dpos = max(inner, key=lambda r: r[2])
    check(ddir[1] < 3.0, "every body bone direction follows Apple's reference: worst %s %.2f deg"
          % (ddir[0], ddir[1]))
    hand = [r for r in rows if "_hand" in r[0]]
    if hand:
        hdir = max(hand, key=lambda r: r[1])
        print("      hand keeps the character's own finger spread: worst %s %.1f deg off "
              "the reference's paddle hand (expected, not an error)" % (hdir[0], hdir[1]))
    if leg_rows:
        ldir = max(leg_rows, key=lambda r: r[1])
        print("      leg keeps the character's own stance: worst %s %.1f deg off the "
              "reference's bent-knee robot (expected, checked below)" % (ldir[0], ldir[1]))
    print("      largest body-joint position deviation vs the reference scaled to %.3f m: "
          "%s %.1f mm" % (height, dpos[0], dpos[2]))
    print("      (residual = segment LENGTH differences; the reference is a robot with much"
          " longer legs / neck / fingers)")
    warn(dpos[2] < 60.0, "body-joint position deviation < 60 mm (worst %.1f mm @%s)"
         % (dpos[2], dpos[0]))

    # ---- the foot has to stay level: no fit = no change of the character's own stance
    print("== foot stance (the mesh must keep the character's own, not Apple's robot) ==")
    slug = os.path.basename(usdz)
    slug = slug[:-len("_ARKit.usdz")] if slug.endswith("_ARKit.usdz") else slug[:-5]
    stance_meta, fit_meta = {}, {}
    meta_path = os.path.join(config.build_dir(slug), "mesh_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            fit_meta = json.load(fh).get("fit", {})
        stance_meta = fit_meta.get("stance", {})
    # only chains that were actually fitted with the character's own stance can be held to
    # it; a run with DEFAULT_SAGITTAL overridden to False is a deliberate choice
    flags = fit_meta.get("stance_sagittal", fit.DEFAULT_SAGITTAL)

    def pitch(v):
        return math.degrees(math.asin(float(np.clip(v[1] / max(np.linalg.norm(v), 1e-9),
                                                  -1.0, 1.0))))

    def _report(chain, cond, msg):
        """stance check that only binds for a chain fitted with the character's own
        stance; a chain deliberately snapped to the reference is just reported"""
        if flags.get(chain, True):
            check(cond, msg)
        else:
            print("      (%s snapped to the reference on purpose) %s" % (chain, msg))

    for s in ("left", "right"):
        ankle, ball = bpos(s + "_foot_joint"), bpos(s + "_toes_joint")
        knee = bpos(s + "_leg_joint")
        foot_pitch = pitch(ball - ankle)
        # a planted foot puts the ball of the foot below the ankle (Apple's robot -22.9,
        # every Mixamo character -26..-38); a flat foot whose ball sits higher means the
        # whole foot was pitched up by the fit
        check(-50.0 < foot_pitch < -8.0,
              "%s foot pitch (ankle -> ball) %+.1f deg - the ball sits below the ankle"
              % (s, foot_pitch))
        own = stance_meta.get(s, {})
        if own:
            d = foot_pitch - own.get("foot_pitch_own", foot_pitch)
            msg = ("%s foot keeps the character's own pitch (%+.2f vs %+.2f deg, "
                   "%+.2f off -> sole stays on the ground)"
                   % (s, foot_pitch, own.get("foot_pitch_own", 0.0), d))
            _report("foot", abs(d) < 1.5, msg)
            knee_now = 180.0 - math.degrees(math.acos(float(np.clip(
                fit.unit(knee - bpos(s + "_upLeg_joint"))
                @ fit.unit(ankle - knee), -1.0, 1.0))))
            dk = knee_now - own.get("knee_bend_own", knee_now)
            # straightening the sideways splay can only make the knee a hair straighter;
            # what must never happen is the reference's extra 8-15 deg of bend
            _report("leg", -3.0 < dk < 6.0,
                    "%s knee keeps the character's own bend (%.1f vs %.1f deg, %+.1f off)"
                    % (s, knee_now, own.get("knee_bend_own", 0.0), dk))
        else:
            print("      (no %s - only the plausible-foot check above)" % meta_path)


    emax, worst = 0.0, None
    rb = {n: np.array(m) for n, m in zip(ref.joints, ref.bind_world)}
    for n in names:
        d = np.abs(np.array(bind[idx[n]])[:3, :3] - rb[n][:3, :3]).max()
        if d > emax:
            emax, worst = d, n
    check(emax < 1e-4, "every joint keeps Apple's bind-pose orientation (max dev %.2e @%s)"
          % (emax, worst))

    print("== bone lengths ==")
    zero_ok = {"spine_1_joint", "left_eyeUpperLid_joint", "left_eyeLowerLid_joint",
               "left_eyeball_joint", "right_eyeUpperLid_joint", "right_eyeLowerLid_joint",
               "right_eyeball_joint", "hips_joint", "root"}
    zero = [n for n in names if n not in zero_ok
            and np.linalg.norm(bpos(n) - bpos(hier[n])) < 1e-4]
    check(not zero, "no collapsed bones (0 mm) outside Apple's own overlaps"
          + ("" if not zero else " -> %s" % zero))

    tips = {}
    for s in ("left", "right"):
        for an, pre in FINGERS:
            j = "%s_%s_3_joint" % (s, pre)
            if j in idx:
                tips[j] = float(np.linalg.norm(bpos("%s_%sEnd_joint" % (s, pre)) - bpos(j)) * 1000)
    # `_3 -> End` is the character's own distal phalanx (see build_target), so it is
    # checked against the middle phalanx instead of an absolute length: Apple's reference
    # ratio is 26 mm / 26 mm = 1.0, a human's tip is a bit shorter than the DIP segment.
    mids = {k: float(np.linalg.norm(bpos(k) - bpos(k.replace("_3_joint", "_2_joint")))) * 1000
            for k in tips}
    bad_tip = {k: (v, mids[k]) for k, v in tips.items()
               if v < 2.0 or not (0.4 <= v / max(mids[k], 1e-6) <= 2.2)}
    check(not bad_tip, "finger _3 -> End is a real distal phalanx (0.4-2.2 x the middle "
                       "phalanx, Apple 1.0) -> %s"
          % ", ".join("%s %.1f mm (mid %.1f)" % (k, v, mids[k])
                      for k, v in sorted(tips.items())[:4]))
    if bad_tip:
        print("     out of range:", {k: "%.1f mm vs mid %.1f mm" % v
                                     for k, v in bad_tip.items()})

    # Apple's finger is  metacarpal(Start->_1) > proximal > middle > distal : the
    # metacarpal is the longest bone of the chain and the joint that carries it sits
    # inside the palm.  Mapping Mixamo's phalanges onto Start.._2 (one joint too
    # proximal) makes the proximal and middle phalanx swap roles, which this catches.
    flat = []
    for s in ("left", "right"):
        for an, pre in FINGERS:
            k = "%s_%s" % (s, pre)
            if k + "_3_joint" not in idx:
                continue
            meta = np.linalg.norm(bpos(k + "Start_joint") - bpos(k + "_1_joint"))
            prox = np.linalg.norm(bpos(k + "_1_joint") - bpos(k + "_2_joint"))
            mid = np.linalg.norm(bpos(k + "_2_joint") - bpos(k + "_3_joint"))
            if meta <= prox or meta <= mid:
                flat.append("%s (%.0f / %.0f / %.0f mm)"
                            % (k, meta * 1000, prox * 1000, mid * 1000))
    check(not flat, "finger chain has Apple's proportions: metacarpal (Start->_1) is the "
                    "longest segment" + ("" if not flat else " -> %s" % flat))

    # ...and the joints run in order from inside the palm to the tip.  Measured on
    # Apple's reference: Start 0.12, _1 0.51, _2 0.74, _3 0.87, End 1.00 of the
    # wrist -> tip distance.  Anything that puts a whole phalanx on the wrong joint
    # (Start at the knuckle) shows up here as t(Start) > 0.35.
    out_of_order = []
    for s in ("left", "right"):
        for an, pre in FINGERS:
            chain = [s + "_" + pre + k for k in
                     ("Start_joint", "_1_joint", "_2_joint", "_3_joint", "End_joint")]
            if any(j not in idx for j in chain):
                continue
            hand = bpos(s + "_hand_joint")
            axis = bpos(chain[-1]) - hand
            L2 = max(float(axis @ axis), 1e-9)
            t = [float((bpos(j) - hand) @ axis / L2) for j in chain]
            if t[0] > 0.35 or not all(b > a + 0.01 for a, b in zip(t, t[1:])):
                out_of_order.append("%s %s (%s)" % (s, pre, np.round(t, 2)))
    check(not out_of_order, "finger joints run metacarpal -> tip in order, Start inside "
                            "the palm (Apple: 0.12/0.51/0.74/0.87/1.00)"
          + ("" if not out_of_order else " -> %s" % out_of_order))

    print("== T-pose sanity ==")
    larm, lfa, lhand = bpos("left_arm_joint"), bpos("left_forearm_joint"), bpos("left_hand_joint")
    drop = abs(lhand[1] - larm[1]) / np.linalg.norm(lhand - larm)
    check(drop < 0.12, "T-pose: left arm horizontal (%.1f deg droop)"
          % math.degrees(math.asin(min(1, drop))))
    check(abs(lfa[1] - larm[1]) < 0.03 * SC, "T-pose: elbow at shoulder height (%.0f mm)"
          % (abs(lfa[1] - larm[1]) * 1000))
    for s in ("left", "right"):
        hip, knee, ankle = bpos(s + "_upLeg_joint"), bpos(s + "_leg_joint"), bpos(s + "_foot_joint")
        lat = max(abs(knee[0] - hip[0]), abs(ankle[0] - knee[0]))
        check(lat < 0.02 * SC, "%s leg is vertical in X: hip %.3f knee %.3f ankle %.3f "
                               "(drift %.0f mm)" % (s, hip[0], knee[0], ankle[0], lat * 1000))
    feet = bpos("left_foot_joint")
    check(feet[1] < bpos("hips_joint")[1] - 0.25 * SC and abs(feet[0]) < 0.20 * SC,
          "legs straight down / feet below hips: %s" % np.round(feet, 3))
    check(np.allclose(bpos("left_upLeg_joint")[1], bpos("right_upLeg_joint")[1], atol=0.02 * SC),
          "symmetric hips")

    print("== skinning ==")
    tot_v, maxinf, sumerr, maxidx = 0, 0, 0.0, 0
    cov = np.zeros(len(names), np.int64)
    wsum = np.zeros(len(names))
    for m in meshes:
        mg = UsdGeom.Mesh(m)
        nv = len(mg.GetPointsAttr().Get())
        tot_v += nv
        b = UsdSkel.BindingAPI(m)
        tgt = b.GetSkeletonRel().GetTargets()
        check(tgt == [skels[0].GetPath()], "%s : skel:skeleton -> %s" % (m.GetName(), tgt))
        ji = b.GetJointIndicesPrimvar()
        jw = b.GetJointWeightsPrimvar()
        if not ji or not jw:
            check(False, "%s: missing skel:jointIndices / skel:jointWeights" % m.GetName())
            continue
        ii = np.array(ji.Get(), np.int32).reshape(nv, -1)
        ww = np.array(jw.Get(), np.float32).reshape(nv, -1)
        maxinf = max(maxinf, int((ww > 0).sum(1).max()))
        sumerr = max(sumerr, float(np.abs(ww.sum(1) - 1.0).max()))
        maxidx = max(maxidx, int(ii.max()))
        check(ji.GetElementSize() == 4 and jw.GetElementSize() == 4, "elementSize 4")
        check(ji.GetInterpolation() == "vertex", "joint indices interpolation == vertex")
        for k in range(ii.shape[1]):
            sel = ww[:, k] > 1e-6
            cov += np.bincount(ii[sel, k], minlength=len(names))
            wsum += np.bincount(ii[sel, k], weights=ww[sel, k], minlength=len(names))
    check(maxinf <= 4, "max influences per vertex <= 4 (got %d)" % maxinf)
    check(sumerr < 1e-3, "weights normalised (max |sum-1| = %.2e)" % sumerr)
    check(maxidx < 91, "joint indices in range (max %d)" % maxidx)
    print("  verts:", tot_v)

    borrowed = sorted(n for i, n in enumerate(names) if cov[i] == 0 and n not in NO_SKIN_OK)
    check(not borrowed, "every deformable joint has vertex weight (%d zero-weight joints, "
                        "all of them face / leaf joints)" % int((cov == 0).sum())
          + ("" if not borrowed else " -> %s" % borrowed))

    print("== spine / neck weight continuity (was: 11 of 12 at zero) ==")
    # 200 vertices on a ~20 k-vertex character; scaled with the mesh so a low-poly
    # character is not failed for having few vertices (each joint still has to own a
    # real band of flesh, no joint may be left at zero)
    need = max(25, int(0.008 * tot_v))
    for j in (["spine_%d_joint" % k for k in range(1, 8)]
              + ["neck_%d_joint" % k for k in range(1, 5)]):
        check(int(cov[idx[j]]) > need,
              "%-14s %6d verts (> %d), total weight %8.1f"
              % (j, int(cov[idx[j]]), need, wsum[idx[j]]))

    print("== left / right symmetry ==")
    for a in ("arm", "forearm", "shoulder_1", "upLeg", "leg", "foot"):
        l, r = wsum[idx["left_" + a + "_joint"]], wsum[idx["right_" + a + "_joint"]]
        ratio = max(l, r) / max(min(l, r), 1e-6)
        warn(ratio < 1.8, "%s weight L %.0f / R %.0f (x%.2f)" % (a, l, r, ratio))

    print("== material subsets ==")
    for m in meshes:
        subs = [c for c in m.GetChildren()
                if c.IsA(UsdGeom.Subset)
                and UsdGeom.Subset(c).GetFamilyNameAttr().Get() == "materialBind"]
        if not subs:
            warn(False, "%s has no materialBind GeomSubset" % m.GetName())
            continue
        ftype = UsdGeom.Subset.GetFamilyType(UsdGeom.Imageable(m), "materialBind")
        check(ftype in ("partition", "nonOverlapping"),
              "%s materialBind familyType is %r (usdchecker wants partition)"
              % (m.GetName(), ftype))
        no_api = [c.GetName() for c in subs if not c.HasAPI(UsdShade.MaterialBindingAPI)]
        check(not no_api, "%d materialBind subsets carry MaterialBindingAPI" % len(subs)
              + ("" if not no_api else " -> not applied on %s" % no_api))
        unbound = [c.GetName() for c in subs
                   if not UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial()[0]]
        check(not unbound, "every materialBind subset resolves to a Material"
              + ("" if not unbound else " -> unbound: %s" % unbound))

    print("== no animation ==")
    anims = [p for p in stage.Traverse() if p.IsA(UsdSkel.Animation)]
    check(not anims, "no SkelAnimation (%d)" % len(anims))
    keyed = sum(1 for p in stage.Traverse() for a in p.GetAttributes() if a.GetNumTimeSamples())
    check(keyed == 0, "no time samples anywhere (%d)" % keyed)

    print("== skinning / LBS functional test ==")
    mg = UsdGeom.Mesh(meshes[0])
    pts = np.array(mg.GetPointsAttr().Get(), np.float32)
    b = UsdSkel.BindingAPI(meshes[0])
    ii = np.array(b.GetJointIndicesPrimvar().Get(), np.int32).reshape(len(pts), -1)
    ww = np.array(b.GetJointWeightsPrimvar().Get(), np.float32).reshape(len(pts), -1)
    bindM = [np.array(x) for x in bind]
    binv = [np.linalg.inv(x) for x in bindM]
    ph = np.c_[pts, np.ones(len(pts))]

    def skin(local_mats):
        world = {}
        for i, n in enumerate(names):
            p = hier[n]
            world[i] = local_mats[i] @ world[idx[p]] if p else local_mats[i]
        out = np.zeros_like(pts)
        for k in range(ii.shape[1]):
            j = ii[:, k]
            w = ww[:, k:k + 1]
            m = np.stack([world[x] for x in j])
            inv = np.stack([binv[x] for x in j])
            out += (ph[:, None, :] @ (inv @ m))[:, 0, :3] * w
        return out

    rest_local = [np.array(x) for x in rest]
    dev = float(np.abs(skin(rest_local) - pts).max())
    check(dev < 1e-3, "identity at bind pose (max deviation %.2e m)" % dev)

    def rotm(deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        R = np.eye(4)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, s, -s, c
        return R

    def posed(pairs):
        pose = [np.array(x) for x in rest_local]
        for name, deg in pairs:
            pose[idx[name]] = rotm(deg) @ rest_local[idx[name]]
        return pose

    fvi = np.array(mg.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
    rest_edge = np.linalg.norm(pts[fvi[:, 0]] - pts[fvi[:, 1]], axis=1)
    vis = rest_edge > 0.005

    for name, deg, max_disp in (("left_forearm_joint", 60.0, 0.55 * SC),
                                ("spine_3_joint", 20.0, 1.10 * SC),
                                ("left_upLeg_joint", 25.0, 0.60 * SC),
                                ("left_handMid_2_joint", 45.0, 0.10 * SC),
                                ("neck_2_joint", 25.0, 0.20 * SC)):
        moved = skin(posed([(name, deg)]))
        d = np.linalg.norm(moved - pts, axis=1)
        e = np.linalg.norm(moved[fvi[:, 0]] - moved[fvi[:, 1]], axis=1)
        ratio = e / np.maximum(rest_edge, 1e-6)
        check(0.0005 < d.max() < max_disp,
              "rotate %s %g deg -> %d verts move, max %.3f m (<= %.2f)"
              % (name, deg, int((d > 1e-4).sum()), d.max(), max_disp))
        # layered clothing (pants over leg, hood over head) legitimately slides; >3x on a
        # >5 mm edge is a torn seam, below that it is a fold
        if vis.any():
            check(float(ratio[vis].max()) < 3.0,
                  "  no stretched triangles (worst visible x%.2f on %.1f mm)"
                  % (float(ratio[vis].max()),
                     float(rest_edge[vis][np.argmax(ratio[vis])]) * 1000))

    # the fingertip must actually be driven by _3 now
    for s in ("left", "right"):
        j3 = idx[s + "_handMid_3_joint"]
        check(int(((ii == j3) & (ww > 0)).sum()) > 3,
              "%s handMid_3_joint drives %d vertex influences"
              % (s, int(((ii == j3) & (ww > 0)).sum())))

    dom = ii[np.arange(len(ii)), ww.argmax(1)]
    for jn, sign in (("left_hand_joint", 1), ("right_hand_joint", -1),
                     ("left_foot_joint", 1), ("right_foot_joint", -1)):
        sel = pts[dom == idx[jn]]
        if len(sel) == 0:
            check(False, "%s has no vertices" % jn)
            continue
        check(sel[:, 0].mean() * sign > 0, "%s region on the %s side (mean x=%.3f)"
              % (jn, "left(+X)" if sign > 0 else "right(-X)", sel[:, 0].mean()))

    d = np.linalg.norm(skin(posed([("left_forearm_joint", 60.0)])) - pts, axis=1)
    for jn in ("right_hand_joint", "left_foot_joint", "right_foot_joint", "head_joint"):
        m = float(d[dom == idx[jn]].max()) if (dom == idx[jn]).any() else 0.0
        check(m < 1e-4, "left forearm bend does not move %s (max %.2e m)" % (jn, m))

    return fails, warns


def main(argv=None):
    ap = argparse.ArgumentParser(description="validate ARKit-ready USDZ files")
    ap.add_argument("usdz", nargs="*", help="files to check (default: every .usdz in dist/)")
    ap.add_argument("--skel", default=None, help="reference skeleton json")
    a = ap.parse_args(argv)

    targets = a.usdz
    if not targets:
        if config.DIST_DIR.is_dir():
            targets = [str(p) for p in sorted(config.DIST_DIR.glob("*.usdz"))]
        if not targets:
            raise SystemExit("no usdz to validate in %s (run run_all.py first)" % config.DIST_DIR)

    total_fail = 0
    for path in targets:
        print("")
        print("=========", os.path.basename(path))
        fails, warns = validate(path, a.skel)
        print("RESULT:", "ALL CHECKS PASSED" if not fails else "%d FAILURES" % len(fails))
        for f in fails:
            print("  -", f)
        if warns:
            print("%d warnings:" % len(warns))
            for w in warns:
                print("  -", w)
        total_fail += len(fails)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
