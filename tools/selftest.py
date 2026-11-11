"""End-to-end smoke test that needs no Mixamo download.

Apple's reference skeleton is turned into a synthetic "Mixamo-like" rig (Mixamo bone
names, 3-bone fingers, no thumbs-in-ARKit quirk), a tube mesh is skinned to it, and the
real build_usd -> validate_usdz -> render_preview chain is run on the result.  Any break
in the USD authoring, the joint hierarchy, the skinning or the packaging shows up here
without waiting for a character download.

usage:
    python tools/selftest.py            # build, validate, render, then clean up
    python tools/selftest.py --keep     # leave build/zz_selftest and the usdz in place
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import arkit_fit as fit
import build_usd
import render_preview
import validate_usdz

SLUG = "zz_selftest"
RADIAL, RINGS = 16, 20          # tube resolution: enough verts per joint for the checks
RADIUS = {"torso": 0.115, "limb": 0.055, "finger": 0.012}


def mixamo_char(ref):
    """the joints build_target() expects, read off Apple's skeleton (so it is a real T-pose)"""
    p = ref.pos
    char = {"Hips": p["hips_joint"], "Spine": p["spine_2_joint"], "Spine1": p["spine_4_joint"],
            "Spine2": p["spine_7_joint"], "Neck": p["neck_1_joint"], "Head": p["head_joint"]}
    for side, S in (("left", "Left"), ("right", "Right")):
        for mix, arkit in (("Shoulder", "shoulder_1"), ("Arm", "arm"), ("ForeArm", "forearm"),
                           ("Hand", "hand"), ("UpLeg", "upLeg"), ("Leg", "leg"),
                           ("Foot", "foot"), ("ToeBase", "toes")):
            char[S + mix] = p["%s_%s_joint" % (side, arkit)]
        for an, mix in fit.FINGERS_ALL:
            pre = "hand" + an
            src = ["%s_%sStart_joint" % (side, pre), "%s_%s_1_joint" % (side, pre),
                   "%s_%s_2_joint" % (side, pre)]
            for i, j in enumerate(src, 1):
                char[S + mix + str(i)] = p[j]
            tip = (side + "_handThumb_2_joint" if an == "Thumb"
                   else side + "_hand%s_3_joint" % an)
            char[side + "_tip_" + an] = p[tip]
        char[side + "_tip_Toe"] = p[side + "_toesEnd_joint"]
    return char


def segments(pos):
    """(proximal, distal, radius, rings) for every bone the mesh has to cover"""
    out = [(pos[a], pos[b], RADIUS["torso"], RINGS)
           for a, b in zip(fit.TORSO_CHAIN, fit.TORSO_CHAIN[1:])]
    for side in ("left", "right"):
        arm = [side + "_shoulder_1_joint", side + "_arm_joint", side + "_forearm_joint",
               side + "_hand_joint"]
        leg = [side + "_upLeg_joint", side + "_leg_joint", side + "_foot_joint",
               side + "_toes_joint"]
        out += [(pos[a], pos[b], RADIUS["limb"], RINGS) for a, b in zip(arm, arm[1:])]
        out += [(pos[a], pos[b], RADIUS["limb"], RINGS) for a, b in zip(leg, leg[1:])]
        for _an, pre in validate_usdz.FINGERS:
            spans = [pre + "Start_joint", pre + "_1_joint", pre + "_2_joint"]
            if pre != "handThumb":
                spans.append(pre + "_3_joint")
            spans = [side + "_" + s for s in spans]
            out += [(pos[a], pos[b], RADIUS["finger"], 4) for a, b in zip(spans, spans[1:])]
    return [(a, b, r, n) for a, b, r, n in out if np.linalg.norm(np.subtract(b, a)) > 1e-4]


def tube(a, b, radius, rings=RINGS):
    """rings x radial cylinder between two joints -> points, triangles, uvs, normals"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return None
    axis = axis / length
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ang = np.linspace(0.0, 2.0 * np.pi, RADIAL, endpoint=False)
    pts, uvs, nrm = [], [], []
    for ti, tt in enumerate(np.linspace(0.0, 1.0, rings)):
        center = a + axis * (length * tt)
        for k, aa in enumerate(ang):
            n = u * np.cos(aa) + v * np.sin(aa)
            pts.append(center + n * radius)
            nrm.append(n)
            uvs.append((k / RADIAL, tt))
    tris = []
    for ti in range(rings - 1):
        for k in range(RADIAL):
            i0 = ti * RADIAL + k
            i1 = ti * RADIAL + (k + 1) % RADIAL
            tris += [[i0, i1, i1 + RADIAL], [i0, i1 + RADIAL, i0 + RADIAL]]
    return np.array(pts), np.array(tris, np.int32), np.array(uvs, np.float32), np.array(nrm, np.float32)


def bone_weights(points, pos, joints, parents, radius=0.13):
    """(nverts, njoints) weights from the distance to every bone segment"""
    dense = np.zeros((len(points), len(joints)))
    for i, j in enumerate(joints):
        parent = parents[i]
        if parent is None:
            continue
        origin = np.asarray(pos[parent], float)
        ab = np.asarray(pos[j], float) - origin
        L2 = float(ab @ ab)
        if L2 < 1e-8:
            d = np.linalg.norm(points - origin, axis=1)
        else:
            t = np.clip(((points - origin) @ ab) / L2, 0.0, 1.0)
            d = np.linalg.norm(points - (origin + t[:, None] * ab), axis=1)
        dense[:, i] = np.maximum(0.0, 1.0 - d / radius) ** 2
    return dense


def synth_mesh(pos, joints, parents):
    pts, tris, uvs, nrm = [], [], [], []
    base = 0
    for a, b, radius, rings in segments(pos):
        seg = tube(a, b, radius, rings)
        if seg is None:
            continue
        p, t, uv, n = seg
        pts.append(p)
        tris.append(t + base)
        uvs.append(uv)
        nrm.append(n)
        base += len(p)
    PTS = np.concatenate(pts)
    TRIS = np.concatenate(tris)

    # weights: smooth falloff from the bone SEGMENTS (a joint-point falloff would need a
    # radius wider than half the hip width to reach the tube surface, which makes both
    # legs share weights - and the LBS test rightly calls that a torn crotch)
    dense = bone_weights(PTS, pos, joints, parents)
    # the same torso redistribution extract_fbx.py does, so the zero-length spine_1 joint
    # is not the only joint left without any vertex
    torso = np.array([joints.index(j) for j in fit.TORSO_CHAIN])
    share = dense[:, torso].sum(1)
    dense[:, torso] = 0.0
    dense[:, torso] = fit.torso_kernel(PTS, pos) * share[:, None]
    jidx, jw, _ = fit.top_influences(dense, 4)

    tri_uv = np.concatenate(uvs)[TRIS].astype(np.float32)
    tri_n = np.concatenate(nrm)[TRIS].astype(np.float32)
    return dict(points=PTS.astype(np.float32), tris=TRIS.astype(np.int32), tri_uv=tri_uv,
                tri_normals=tri_n, tri_mat=np.zeros(len(TRIS), np.int32),
                joint_idx=jidx, joint_w=jw)


def run(keep=False):
    ref = fit.Ref(config.SKEL_JSON)
    char = mixamo_char(ref)
    pos, _D = fit.build_target(char, ref, height_scale=1.0)
    mesh = synth_mesh(pos, ref.joints, ref.parents)

    bdir = config.build_dir(SLUG)
    if os.path.isdir(bdir):
        shutil.rmtree(bdir)
    os.makedirs(bdir, exist_ok=True)
    np.savez_compressed(os.path.join(bdir, "mesh.npz"), **mesh)
    meta = {"model": SLUG, "joints": ref.joints,
            "joint_positions": {j: [float(x) for x in pos[j]] for j in ref.joints},
            "materials": [{"name": "Synth", "baseColor": None, "normal": None,
                           "alpha": False, "roughness": 0.6, "metallic": 0.0}],
            "counts": {"verts": int(len(mesh["points"])), "tris": int(len(mesh["tris"])),
                       "geometries": 1},
            "warnings": []}
    with open(os.path.join(bdir, "mesh_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("synthetic rig: %d verts, %d tris" % (meta["counts"]["verts"], meta["counts"]["tris"]))

    print("\n== build_usd")
    if not build_usd.build(SLUG):
        return 1
    print("\n== validate_usdz")
    fails, warns = validate_usdz.validate(config.usdz_path(SLUG))
    print("\n== render_preview")
    render_preview.render_character(SLUG)

    print("")
    print("SELFTEST:", "PASSED" if not fails else "%d FAILURES" % len(fails))
    for f in fails:
        print("  -", f)
    if keep:
        print("kept:", bdir, "and", config.usdz_path(SLUG))
    else:
        shutil.rmtree(bdir, ignore_errors=True)
        if os.path.exists(config.usdz_path(SLUG)):
            os.remove(config.usdz_path(SLUG))
        print("cleaned up (use --keep to inspect the output)")
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="end-to-end smoke test without a Mixamo model")
    ap.add_argument("--keep", action="store_true", help="keep the generated files")
    a = ap.parse_args(argv)
    config.ensure_dirs()
    return run(a.keep)


if __name__ == "__main__":
    sys.exit(main())
