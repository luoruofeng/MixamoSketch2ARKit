"""Tiny software rasteriser: preview the produced USDZ (bind pose + a test pose).

Renders straight from the packaged .usdz, so what is checked is exactly what ships:
bind pose front / side, plus one test pose (bent elbows, bent knee) that shows the
skin weights working.

usage: python tools/render_preview.py <ModelSlug> [...]
Writes build/<slug>/preview_tpose_front.png, preview_tpose_side.png, preview_pose_front.png
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom, UsdSkel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load(slug):
    bdir = config.build_dir(slug)
    usdz = config.usdz_path(slug)
    if not os.path.exists(usdz):
        raise SystemExit("%s missing - run tools/build_usd.py %s" % (usdz, slug))
    texdir = config.tex_dir(slug)

    meta_path = os.path.join(bdir, "mesh_meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    info_path = os.path.join(bdir, "usd_info.json")
    subset_mat = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as fh:
            subset_mat = json.load(fh).get("subsets", {})

    stage = Usd.Stage.Open(usdz)
    skel = UsdSkel.Skeleton([p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)][0])
    raw = [str(j) for j in skel.GetJointsAttr().Get()]
    names = [j.split("/")[-1] for j in raw]
    parents = {n: (r.split("/")[-2] if "/" in r else None) for n, r in zip(names, raw)}
    idx = {n: i for i, n in enumerate(names)}
    rest = [np.array(x) for x in skel.GetRestTransformsAttr().Get()]
    bind = [np.array(x) for x in skel.GetBindTransformsAttr().Get()]
    binv = [np.linalg.inv(b) for b in bind]

    mesh = UsdGeom.Mesh([p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)][0])
    pts = np.array(mesh.GetPointsAttr().Get(), float)
    fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
    fvc = np.array(mesh.GetFaceVertexCountsAttr().Get(), np.int32)
    if not (fvc == 3).all():
        raise SystemExit("%s: preview only handles triangles" % usdz)
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = np.array(pv.GetPrimvar("st").Get(), float).reshape(-1, 3, 2)
    b = UsdSkel.BindingAPI(mesh)
    ji = np.array(b.GetJointIndicesPrimvar().Get(), np.int32).reshape(len(pts), -1)
    jw = np.array(b.GetJointWeightsPrimvar().Get(), float).reshape(len(pts), -1)

    # material per face from the GeomSubsets
    face_mat = np.zeros(len(fvi), np.int32)
    mat_names = []
    for c in mesh.GetPrim().GetChildren():
        if c.GetTypeName() == "GeomSubset":
            mi = len(mat_names)
            mat_names.append(c.GetName())
            face_mat[np.array(UsdGeom.Subset(c).GetIndicesAttr().Get(), np.int32)] = mi
    print("subsets:", mat_names, "faces per mat:", np.bincount(face_mat))

    tex = []
    for mn in mat_names:
        k = subset_mat.get(mn, 0)
        ent = meta["materials"][k] if k < len(meta["materials"]) else {}
        path = os.path.join(texdir, ent.get("baseColor") or "")
        if not ent.get("baseColor") or not os.path.exists(path):
            tex.append(np.full((4, 4, 3), 0.75, np.float32))
            continue
        tex.append(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0)

    print("joints:", len(names), "| vertices:", len(pts), "| tris:", len(fvi))
    return dict(bdir=bdir, names=names, parents=parents, idx=idx, rest=rest, bind=bind,
                binv=binv, pts=pts, fvi=fvi, st=st, ji=ji, jw=jw, tex=tex, face_mat=face_mat)


def skin(scene, local):
    """linear blend skinning of the shipped mesh with the given local joint matrices"""
    names, parents, idx, pts, ji, jw, binv = (scene["names"], scene["parents"], scene["idx"],
                                              scene["pts"], scene["ji"], scene["jw"],
                                              scene["binv"])
    world = {}
    for i, n in enumerate(names):
        p = parents[n]
        world[i] = local[i] @ world[idx[p]] if p else local[i]
    ph = np.c_[pts, np.ones(len(pts))]
    out = np.zeros_like(pts)
    normals = np.zeros_like(pts)
    for k in range(ji.shape[1]):
        j = ji[:, k]
        w = jw[:, k:k + 1]
        m = np.stack([world[x] for x in j])
        inv = np.stack([binv[x] for x in j])
        out += (ph[:, None, :] @ (inv @ m))[:, 0, :3] * w
        R = np.stack([inv[i][:3, :3] @ m[i][:3, :3] for i in range(len(j))])
        normals += np.einsum("nij,nj->ni", R, scene["vn"]) * w
    return out, normals


def render(scene, pos, shade_n, path, cam_side=False, W=760, H=980, fov=32.0):
    fvi, st, tex, face_mat = scene["fvi"], scene["st"], scene["tex"], scene["face_mat"]
    lo, hi = pos.min(0), pos.max(0)
    center = (lo + hi) / 2.0
    height = max(float(hi[1] - lo[1]), 0.2)

    f = (H / 2) / math.tan(math.radians(fov) / 2)
    dist = height / (2 * math.tan(math.radians(fov) / 2)) * 1.16
    if cam_side:
        cp = np.array([dist, center[1] + 0.05 * height, dist * 0.18])
    else:
        cp = np.array([0.0, center[1] + 0.02 * height, dist])
    tg = np.array([0.0, center[1], 0.0])
    fwd = tg - cp
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = pos - cp
    cx, cy, cz = rel @ right, rel @ up, rel @ fwd
    sx = W / 2 + cx / np.maximum(cz, 1e-6) * f
    sy = H / 2 - cy / np.maximum(cz, 1e-6) * f
    order = np.argsort(-cz[fvi].mean(1))

    img = Image.new("RGB", (W, H), (18, 20, 26))
    dr = ImageDraw.Draw(img)
    L = np.array([0.35, 0.55, 0.75])
    L /= np.linalg.norm(L)
    # PIL allocates a mask the size of the polygon's bounding box, so a triangle that
    # projects far outside the canvas (a vertex right in front of the camera, where
    # cx/cz explodes, or a stretched test pose) makes it ask for gigabytes and crash the
    # process with a native access violation. Clamp and drop the ones that cannot show.
    mx = my = 8.0
    for t in order:
        v = fvi[t]
        if (cz[v] < 0.05).any():
            continue                                  # behind the camera
        px = np.clip(sx[v], -mx, W + mx)
        py = np.clip(sy[v], -my, H + my)
        if not (np.isfinite(px).all() and np.isfinite(py).all()):
            continue
        if px.max() <= 0.0 or px.min() >= W or py.max() <= 0.0 or py.min() >= H:
            continue                                  # entirely off screen
        poly = [(float(px[0]), float(py[0])), (float(px[1]), float(py[1])),
                (float(px[2]), float(py[2]))]
        uv = st[t].mean(0)
        im = tex[face_mat[t]]
        th, tw = im.shape[:2]
        u = int(np.clip(uv[0] % 1.0, 0, 0.999) * tw)
        vv = int(np.clip(1.0 - (uv[1] % 1.0), 0, 0.999) * th)
        base = im[vv, u]
        n = shade_n[v].mean(0)
        n = n / (np.linalg.norm(n) + 1e-9)
        lam = 0.30 + 0.70 * max(0.0, float(abs(n @ L)))
        dr.polygon(poly, fill=tuple(int(np.clip(x * 255, 0, 255)) for x in base * lam))
    img.save(path)
    print("wrote", path)


def render_character(slug):
    scene = load(slug)
    # per-vertex normals for shading (the USDZ only stores face-varying ones)
    pts, fvi = scene["pts"], scene["fvi"]
    vn = np.zeros_like(pts)
    fn = np.cross(pts[fvi[:, 1]] - pts[fvi[:, 0]], pts[fvi[:, 2]] - pts[fvi[:, 0]])
    for k in range(3):
        np.add.at(vn, fvi[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    scene["vn"] = vn

    p0, n0 = skin(scene, [np.array(x) for x in scene["rest"]])
    render(scene, p0, n0, os.path.join(scene["bdir"], "preview_tpose_front.png"))
    render(scene, p0, n0, os.path.join(scene["bdir"], "preview_tpose_side.png"), cam_side=True)

    # test pose: bent elbows, bent knee, a little spine / neck
    pose = [np.array(x) for x in scene["rest"]]

    def rot(axis, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        M = np.eye(4)
        if axis == "x":
            M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, s, -s, c
        elif axis == "y":
            M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, -s, s, c
        else:
            M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, s, -s, c
        return M

    idx = scene["idx"]
    for jn, ax, dg in (("left_forearm_joint", "y", 70), ("right_forearm_joint", "y", -70),
                       ("left_leg_joint", "x", -55), ("right_upLeg_joint", "x", 35),
                       ("spine_4_joint", "x", 12), ("neck_2_joint", "x", -12)):
        if jn in idx:
            pose[idx[jn]] = rot(ax, dg) @ pose[idx[jn]]
    p1, n1 = skin(scene, pose)
    render(scene, p1, n1, os.path.join(scene["bdir"], "preview_pose_front.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="render T-pose / test-pose previews")
    ap.add_argument("slugs", nargs="*", help="model slugs (default: every .fbx in models/)")
    a = ap.parse_args(argv)
    slugs = a.slugs or config.all_models()
    if not slugs:
        raise SystemExit("no models: put the Mixamo .fbx files in %s" % config.MODELS_DIR)
    for s in slugs:
        print("== render_preview", s)
        render_character(s)


if __name__ == "__main__":
    main()
