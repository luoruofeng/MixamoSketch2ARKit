"""Author the RealityKit / ARKit BodyTrackedEntity USDZ for one character.

  * 91 joints, Apple ARSkeletonDefinition.defaultBody3D naming + hierarchy
  * every joint keeps the bind-pose orientation of Apple's own biped_robot.usdz
  * joint positions fitted to the character (T-pose, metres, +Y up, +Z forward)
  * single skeleton, <=4 influences per vertex, no animation

usage: python tools/build_usd.py <ModelSlug> [...]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, UsdUtils, Vt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def ident(name, used):
    """USD-safe, unique identifier for a material / subset name"""
    n = re.sub(r"[^0-9A-Za-z_]", "_", name or "")
    if not n or n[0].isdigit():
        n = "m_" + n
    base, k = n, 1
    while n in used:
        k += 1
        n = "%s_%d" % (base, k)
    used.add(n)
    return n


def gf(m):
    return Gf.Matrix4d([[float(m[r][c]) for c in range(4)] for r in range(4)])


def build(slug):
    bdir = config.build_dir(slug)
    for needed in ("mesh_meta.json", "mesh.npz"):
        if not os.path.exists(os.path.join(bdir, needed)):
            raise SystemExit("%s: missing %s - run tools/extract_fbx.py %s"
                             % (slug, needed, slug))
    with open(config.SKEL_JSON, encoding="utf-8") as fh:
        SKEL = json.load(fh)
    with open(os.path.join(bdir, "mesh_meta.json"), encoding="utf-8") as fh:
        META = json.load(fh)
    MESH = np.load(os.path.join(bdir, "mesh.npz"))

    JOINTS = SKEL["joints"]
    PARENTS = SKEL["parents"]
    ROBOT_BIND = [np.array(m, float) for m in SKEL["bind_world"]]
    POS = {k: np.array(v, float) for k, v in META["joint_positions"].items()}
    N = len(JOINTS)
    if N != len(PARENTS) or len(POS) != N:
        raise SystemExit("%s: reference skeleton / joint_positions disagree (%d / %d joints)"
                         % (slug, N, len(POS)))

    # Apple's biped_robot.usdz has hips_joint exactly at the origin: the ARBodyAnchor
    # origin *is* the hips, so the asset must be built the same way or the character
    # floats / sinks in AR.
    ORIGIN_SHIFT = -POS["hips_joint"]
    for k in POS:
        POS[k] = POS[k] + ORIGIN_SHIFT

    # ------------------------------------------------------------ joint transforms
    # world matrix = Apple's bind-pose orientation + this character's joint position
    W = {}
    for i, j in enumerate(JOINTS):
        R = ROBOT_BIND[i][:3, :3]
        u, _, vt = np.linalg.svd(R)                    # re-orthonormalise
        M = np.eye(4)
        M[:3, :3] = u @ vt
        M[3, :3] = POS[j]
        W[j] = M

    REST = {}
    for i, j in enumerate(JOINTS):
        p = PARENTS[i]
        # row-vector convention: bind[child] = rest[child] * bind[parent]
        REST[j] = W[j] if p is None else W[j] @ np.linalg.inv(W[p])

    # ------------------------------------------------------------ stage
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Apple's biped_robot.usdz uses SkelRoot + kind=component as the default prim; the
    # loader looks for a SkelRoot to find the skeleton, so mirror that layout.
    root = UsdSkel.Root.Define(stage, "/" + slug).GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    root.SetAssetInfo({"name": slug})
    UsdGeom.Xform.Define(stage, "/%s/Looks" % slug)

    # --- skeleton (mirrors Apple's biped_robot.usdz: SkelRoot/xform/skeleton "root")
    UsdGeom.Xform.Define(stage, "/%s/%s_skeleton" % (slug, slug))
    skel = UsdSkel.Skeleton.Define(stage, "/%s/%s_skeleton/root" % (slug, slug))
    # Apple encodes the joint hierarchy as nested relative paths
    # ("root/hips_joint/spine_1_joint/..."), which is how the parent/child tree is
    # recovered at load time; the flat names are the last path component.
    PATHS = {}
    for i, j in enumerate(JOINTS):
        p = PARENTS[i]
        PATHS[j] = j if p is None else PATHS[p] + "/" + j
    skel.GetJointsAttr().Set(Vt.TokenArray([PATHS[j] for j in JOINTS]))
    skel.GetRestTransformsAttr().Set(Vt.Matrix4dArray([gf(REST[j]) for j in JOINTS]))
    skel.GetBindTransformsAttr().Set(Vt.Matrix4dArray([gf(W[j]) for j in JOINTS]))
    jp = np.array([POS[j] for j in JOINTS], np.float32)
    skel.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in jp.min(0)]),
                                            Gf.Vec3f(*[float(x) for x in jp.max(0)])]))
    skel_path = skel.GetPath()

    # --- geometry
    UsdGeom.Xform.Define(stage, "/%s/%s_geom" % (slug, slug))
    pts = (MESH["points"] + ORIGIN_SHIFT).astype(np.float32)
    tris = MESH["tris"].astype(np.int32)
    tri_uv = MESH["tri_uv"].astype(np.float32)
    tri_n = MESH["tri_normals"].astype(np.float32)
    tri_mat = MESH["tri_mat"].astype(np.int32)
    jidx = MESH["joint_idx"].astype(np.int32)
    jw = MESH["joint_w"].astype(np.float32)

    ln = np.linalg.norm(tri_n, axis=2, keepdims=True)
    tri_n = np.divide(tri_n, np.maximum(ln, 1e-8), out=np.zeros_like(tri_n), where=ln > 1e-8)

    mesh = UsdGeom.Mesh.Define(stage, "/%s/%s_geom/%s_mesh" % (slug, slug, slug))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(tris.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    mats = META["materials"]
    mesh.CreateDoubleSidedAttr().Set(any(m.get("alpha") for m in mats))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in pts.min(0)]),
                                            Gf.Vec3f(*[float(x) for x in pts.max(0)])]))

    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    nrm = pv.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying)
    nrm.Set(Vt.Vec3fArray.FromNumpy(tri_n.reshape(-1, 3).astype(np.float32)))
    st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray.FromNumpy(tri_uv.reshape(-1, 2).astype(np.float32)))

    # --- skinning
    bapi = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    bapi.CreateSkeletonRel().SetTargets([skel_path])
    bapi.CreateGeomBindTransformAttr().Set(Gf.Matrix4d(1.0))
    ji = bapi.CreateJointIndicesPrimvar(False, 4)
    ji.SetInterpolation(UsdGeom.Tokens.vertex)
    ji.Set(Vt.IntArray.FromNumpy(jidx.reshape(-1).astype(np.int32)))
    jwa = bapi.CreateJointWeightsPrimvar(False, 4)
    jwa.SetInterpolation(UsdGeom.Tokens.vertex)
    jwa.Set(Vt.FloatArray.FromNumpy(jw.reshape(-1).astype(np.float32)))

    # ------------------------------------------------------------ materials + subsets
    def make_material(info, subset_name):
        base = "/%s/Looks/%s" % (slug, subset_name)
        mp = UsdShade.Material.Define(stage, base)
        pbr = UsdShade.Shader.Define(stage, base + "/Surface")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(info.get("roughness", 0.62)))
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(info.get("metallic", 0.0)))
        pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        pbr.CreateInput("opacityMode", Sdf.ValueTypeNames.Token).Set("opaque")
        pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.45)
        mp.CreateSurfaceOutput().ConnectToSource(pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token))

        uv = UsdShade.Shader.Define(stage, base + "/uvReader")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        if info.get("baseColor"):
            tex = UsdShade.Shader.Define(stage, base + "/baseColorTex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("tex/" + info["baseColor"]))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.CreateOutput("result", Sdf.ValueTypeNames.Float2))
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
            pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            if info.get("alpha"):
                pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
                    tex.CreateOutput("a", Sdf.ValueTypeNames.Float))
                pbr.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.35)
        else:
            pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))

        if info.get("normal"):
            ntex = UsdShade.Shader.Define(stage, base + "/normalTex")
            ntex.CreateIdAttr("UsdUVTexture")
            ntex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("tex/" + info["normal"]))
            ntex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            ntex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
            ntex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
            ntex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.CreateOutput("result", Sdf.ValueTypeNames.Float2))
            ntex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                ntex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        return mp

    # one pass: every material gets exactly one GeomSubset, and that subset is bound to
    # the material authored for it (the previous version created the subsets twice and
    # relied on the name generator producing the same names both times)
    used = set()
    subset_of_mat = {}
    for mi, info in enumerate(mats):
        faces = np.nonzero(tri_mat == mi)[0]
        if not len(faces):
            continue
        nm = ident(info["name"], used)
        sub = UsdGeom.Subset.Define(stage, "/%s/%s_geom/%s_mesh/%s" % (slug, slug, slug, nm))
        sub.CreateFamilyNameAttr().Set("materialBind")
        sub.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        sub.CreateIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.astype(np.int32)))
        sub.GetPrim().SetDisplayName(info["name"])
        UsdShade.MaterialBindingAPI(sub.GetPrim()).Bind(make_material(info, nm))
        subset_of_mat[nm] = mi

    covered = np.zeros(len(tris), bool)
    for mi in subset_of_mat.values():
        covered |= (tri_mat == mi)
    if not covered.all():
        print("warning: %d faces have no material subset" % int((~covered).sum()))

    # ------------------------------------------------------------ write
    with open(os.path.join(bdir, "usd_info.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": slug, "root": "/" + slug, "subsets": subset_of_mat,
                   "doubleSided": bool(mesh.GetDoubleSidedAttr().Get())}, fh, indent=1)

    usdc = config.usdc_path(slug)
    stage.GetRootLayer().Export(usdc)
    out_usdz = config.usdz_path(slug)
    os.makedirs(os.path.dirname(out_usdz), exist_ok=True)
    if os.path.exists(out_usdz):
        os.remove(out_usdz)

    # CreateNewARKitUsdzPackage is the ARKit-specific packager (it also enforces the
    # flattened-skinning layout); fall back to the plain one on very new USD builds.
    packager = getattr(UsdUtils, "CreateNewARKitUsdzPackage", None) or UsdUtils.CreateNewUsdzPackage
    ok = packager(usdc, out_usdz)
    size = os.path.getsize(out_usdz) if os.path.exists(out_usdz) else 0
    print("wrote", usdc)
    print("usdz package:", ok, out_usdz, "%.1f MB" % (size / 1e6))
    return ok and size > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="author the ARKit USDZ for one character")
    ap.add_argument("slugs", nargs="*", help="model slugs (default: every .fbx in models/)")
    a = ap.parse_args(argv)
    slugs = a.slugs or config.all_models()
    if not slugs:
        raise SystemExit("no models: put the Mixamo .fbx files in %s" % config.MODELS_DIR)
    bad = [s for s in slugs if not build(s)]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
