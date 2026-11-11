"""FBX scene graph: object tables, local/world matrices, bind pose, skin clusters.

Everything here works in *FBX world space* and in centimetres (the FBX unit);
the caller converts to metres / ARKit axes with `Scene.axis_matrix()`.
"""
import math
import re

import numpy as np

from fbx_binary import FBX, name_of

# ------------------------------------------------------------------ matrices
def vec(node, name, default):
    n = node.child(name)
    return np.array(n.props, float) if n and n.props else np.array(default, float)


def scalar(node, name, default):
    n = node.child(name)
    return n.props[0] if n and n.props else default


def _T(v):
    m = np.eye(4)
    m[3, :3] = v
    return m


def _axis_rot(axis, deg):
    """single axis rotation, row-vector convention (v' = v @ R)"""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    if axis == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, s, -s, c
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, -s, s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, s, -s, c
    return m


# FBX eEulerOrder: the order in which the axes are applied, first one first.
# With row vectors that is simply R = R_first @ R_second @ R_third.
_ROTATION_ORDER = {
    0: "xyz", 1: "xzy", 2: "yzx", 3: "yxz", 4: "zxy", 5: "zyx",
}


def _R(xyz, order=0):
    """Euler angles (degrees) -> 4x4 rotation, honouring the node's RotationOrder.

    FBX defaults to eEulerXYZ, which is what every Mixamo export uses; reading the
    order anyway keeps hand-edited or Maya-exported files from coming out sheared.
    """
    seq = _ROTATION_ORDER.get(int(order), "xyz")
    angles = dict(zip("xyz", xyz))
    m = np.eye(4)
    for axis in seq:
        m = m @ _axis_rot(axis, angles[axis])
    return m


def _S(v):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = v
    return m


def local_matrix(model):
    """Lcl Translation/Rotation/Scaling plus the pivots and pre/post rotations."""
    t = vec(model, "Lcl Translation", [0, 0, 0])
    r = vec(model, "Lcl Rotation", [0, 0, 0])
    s = vec(model, "Lcl Scaling", [1, 1, 1])
    roff = vec(model, "RotationOffset", [0, 0, 0])
    rpiv = vec(model, "RotationPivot", [0, 0, 0])
    rpre = vec(model, "PreRotation", [0, 0, 0])
    rpost = vec(model, "PostRotation", [0, 0, 0])
    soff = vec(model, "ScalingOffset", [0, 0, 0])
    spiv = vec(model, "ScalingPivot", [0, 0, 0])
    order = scalar(model, "RotationOrder", 0)
    return (_T(t) @ _T(roff) @ _T(rpiv) @ _R(rpre) @ _R(r, order) @ np.linalg.inv(_R(rpost))
            @ np.linalg.inv(_T(rpiv)) @ _T(soff) @ _T(spiv) @ _S(s) @ np.linalg.inv(_T(spiv)))


def clean_name(raw):
    """'mixamorig2:LeftHand' -> 'LeftHand' (any Mixamo namespace prefix)"""
    return re.sub(r"^mixamorig\d*:", "", name_of(raw)).strip()


class Scene:
    """Parsed FBX: object tables + connections (parent links restricted to Model<->Model)."""

    def __init__(self, path):
        self.f = FBX(path)
        objs = self.f.top("Objects")
        self.by = {}
        for o in objs.children:
            self.by.setdefault(o.name, []).append(o)
        self.uid2any = {}
        for lst in self.by.values():
            for o in lst:
                self.uid2any[o.props[0]] = o
        self.models = {clean_name(m.props[1]): m for m in self.by.get("Model", [])}
        self.uid2model = {m.props[0]: m for m in self.by.get("Model", [])}

        self.children_of = {}
        self.parent_of = {}       # any object -> any object
        self.parent_model = {}    # Model -> Model only
        for c in self.f.top("Connections").children:
            if c.props[0] != "OO":
                continue
            self.children_of.setdefault(c.props[2], []).append(c.props[1])
            self.parent_of[c.props[1]] = c.props[2]
            if c.props[1] in self.uid2model and c.props[2] in self.uid2model:
                self.parent_model[c.props[1]] = c.props[2]

        gs = self.f.top("GlobalSettings")
        props = gs.child("Properties70") if gs is not None else None

        def setting(name, default):
            for p in (props.children if props is not None else []):
                if p.props[0] == name:
                    return p.props[-1]
            return default

        self.up_axis = setting("UpAxis", 1)
        self.up_sign = setting("UpAxisSign", 1)
        self.front_axis = setting("FrontAxis", 2)
        self.front_sign = setting("FrontAxisSign", -1)
        self.coord_axis = setting("CoordAxis", 0)
        self.coord_sign = setting("CoordAxisSign", 1)
        self.unit_scale = setting("UnitScaleFactor", 1.0) / 100.0   # cm -> m

        self._world = {}
        self._axis = None

    # ---- space conversions
    def axis_matrix(self):
        """4x4 (row-vector) transform from FBX world space to ARKit space.

        ARKit wants +Y up and +Z forward.  Mixamo always exports Y-up, so this is
        usually the identity; a Z-up file (Maya / 3ds Max / Blender round trip) is
        rotated into place instead of producing a lying-down character.
        """
        if self._axis is not None:
            return self._axis
        if self.up_axis == 1 and self.up_sign == 1:
            m = np.eye(4)
        elif self.up_axis == 2 and self.up_sign == 1:
            m = np.eye(4)
            m[1, 2], m[2, 1] = -1.0, 1.0      # (x, y, z) -> (x, z, -y)
        else:
            raise SystemExit("unsupported FBX orientation: UpAxis=%r sign=%r"
                             % (self.up_axis, self.up_sign))
        self._axis = m
        return m

    def world(self, uid, depth=0):
        if uid in self._world:
            return self._world[uid]
        m = local_matrix(self.uid2model[uid])
        p = self.parent_model.get(uid)
        if p in self.uid2model and depth < 64:
            m = m @ self.world(p, depth + 1)
        self._world[uid] = m
        return m

    # ---- bind pose: union over every Pose node (multi-mesh files have one each)
    def bind_pose(self):
        """name -> 4x4 bind world matrix (row-vector, translation in row 3)"""
        out, dup = {}, {}
        for pose in self.by.get("Pose", []):
            for pn in pose.all("PoseNode"):
                node = pn.child("Node")
                matrix = pn.child("Matrix")
                if node is None or matrix is None:
                    continue
                u = node.props[0]
                if u not in self.uid2model:
                    continue
                nm = clean_name(self.uid2model[u].props[1])
                M = np.array(matrix.props[0], float).reshape(4, 4)
                if nm in out:
                    dup[nm] = max(dup.get(nm, 0.0), float(np.abs(M - out[nm]).max()))
                else:
                    out[nm] = M
        self.bind_dup_err = max(dup.values()) if dup else 0.0
        return out

    def bind_pose_clusters(self):
        """name -> 4x4 bind world matrix, read from the skin clusters' TransformLink.

        The Pose nodes are the primary source, but some Mixamo exports scramble them
        (every limb collapsed onto one line, feet above the hips).  TransformLink is the
        same bind pose the skinning itself uses, so it is the reliable fallback.

        returns (dict, max deviation between clusters that share a bone)
        """
        out, dup = {}, {}
        for d in self.by.get("Deformer", []):
            if len(d.props) < 3 or d.props[2] != "Cluster":
                continue
            m = d.child("TransformLink")
            if m is None:
                continue
            bone = None
            for ch in self.children_of.get(d.props[0], []):
                node = self.uid2model.get(ch)
                if node is not None and node.props[2] in ("LimbNode", "Limb"):
                    bone = clean_name(node.props[1])
            if bone is None:
                continue
            M = np.array(m.props[0], float).reshape(4, 4)
            if bone in out:
                dup[bone] = max(dup.get(bone, 0.0), float(np.abs(M - out[bone]).max()))
            else:
                out[bone] = M
        return out, (max(dup.values()) if dup else 0.0)

    def geometry_of(self):
        """[(geometry_node, mesh_model_uid, mesh_model_name)]"""
        out = []
        for g in self.by.get("Geometry", []):
            pm = self.parent_of.get(g.props[0])
            if pm not in self.uid2model:
                continue
            out.append((g, pm, clean_name(self.uid2model[pm].props[1])))
        return out

    def skin_clusters(self, geo_uid):
        """bone -> {local vertex index: weight} for one geometry, plus the Skin uid"""
        skinners = [d for d in self.by.get("Deformer", [])
                    if len(d.props) > 2 and d.props[2] == "Skin"]
        # FbxGeometry::AddDeformer connects the Skin as a child of the Geometry
        skin_uid = None
        for uid in self.children_of.get(geo_uid, []):
            o = self.uid2any.get(uid)
            if o is not None and o.name == "Deformer" and len(o.props) > 2 and o.props[2] == "Skin":
                skin_uid = uid
                break
        if skin_uid is None:                       # some exporters use the opposite direction
            for d in skinners:
                if geo_uid in self.children_of.get(d.props[0], []):
                    skin_uid = d.props[0]
                    break
        out = {}
        if skin_uid is None:
            return out, None
        for cl_uid in self.children_of.get(skin_uid, []):
            cl = self.uid2any.get(cl_uid)
            if cl is None or cl.name != "Deformer":
                continue
            bone = None
            for ch in self.children_of.get(cl_uid, []):
                m = self.uid2model.get(ch)
                if m is not None and m.props[2] in ("LimbNode", "Limb"):
                    bone = clean_name(m.props[1])
            if bone is None:
                continue
            idxs = cl.child("Indexes")
            ws = cl.child("Weights")
            if idxs is None or ws is None:
                continue
            d = out.setdefault(bone, {})
            for vi, w in zip(idxs.props[0], ws.props[0]):
                if w > 0:
                    d[vi] = d.get(vi, 0.0) + w
        return out, skin_uid
