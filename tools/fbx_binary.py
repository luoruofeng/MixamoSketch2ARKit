"""Minimal pure-python reader for binary FBX 7.x files.

Only what the pipeline needs: node tree, typed properties, flat arrays.

Two size conventions exist in the wild and both are handled:

  * node records        - 32-bit for version < 7500, 64-bit from 7500 on
  * array properties    - the spec says 64-bit from 7500 on, but the Autodesk
                          SDK (and Apple's own biped_robot.fbx, version 7500)
                          still write 32-bit array headers.  Every layout is
                          tried against the enclosing node boundary and the
                          byte count, so a wrong guess cannot be mistaken for
                          a valid array.
"""
import struct

import zlib

# elementary size of every array property type
_ARRAY_ELEM = {b"f": 4, b"d": 8, b"l": 8, b"i": 4, b"b": 1, b"c": 1}


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name):
        self.name = name
        self.props = []
        self.children = []

    def child(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def all(self, name):
        return [c for c in self.children if c.name == name]

    def __repr__(self):
        return "<Node %s props=%d children=%d>" % (self.name, len(self.props), len(self.children))


def _unpack_array(raw, kind):
    if kind == b"f":
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))
    if kind == b"d":
        return list(struct.unpack("<%dd" % (len(raw) // 8), raw))
    if kind == b"l":
        return list(struct.unpack("<%dq" % (len(raw) // 8), raw))
    if kind == b"i":
        return list(struct.unpack("<%di" % (len(raw) // 4), raw))
    if kind == b"b":
        return [bool(b) for b in raw]
    return bytes(raw)                      # 'c' - raw bytes


def _read_array(buf, off, wide, end, kind):
    """Decode one array property; returns (values, next offset)."""
    elem = _ARRAY_ELEM[kind]
    layouts = [(12, "<III"), (20, "<QIQ")] if wide else [(12, "<III")]
    for header_len, fmt in layouts:
        if off + header_len > end:
            continue
        count, encoding, byte_count = struct.unpack_from(fmt, buf, off)
        if encoding not in (0, 1):
            continue
        start = off + header_len
        stop = start + byte_count
        if stop > end:
            continue
        raw = buf[start:stop]
        if encoding == 1:
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                continue
        if len(raw) != count * elem:
            continue
        return _unpack_array(raw, kind), stop
    raise ValueError("cannot decode %s array at offset %d" % (kind, off - 1))


def _read_prop(buf, off, wide, end):
    t = buf[off:off + 1]
    off += 1
    if t == b"Y":
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if t == b"C":
        return bool(buf[off]), off + 1
    if t == b"I":
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if t == b"F":
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if t == b"D":
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if t == b"L":
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if t in _ARRAY_ELEM:
        return _read_array(buf, off, wide, end, t)
    if t in (b"S", b"R"):
        n = struct.unpack_from("<I", buf, off)[0]
        off += 4
        val = buf[off:off + n]
        off += n
        return (val.decode("utf-8", "replace") if t == b"S" else bytes(val)), off
    raise ValueError("bad property type %r at %d" % (t, off - 1))


def _read_node(buf, off, wide):
    if wide:
        end, nprops, _plen = struct.unpack_from("<QQQ", buf, off)
        off += 24
    else:
        end, nprops, _plen = struct.unpack_from("<III", buf, off)
        off += 12
    nlen = buf[off]
    off += 1
    if end == 0:                            # null record = end of a child list
        return None, off
    name = buf[off:off + nlen].decode("ascii", "replace")
    off += nlen
    node = Node(name)
    for _ in range(nprops):
        value, off = _read_prop(buf, off, wide, end)
        node.props.append(value)
    while off < end:
        child, off = _read_node(buf, off, wide)
        if child is None:
            break
        node.children.append(child)
    return node, end


class FBX:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            self.buf = f.read()
        if not self.buf.startswith(b"Kaydara FBX Binary"):
            raise ValueError("%s: not a binary FBX (ASCII FBX is not supported, "
                             "re-export as binary)" % self.path)
        self.version = struct.unpack_from("<I", self.buf, 23)[0]
        self.wide = self.version >= 7500
        off = 27
        self.root = Node("ROOT")
        while off < len(self.buf):
            node, off = _read_node(self.buf, off, self.wide)
            if node is None:
                break
            self.root.children.append(node)
        for required in ("Objects", "Connections"):
            if self.top(required) is None:
                raise ValueError("%s: no %s section - not an FBX scene" % (self.path, required))

    def top(self, name):
        return self.root.child(name)


def name_of(raw):
    """FBX object names come as 'Name\\x00\\x01Class'"""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return raw.split("\x00")[0]
