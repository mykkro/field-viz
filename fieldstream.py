"""Binary frame format + ZeroMQ publisher for streaming 3D vector fields to field-viz.

Frame layout (little-endian), 40-byte header followed by the payload:

    offset  type     field
    0       4s       magic b"VF01"
    4       u32      seq          producer frame counter
    8       u32      nx
    12      u32      ny
    16      u32      nz
    20      u8       dtype        0 = float32, 1 = int16 quantized (value = q * scale)
    21      3x       padding
    24      f32      scale        dequantization factor (1.0 for float32)
    28      u32      reserved
    32      f64      timestamp    seconds (unix time or simulation time)
    40      ...      nx*ny*nz*3 values: (vx,vy,vz) per point, x fastest, then y, then z
                     == a C-ordered numpy array of shape (nz, ny, nx, 3)

ZeroMQ message = two parts: [meta JSON (utf-8), frame bytes].
meta: {"name": str, "desc": str, "seeding": "random"|"inlet"|"ring",
       "origin": [x0,y0,z0], "spacing": [dx,dy,dz]}   (origin/spacing optional; resent with every frame)

Usage from your own code:

    from fieldstream import FieldPublisher
    pub = FieldPublisher("tcp://*:5556")
    while True:
        v = my_solver_step()                  # numpy array (nz, ny, nx, 3)
        pub.send(v, name="My simulation")
"""
import json
import struct
import time

import numpy as np

MAGIC = b"VF01"
HEADER = struct.Struct("<4sIIIIB3xfId")   # 40 bytes
DTYPES = {"float32": 0, "int16": 1}


def encode_frame(v, seq=0, dtype="float32", timestamp=None):
    """v: array of shape (nz, ny, nx, 3). Returns header + payload bytes."""
    v = np.asarray(v)
    if v.ndim != 4 or v.shape[3] != 3:
        raise ValueError(f"expected shape (nz, ny, nx, 3), got {v.shape}")
    nz, ny, nx, _ = v.shape
    if dtype == "int16":
        m = float(np.abs(v).max()) or 1.0
        scale = m / 32767.0
        payload = np.round(v / scale).astype("<i2").tobytes()
    elif dtype == "float32":
        scale = 1.0
        payload = np.ascontiguousarray(v, dtype="<f4").tobytes()
    else:
        raise ValueError(f"dtype must be one of {list(DTYPES)}")
    ts = time.time() if timestamp is None else timestamp
    return HEADER.pack(MAGIC, seq & 0xFFFFFFFF, nx, ny, nz, DTYPES[dtype], scale, 0, ts) + payload


def decode_frame(buf):
    """Inverse of encode_frame. Returns (header dict, float32 array of shape (nz, ny, nx, 3))."""
    magic, seq, nx, ny, nz, code, scale, _, ts = HEADER.unpack_from(buf)
    if magic != MAGIC:
        raise ValueError("bad magic")
    n = nx * ny * nz * 3
    if code == 1:
        v = np.frombuffer(buf, "<i2", n, HEADER.size).astype(np.float32) * scale
    else:
        v = np.frombuffer(buf, "<f4", n, HEADER.size).astype(np.float32)
    return {"seq": seq, "dims": (nx, ny, nz), "dtype": code, "time": ts}, v.reshape(nz, ny, nx, 3)


class FieldPublisher:
    """ZeroMQ PUB socket sending [meta, frame] messages. Binds by default (relay connects)."""

    def __init__(self, endpoint="tcp://*:5556", bind=True, dtype="float32"):
        import zmq
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 2)       # never queue more than a couple of frames
        (self.sock.bind if bind else self.sock.connect)(endpoint)
        self.dtype = dtype
        self.seq = 0

    def send(self, v, name="Live field", desc="", seeding="random", timestamp=None, origin=None, spacing=None):
        """origin/spacing: physical grid geometry (x, y, z); default is the cube [-1,1]^3."""
        meta = {"name": name, "desc": desc, "seeding": seeding}
        if origin is not None:
            meta["origin"] = [float(c) for c in origin]
        if spacing is not None:
            meta["spacing"] = [float(c) for c in spacing]
        meta = json.dumps(meta).encode()
        self.sock.send_multipart([meta, encode_frame(v, self.seq, self.dtype, timestamp)])
        self.seq += 1
