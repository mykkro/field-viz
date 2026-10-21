"""Minimal REST backend serving synthetic 3D vector fields (Python stdlib only).

Run:   python server.py [port]        (default port 8000)
Open:  http://localhost:8000/

Endpoints
  GET /api/fields                       -> list of available generators
  GET /api/fields/<id>?n=24&seed=1      -> {"id","name","desc","seeding","dims":[nx,ny,nz],"data":[vx,vy,vz,...]}
      data is flattened x-fastest, then y, then z; the grid spans [-1,1] on every axis.
  GET /api/fields/<id>?...&format=vtk   -> file download; format = json | vtk | vti | npy | csv
Everything else is served as static files from this directory (index.html etc.).
"""
import base64
import json
import math
import random
import struct
import sys
from array import array
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
MAX_N = 64

# ---------------------------------------------------------------------------
# Generators: make(seed) -> f(x, y, z) -> (vx, vy, vz), domain [-1,1]^3, z is up
# ---------------------------------------------------------------------------

def tornado(_seed):
    def f(x, y, z):
        r = math.hypot(x, y) + 1e-6
        h = (z + 1) / 2
        rc = 0.16 + 0.12 * h
        vt = 0.35 * (1 - math.exp(-r * r / (rc * rc))) / r * (0.7 + 0.3 * h)
        vr = -0.9 * r * math.exp(-h * 6) + 0.7 * r * math.exp(-(1 - h) * 6)
        core = math.exp(-r * r / (rc * rc * 2.5))
        return ((-y * vt + x * vr) / r, (x * vt + y * vr) / r,
                1.5 * core * (0.4 + 0.6 * h) - 0.12 * (1 - core))
    return f


SPHERE_R = 0.35

def sphere_flow(_seed):
    a3 = SPHERE_R ** 3
    def f(x, y, z):
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        if r < SPHERE_R:
            return (0.0, 0.0, 0.0)
        r3 = r2 * r
        k = 1.5 * a3 * x / (r3 * r2)
        return (1 + a3 / (2 * r3) - k * x, -k * y, -k * z)
    return f


def abc_flow(_seed):
    A, B, C, k = 1.0, math.sqrt(2 / 3), math.sqrt(1 / 3), math.pi
    def f(x, y, z):
        return (A * math.sin(k * z) + C * math.cos(k * y),
                B * math.sin(k * x) + A * math.cos(k * z),
                C * math.sin(k * y) + B * math.cos(k * x))
    return f


def _rand_dir(rng):
    z = rng.uniform(-1, 1)
    a = rng.uniform(0, 2 * math.pi)
    r = math.sqrt(1 - z * z)
    return (r * math.cos(a), r * math.sin(a), z)


def curl_noise(seed):
    rng = random.Random(seed)
    modes = []
    for _ in range(10):
        km = 2 + rng.random() * 5
        k = tuple(c * km for c in _rand_dir(rng))
        amp = 0.55 / km * (0.6 + 0.8 * rng.random())
        a = tuple(c * amp for c in _rand_dir(rng))
        # curl(a sin(k.p + phi)) = (k x a) cos(k.p + phi)  -> divergence free
        c = (k[1] * a[2] - k[2] * a[1], k[2] * a[0] - k[0] * a[2], k[0] * a[1] - k[1] * a[0])
        modes.append((k, c, rng.random() * 2 * math.pi))
    def f(x, y, z):
        vx, vy, vz = 0.8, 0.0, 0.0
        for k, c, ph in modes:
            w = math.cos(k[0] * x + k[1] * y + k[2] * z + ph)
            vx += c[0] * w; vy += c[1] * w; vz += c[2] * w
        return (vx, vy, vz)
    return f


def lorenz(_seed):
    def f(x, y, z):
        X, Y, Z = x * 22, y * 28, (z + 1) * 25
        return (10 * (Y - X) / 22, (X * (28 - Z) - Y) / 28, (X * Y - 8 / 3 * Z) / 25)
    return f


CHARGES = [(0.5, 0.3, 0, 1), (-0.5, 0.3, 0, -1), (0, -0.45, 0.4, 0.7), (0, -0.45, -0.4, -0.7)]

def charges(_seed):
    def f(x, y, z):
        vx = vy = vz = 0.0
        for cx, cy, cz, q in CHARGES:
            dx, dy, dz = x - cx, y - cy, z - cz
            r2 = dx * dx + dy * dy + dz * dz + 0.004
            s = q / (r2 * math.sqrt(r2))
            vx += s * dx; vy += s * dy; vz += s * dz
        return (vx, vy, vz)
    return f


def convection(_seed):
    P = math.pi
    def f(x, y, z):
        zz = (z + 1) / 2
        return (-math.sin(1.5 * P * x) * math.cos(P * zz) * 0.9,
                0.25 * math.sin(P * y) * math.sin(P * zz) + 0.1 * math.cos(1.5 * P * x),
                0.9 * math.cos(1.5 * P * x) * math.sin(P * zz))
    return f


# id -> (display name, description, default seeding, factory). Names match the front end.
GENERATORS = {
    "tornado": ("Tornado", "Swirling vortex with central updraft, ground inflow and outflow aloft.", "ring", tornado),
    "sphere": ("Flow past sphere", "Potential flow along +X around a solid sphere.", "inlet", sphere_flow),
    "abc": ("ABC flow", "Arnold-Beltrami-Childress flow with chaotic streamlines.", "random", abc_flow),
    "turbulence": ("Turbulent wind (curl noise)", "Mean wind plus divergence-free random eddies.", "inlet", curl_noise),
    "lorenz": ("Lorenz attractor", "Lorenz system rescaled into the box.", "random", lorenz),
    "charges": ("Charges (sources & sinks)", "Coulomb-like field of two sources and two sinks.", "charges", charges),
    "convection": ("Convection rolls", "Rayleigh-Benard style counter-rotating convection cells.", "random", convection),
}


def _float32_bytes(values, big_endian=False):
    arr = array("f", values)
    if (sys.byteorder == "little") == big_endian:
        arr.byteswap()
    return arr.tobytes()


def to_vtk(field):
    """VTK legacy STRUCTURED_POINTS, binary (big-endian float32)."""
    nx, ny, nz = field["dims"]
    h = " ".join(f"{2 / (d - 1):.9g}" for d in field["dims"])
    head = (f"# vtk DataFile Version 3.0\n{field['name']}\nBINARY\nDATASET STRUCTURED_POINTS\n"
            f"DIMENSIONS {nx} {ny} {nz}\nORIGIN -1 -1 -1\nSPACING {h}\n"
            f"POINT_DATA {nx * ny * nz}\nVECTORS velocity float\n")
    return head.encode() + _float32_bytes(field["data"], big_endian=True) + b"\n"


def to_vti(field):
    """VTK XML ImageData with inline base64 binary (UInt32 header + little-endian float32)."""
    nx, ny, nz = field["dims"]
    raw = _float32_bytes(field["data"])
    payload = base64.b64encode(struct.pack("<I", len(raw)) + raw).decode()
    ext = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    h = " ".join(f"{2 / (d - 1):.9g}" for d in field["dims"])
    return (f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n'
            f'  <ImageData WholeExtent="{ext}" Origin="-1 -1 -1" Spacing="{h}">\n    <Piece Extent="{ext}">\n'
            f'      <PointData Vectors="velocity">\n'
            f'        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="binary">\n{payload}\n'
            f'        </DataArray>\n      </PointData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n').encode()


def to_npy(field):
    """NumPy .npy, float32, shape (nz, ny, nx, 3) -> a[k, j, i, :]."""
    nx, ny, nz = field["dims"]
    header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({nz}, {ny}, {nx}, 3), }}"
    total = -(-(10 + len(header) + 1) // 64) * 64
    header = header.ljust(total - 10 - 1) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + _float32_bytes(field["data"])


def to_csv(field):
    nx, ny, nz = field["dims"]
    d = field["data"]
    lines = ["x,y,z,vx,vy,vz"]
    p = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                lines.append(f"{-1 + 2 * i / (nx - 1):.6f},{-1 + 2 * j / (ny - 1):.6f},{-1 + 2 * k / (nz - 1):.6f},"
                             f"{d[p]:.7g},{d[p + 1]:.7g},{d[p + 2]:.7g}")
                p += 3
    return ("\n".join(lines) + "\n").encode()


# format -> (writer, content type, file extension); json is handled separately
FORMATS = {
    "vtk": (to_vtk, "application/octet-stream", "vtk"),
    "vti": (to_vti, "application/xml", "vti"),
    "npy": (to_npy, "application/octet-stream", "npy"),
    "csv": (to_csv, "text/csv", "csv"),
}


SCALAR_GENERATORS = {
    'scalar_gaussian': ('Gaussian hot spot', '', lambda x,y,z: math.exp(-5*(x*x+y*y+z*z))),
    'scalar_two': ('Two hot spots', '', lambda x,y,z: math.exp(-12*((x-.4)**2+y*y+z*z))+.8*math.exp(-12*((x+.4)**2+y*y+z*z))),
    'scalar_temperature': ('Atmospheric temperature', '°C', lambda x,y,z: 18-6*(z+1)+5*math.exp(-35*(z-.2)**2)+10*math.exp(-9*(x*x+y*y)-3*(z+.5)**2)),
    'scalar_potential': ('Electric dipole potential', 'arb.', lambda x,y,z: 1/math.sqrt((x-.4)**2+y*y+z*z+.02)-1/math.sqrt((x+.4)**2+y*y+z*z+.02)),
    'scalar_charge': ('Signed charge density', 'arb.', lambda x,y,z: math.exp(-15*((x-.4)**2+y*y+z*z))-math.exp(-15*((x+.4)**2+y*y+z*z))),
    'scalar_heat': ('Heat pulse', '', lambda x,y,z: math.exp(-8*(x*x+y*y+z*z))),
    'scalar_gyroid': ('Gyroid', '', lambda x,y,z: math.sin(3*x)*math.cos(3*y)+math.sin(3*y)*math.cos(3*z)+math.sin(3*z)*math.cos(3*x)),
}


def scalar_export(field, fmt):
    nx, ny, nz = field['dims']
    raw = _float32_bytes(field['data'])
    if fmt == 'npy':
        header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({nz}, {ny}, {nx}), }}"
        header = header.ljust(-(-(10 + len(header) + 1) // 64) * 64 - 11) + '\n'
        return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode() + raw
    spacing = ' '.join(f'{s:.12g}' for s in field['spacing'])
    if fmt == 'vtk':
        return (f'# vtk DataFile Version 3.0\nscalar field\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {nx} {ny} {nz}\nORIGIN -1 -1 -1\nSPACING {spacing}\nPOINT_DATA {nx*ny*nz}\nSCALARS scalar float 1\nLOOKUP_TABLE default\n').encode() + _float32_bytes(field['data'], big_endian=True) + b'\n'
    if fmt == 'vti':
        ext = f'0 {nx-1} 0 {ny-1} 0 {nz-1}'
        payload = base64.b64encode(struct.pack('<I', len(raw)) + raw).decode()
        return (f'<?xml version="1.0"?><VTKFile type="ImageData" byte_order="LittleEndian" header_type="UInt32"><ImageData WholeExtent="{ext}" Origin="-1 -1 -1" Spacing="{spacing}"><Piece Extent="{ext}"><PointData Scalars="scalar"><DataArray type="Float32" Name="scalar" NumberOfComponents="1" format="binary">{payload}</DataArray></PointData></Piece></ImageData></VTKFile>').encode()
    lines = ['# field-viz ' + json.dumps({'name': field['name'], 'units': field['units']}), 'x,y,z,value']
    p = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                lines.append(','.join(str(v) for v in (-1+i*field['spacing'][0], -1+j*field['spacing'][1], -1+k*field['spacing'][2], field['data'][p])))
                p += 1
    return ('\n'.join(lines)+'\n').encode()


def build_field(gen_id, n, seed):
    if gen_id in SCALAR_GENERATORS:
        name, units, fn = SCALAR_GENERATORS[gen_id]
        coords = [-1 + 2*i/(n-1) for i in range(n)]
        return {'id': gen_id, 'name': name, 'kind': 'scalar', 'units': units, 'desc': 'Synthetic scalar example.',
                'dims': [n,n,n], 'origin': [-1,-1,-1], 'spacing': [2/(n-1)]*3,
                'data': [fn(x,y,z) for z in coords for y in coords for x in coords]}
    name, desc, seeding, make = GENERATORS[gen_id]
    f = make(seed)
    coords = [-1 + 2 * i / (n - 1) for i in range(n)]
    data = []
    for z in coords:
        for y in coords:
            for x in coords:
                data.extend(round(v, 5) for v in f(x, y, z))
    return {"id": gen_id, "name": name, "desc": desc, "seeding": seeding, "dims": [n, n, n],
            "origin": [-1, -1, -1], "spacing": [2 / (n - 1)] * 3, "data": data}


def list_generators():
    return ([{"id": k, "name": v[0], "desc": v[1], "seeding": v[2], 'kind': 'vector'} for k, v in GENERATORS.items()]
            + [{'id': k, 'name': v[0], 'kind': 'scalar', 'units': v[1], 'desc': 'Synthetic scalar example.'} for k,v in SCALAR_GENERATORS.items()])


def field_response(gen_id, query):
    """Shared by server.py and stream_server.py.
    query: dict of str -> str. Returns (status, content_type, body_bytes, extra_headers)."""
    def err(status, msg):
        return status, "application/json", json.dumps({"error": msg}).encode(), {}
    if gen_id not in GENERATORS and gen_id not in SCALAR_GENERATORS:
        return err(404, "unknown generator")
    try:
        n = int(query.get("n", "24"))
        seed = int(query.get("seed", "1"))
    except ValueError:
        return err(400, "n and seed must be integers")
    if not 2 <= n <= MAX_N:
        return err(400, f"n must be in [2, {MAX_N}]")
    fmt = query.get("format", "json")
    if fmt != "json" and fmt not in FORMATS:
        return err(400, f"format must be one of json, {', '.join(FORMATS)}")
    field = build_field(gen_id, n, seed)
    if fmt == "json":
        return 200, "application/json", json.dumps(field, separators=(",", ":")).encode(), {}
    writer, ctype, ext = FORMATS[fmt]
    body = scalar_export(field, fmt) if field.get('kind') == 'scalar' else writer(field)
    return 200, ctype, body, {"Content-Disposition": f'attachment; filename="{gen_id}_{n}x{n}x{n}.{ext}"'}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_json(self, obj, status=200):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if parts[:2] != ["api", "fields"]:
            return super().do_GET()
        if len(parts) == 2:
            return self.send_json(list_generators())
        if len(parts) == 3:
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            status, ctype, body, headers = field_response(parts[2], query)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            return
        return self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on http://localhost:{port}/  (API: /api/fields)")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
