"""Generate a WindNinja-like wind dataset: VTK legacy STRUCTURED_GRID on terrain-following layers.

    python examples/make_windninja_like.py                 # -> examples/windninja_like.vtk
    python examples/make_windninja_like.py --ascii -o out.vtk --nx 101 --ny 81 --nz 24

Mesh (like WindNinja's 3D output):
  * horizontal: regular DEM cells (x depends only on i, y only on j)
  * vertical:   nz layers following the terrain, geometrically stretched (thin near the
                ground), flattening out to a flat model top above the highest peak
Wind (a plausible, NOT physically solved field):
  * log profile from a 10 m reference speed, zero at the ground
  * speed-up over ridges and slow-down in valleys, decaying with height
  * horizontal deflection along terrain slopes, terrain-following vertical motion (w = v_h . grad z)
Point data: VECTORS "wind" (m/s) and SCALARS "speed" (to exercise readers that skip scalars).
"""
import argparse
from pathlib import Path

import numpy as np


def terrain(X, Y):
    """Hills, a ridge and a valley over a ~3.6 x 2.7 km area (heights in m a.s.l.)."""
    z = 420.0
    z += 260 * np.exp(-(((X - 2300) / 520) ** 2 + ((Y - 1500) / 380) ** 2))           # main hill
    z += 170 * np.exp(-(((X - 1100) / 300) ** 2 + ((Y - 650) / 450) ** 2))            # second hill
    ridge = np.exp(-(((Y - 0.35 * X - 1500) / 220) ** 2)) * np.exp(-((X - 1700) / 1300) ** 2)
    z += 130 * ridge                                                                  # diagonal ridge
    z -= 70 * np.exp(-(((Y - 2300) / 260) ** 2))                                       # valley in the north
    z += 15 * np.sin(X / 170) * np.cos(Y / 230)                                        # small-scale roughness
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(Path(__file__).with_name("windninja_like.vtk")))
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--ny", type=int, default=46)
    ap.add_argument("--nz", type=int, default=18)
    ap.add_argument("--cell", type=float, default=60.0, help="horizontal cell size [m]")
    ap.add_argument("--speed", type=float, default=7.0, help="wind speed at 10 m above ground [m/s]")
    ap.add_argument("--direction", type=float, default=240.0,
                    help="meteorological wind direction [deg], i.e. where the wind comes FROM (240 = from SW)")
    ap.add_argument("--ascii", action="store_true", help="write ASCII instead of BINARY")
    a = ap.parse_args()

    x = np.arange(a.nx) * a.cell
    y = np.arange(a.ny) * a.cell
    X2, Y2 = np.meshgrid(x, y)                    # (ny, nx)
    zg = terrain(X2, Y2)
    ztop = zg.max() + 900.0                       # flat model top

    # terrain-following vertical coordinate, geometric stretching (first layer ~ 4 m)
    r = 1.28
    s = (r ** np.arange(a.nz) - 1) / (r ** (a.nz - 1) - 1)          # 0 .. 1
    Z = zg[None] + (ztop - zg[None]) * s[:, None, None]              # (nz, ny, nx)
    X = np.broadcast_to(X2, Z.shape)
    Y = np.broadcast_to(Y2, Z.shape)
    agl = Z - zg[None]                                               # height above ground

    # log wind profile, direction "from" -> flow vector "to"
    z0 = 0.05
    prof = np.log(np.maximum(agl, z0) / z0) / np.log(10 / z0)
    th = np.radians(a.direction)
    ux, uy = -np.sin(th), -np.cos(th)

    # terrain effects, decaying with height
    gy, gx = np.gradient(zg, a.cell, a.cell)                         # dz/dy, dz/dx  (ny, nx)
    rel = (zg - zg.mean()) / (zg.max() - zg.min())                   # -0.5 .. 0.5
    decay = np.exp(-agl / 350.0)
    speedup = 1 + 0.9 * rel[None] * decay
    U = a.speed * prof * speedup * ux
    V = a.speed * prof * speedup * uy
    # deflect part of the upslope component sideways (flow goes around steep obstacles)
    dot = U * gx[None] + V * gy[None]
    g2 = gx ** 2 + gy ** 2 + 1e-9
    defl = 0.6 * decay * np.minimum(1, 8 * np.sqrt(g2))[None]
    U -= defl * dot * gx[None] / g2[None] * 0.5
    V -= defl * dot * gy[None] / g2[None] * 0.5
    # terrain-following vertical motion, vanishing at the model top
    W = (U * gx[None] + V * gy[None]) * (1 - s[:, None, None]) ** 2
    U[0] = V[0] = W[0] = 0.0                                         # no-slip ground layer

    P = np.stack([X, Y, Z], -1).reshape(-1, 3)                      # x fastest, then y, then z
    vec = np.stack([U, V, W], -1).reshape(-1, 3)
    spd = np.sqrt((vec ** 2).sum(-1))
    n = P.shape[0]

    out = Path(a.out)
    with open(out, "wb") as f:
        fmt = "ASCII" if a.ascii else "BINARY"
        f.write(f"# vtk DataFile Version 3.0\nWindNinja-like synthetic wind ({a.direction:.0f} deg, "
                f"{a.speed:g} m/s @10m)\n{fmt}\nDATASET STRUCTURED_GRID\nDIMENSIONS {a.nx} {a.ny} {a.nz}\n"
                f"POINTS {n} float\n".encode())

        def block(arr):
            if a.ascii:
                np.savetxt(f, arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1), fmt="%.6g")
            else:
                f.write(np.ascontiguousarray(arr, dtype=">f4").tobytes())
                f.write(b"\n")

        block(P)
        f.write(f"POINT_DATA {n}\nSCALARS speed float 1\nLOOKUP_TABLE default\n".encode())
        block(spd)
        f.write(b"VECTORS wind float\n")
        block(vec)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {a.nx}x{a.ny}x{a.nz} points, "
          f"terrain {zg.min():.0f}-{zg.max():.0f} m, top {ztop:.0f} m, max speed {spd.max():.1f} m/s")


if __name__ == "__main__":
    main()
