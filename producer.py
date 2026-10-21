"""Demo producer: publishes time-varying 3D vector fields over ZeroMQ (see fieldstream.py).

    python producer.py                          # wandering tornado, 32^3, 10 fps, tcp://*:5556
    python producer.py --scenario turbulence --n 40 --fps 5 --dtype int16
    python producer.py --scenario cycle         # switch scenario every 20 s
"""
import argparse
import itertools
import time

import numpy as np

from fieldstream import FieldPublisher


def grid(n):
    c = np.linspace(-1, 1, n, dtype=np.float32)
    Z, Y, X = np.meshgrid(c, c, c, indexing="ij")   # shape (nz, ny, nx), x varies fastest
    return X, Y, Z


def tornado(X, Y, Z, t, _rng):
    h = (Z + 1) / 2
    # the core wanders and leans with height
    cx = 0.3 * np.cos(0.4 * t) + 0.15 * h * np.cos(0.3 * t)
    cy = 0.3 * np.sin(0.55 * t) + 0.15 * h * np.sin(0.3 * t)
    x, y = X - cx, Y - cy
    r = np.hypot(x, y) + 1e-6
    rc = 0.16 + 0.12 * h
    s = 1 + 0.3 * np.sin(1.1 * t)
    vt = 0.35 * s * (1 - np.exp(-r * r / (rc * rc))) / r * (0.7 + 0.3 * h)
    vr = -0.9 * r * np.exp(-h * 6) + 0.7 * r * np.exp(-(1 - h) * 6)
    core = np.exp(-r * r / (rc * rc * 2.5))
    return np.stack([(-y * vt + x * vr) / r, (x * vt + y * vr) / r,
                     1.5 * s * core * (0.4 + 0.6 * h) - 0.12 * (1 - core)], axis=-1)


def turbulence(X, Y, Z, t, rng):
    if not hasattr(turbulence, "modes"):
        km = 2 + rng.random(10) * 5
        d = rng.normal(size=(10, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        k = d * km[:, None]
        a = rng.normal(size=(10, 3)); a *= (0.55 / km * (0.6 + 0.8 * rng.random(10)) / np.linalg.norm(a, axis=1))[:, None]
        turbulence.modes = (k, np.cross(k, a), rng.random(10) * 2 * np.pi, 0.5 + rng.random(10) * 1.5)
    k, c, ph, om = turbulence.modes
    wind = np.array([0.8, 0, 0], np.float32)
    P = np.stack([X, Y, Z], axis=-1) - wind * t * 0.5          # eddies drift with the mean wind
    w = np.cos(P @ k.T + ph + om * t)                          # (nz, ny, nx, modes)
    return wind + w @ c                                        # divergence-free + mean flow


def charges(X, Y, Z, t, _rng):
    v = np.zeros(X.shape + (3,), np.float32)
    for i in range(4):
        a = 0.5 * t + i * np.pi / 2
        p = np.array([0.5 * np.cos(a), 0.5 * np.sin(a), 0.3 * np.sin(0.7 * t + i)], np.float32)
        q = (1 if i % 2 == 0 else -1) * (1 + 0.4 * np.sin(t + i))
        d = np.stack([X, Y, Z], axis=-1) - p
        r2 = (d * d).sum(-1, keepdims=True) + 0.004
        v += q * d / (r2 * np.sqrt(r2))
    return v


def heat(X, Y, Z, t, _rng):
    width = .08 + .08 * (1 + np.sin(t * .5))
    return np.exp(-((X-.25*np.sin(t*.4))**2 + Y*Y + Z*Z) / width)


SCENARIOS = {
    'heat': (heat, 'Live: heat pulse', 'A moving Gaussian heat pulse with changing width (illustrative scalar field).', 'random'),
    "tornado": (tornado, "Live: wandering tornado", "Tornado whose core wanders, leans and pulses in strength.", "ring"),
    "turbulence": (turbulence, "Live: evolving turbulence", "Mean wind with drifting, evolving divergence-free eddies.", "inlet"),
    "charges": (charges, "Live: orbiting charges", "Two sources and two sinks orbiting the z axis with pulsing strength.", "random"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=[*SCENARIOS, "cycle"], default="tornado")
    ap.add_argument("--n", type=int, default=32, help="grid points per axis")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--dtype", choices=["float32", "int16"], default="float32")
    ap.add_argument("--endpoint", default="tcp://*:5556", help="ZeroMQ PUB endpoint to bind")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation time per wall-clock second")
    args = ap.parse_args()

    pub = FieldPublisher(args.endpoint, dtype=args.dtype)
    X, Y, Z = grid(args.n)
    rng = np.random.default_rng(1)
    names = list(SCENARIOS)
    print(f"publishing {args.scenario} {args.n}^3 {args.dtype} at {args.fps} fps on {args.endpoint}  (Ctrl+C to stop)")
    t0 = time.perf_counter()
    for frame in itertools.count():
        now = time.perf_counter() - t0
        key = names[int(now // 20) % len(names)] if args.scenario == "cycle" else args.scenario
        fn, name, desc, seeding = SCENARIOS[key]
        v = fn(X, Y, Z, now * args.speed, rng).astype(np.float32)
        pub.send(v, name=name, desc=desc, seeding=seeding, timestamp=now)
        if frame % 50 == 0:
            print(f"frame {frame}  t={now:6.1f}s  {key}")
        time.sleep(max(0.0, (frame + 1) / args.fps - (time.perf_counter() - t0)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
