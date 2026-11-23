#!/usr/bin/env python3
"""Оценка политики Momduck: N сидов x 20 с, пинкоманда, детерминированный ONNX.

    python tools/probe_forward.py --policy policies/momduck_walk.onnx --cmd 0.25 --seeds 8

Протокол: сиды 20..20+N-1, спавн из STAND с yaw ~ U(-pi, pi) (np.random.default_rng(seed)),
команда [cmd, 0, 0] каждый управляющий шаг, head/body-команды нули, чистая физика
(без шума/задержки/DR). Эпизод 20 с; "full" = не упал (trunk_z >= 0.07 всю дорогу).

Метрики: full, продвижение по начальному курсу (net), боковое смещение (lat_off),
RMS боковой скорости (lat_rms), heading RMSE (град), средние hip yaw L/R (град).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_duck import Momduck, DEFAULT_POLICY  # noqa: E402


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_episode(duck, seed, cmd, seconds=20.0):
    import mujoco

    rng = np.random.default_rng(seed)
    duck.reset()
    # Спавн с случайным yaw, как в обучающей среде.
    yaw = rng.uniform(-np.pi, np.pi)
    duck.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(duck.model, duck.data)
    duck.prev_joint_vel[:] = 0.0

    yaw0 = yaw_of(duck.data.qpos[3:7])
    heading = np.array([np.cos(yaw0), np.sin(yaw0)])
    p0 = duck.data.qpos[:2].copy()
    twist = np.array([cmd, 0, 0], dtype=np.float32)
    lat_vs, head_errs, lyaws, ryaws = [], [], [], []
    min_z = float("inf")
    steps = int(round(seconds / 0.02))
    done_at = steps
    for i in range(steps):
        z = duck.step(twist)
        min_z = min(min_z, z)
        q = duck.data.qpos
        quat = q[3:7]
        w, x, y, zq = quat
        v = duck.data.qvel[0:3]
        t = 2 * np.cross(quat[1:], v)
        vb = v - w * t + np.cross(quat[1:], t)
        lat_vs.append(vb[1])
        head_errs.append(np.degrees((yaw_of(quat) - yaw0 + np.pi) % (2 * np.pi) - np.pi))
        # qpos[7] = left_hip_yaw, qpos[16] = right_hip_yaw (порядок актуаторов).
        lyaws.append(np.degrees(q[7]))
        ryaws.append(np.degrees(q[16]))
        if z < 0.07:
            done_at = i + 1
            break
    secs = done_at * 0.02
    disp = duck.data.qpos[:2] - p0
    return dict(
        full=done_at == steps,
        secs=secs,
        net=float(disp @ heading),
        lat_off=float(disp @ np.array([-heading[1], heading[0]])),
        lat_rms=float(np.sqrt(np.mean(np.square(lat_vs)))),
        head_rmse=float(np.sqrt(np.mean(np.square(head_errs)))),
        lyaw=float(np.mean(lyaws)),
        ryaw=float(np.mean(ryaws)),
        min_z=min_z,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--cmd", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=20)
    args = ap.parse_args()

    duck = Momduck(Path(__file__).resolve().parents[1] / "models" / "scene_walk.xml", args.policy)
    results = []
    for seed in range(args.seed0, args.seed0 + args.seeds):
        r = run_episode(duck, seed, args.cmd)
        results.append(r)
        print(f"seed {seed}: full={r['full']} t={r['secs']:.1f}s net={r['net']:+.2f}m "
              f"lat_off={r['lat_off']:+.2f} lat_rms={r['lat_rms']:.3f} "
              f"headRMSE={r['head_rmse']:.1f}deg yawL={r['lyaw']:+.1f} yawR={r['ryaw']:+.1f} "
              f"min_z={r['min_z']:.3f}")
    n = len(results)
    print(f"\nfull {sum(r['full'] for r in results)}/{n}; "
          f"net mean {np.mean([r['net'] for r in results]):+.2f} m; "
          f"lat_rms mean {np.mean([r['lat_rms'] for r in results]):.3f}; "
          f"headRMSE mean {np.mean([r['head_rmse'] for r in results]):.1f} deg; "
          f"yawL {np.mean([r['lyaw'] for r in results]):+.1f} "
          f"yawR {np.mean([r['ryaw'] for r in results]):+.1f}")


if __name__ == "__main__":
    main()
