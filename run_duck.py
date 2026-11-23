#!/usr/bin/env python3
"""Запуск Momduck в MuJoCo с обученной ONNX-политикой. Без duck-lab.

Интерактивно (окно MuJoCo viewer):
    python run_duck.py
    python run_duck.py --vx 0.25

Headless, записать видео и кадры:
    MUJOCO_GL=egl python run_duck.py --vx 0.15 --seconds 20 \
        --out media/walk_015.mp4 --stills media

Клавиши в интерактивном режиме (вводить в ТЕРМИНАЛ, не в окно):
    стрелки вверх/вниз  vx  +/- 0.05 м/с
    стрелки влево/вправо wz  +/- 0.1 рад/с
    пробел             стоп (vx=vy=wz=0)
    q                  выход
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import select
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / "models" / "scene_walk.xml"
DEFAULT_POLICY = ROOT / "policies" / "momduck_walk.onnx"

# Разгон/рендер на headless-машинах (Linux сервер без X): программный EGL.
_HEADLESS = any(a in sys.argv for a in ("--out", "--stills", "--headless"))
if _HEADLESS and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402  (после настройки MUJOCO_GL)
import onnxruntime as ort  # noqa: E402

# ----------------------------------------------------------------- контракт
# momduck-61-v2: 61 наблюдение / 14 действий, 50 Гц.
# obs = [gyro(3), projected_gravity(3), joint_pos_rel(14), joint_vel(14),
#        last_action(14), twist_cmd(3), head_cmd(4), body_cmd(6)]
# target = DEFAULT_POSE + action  (scale 1.0, радианы).
PHYSICS_DT = 0.005
DECIMATION = 4
CTRL_DT = PHYSICS_DT * DECIMATION  # 0.02 с = 50 Гц
OBS_DIM = 61

JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

# Momduck: цели по yaw бедра ограничены +/-10 градусов ДО записи в last_action.
HIP_YAW_IDS = (0, 9)
HIP_YAW_LIMIT = math.radians(10.0)
ACTION_CLIP = 4.0
GYRO_SENSOR = "angular-velocity"
TRUNK_BODY = "trunk_base"


def quat_rotate_inverse(q, v):
    """Повернуть v обратным кватернионом [w, x, y, z]."""
    w, x, y, z = q
    t0 = (y * v[2] - z * v[1]) * 2
    t1 = (z * v[0] - x * v[2]) * 2
    t2 = (x * v[1] - y * v[0]) * 2
    return np.array((v[0] - w * t0 + (y * t2 - z * t1),
                     v[1] - w * t1 + (z * t0 - x * t2),
                     v[2] - w * t2 + (x * t1 - y * t0)), dtype=np.float32)


class Momduck:
    yaw_bound = True

    def __init__(self, scene: Path, policy: Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.session = ort.InferenceSession(str(policy))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        m = self.model
        self.key_stand = m.key("STAND").id
        self.trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY)
        gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
        self.gyro_adr = m.sensor_adr[gyro_id]
        # Позиции/скорости суставов — по передаточным связям актуаторов.
        self.qpos_adr = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.qvel_adr = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        assert m.nu == len(DEFAULT_POSE), f"ожидалось 14 актуаторов, в модели {m.nu}"
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_stand)
        self.data.ctrl[:] = DEFAULT_POSE
        self.last_action = np.zeros(14, dtype=np.float32)
        self.prev_joint_vel = np.zeros(14, dtype=np.float32)
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)

    def obs(self, twist_cmd):
        d = self.data
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3].astype(np.float32)
        gravity = quat_rotate_inverse(d.xquat[self.trunk_id], np.array([0.0, 0.0, -1.0]))
        joint_pos = (d.qpos[self.qpos_adr] - DEFAULT_POSE).astype(np.float32)
        # Отставание на 1 управляющий шаг, как в обучающей среде
        # (Dynamixel present_velocity — скользящее среднее с задержкой).
        joint_vel = self.prev_joint_vel
        self.prev_joint_vel = d.qvel[self.qvel_adr].astype(np.float32).copy()

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:3] = gyro
        obs[3:6] = gravity
        obs[6:20] = joint_pos
        obs[20:34] = joint_vel
        obs[34:48] = self.last_action
        obs[48:51] = twist_cmd
        # head_cmd [51:55] и body_cmd [55:61] — нули (не используются политикой).
        return obs

    def step(self, twist_cmd):
        obs = self.obs(twist_cmd)
        action = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, -1)})[0]
        action = action.squeeze(0).astype(np.float32)
        if self.yaw_bound:
            yaw_ids = list(HIP_YAW_IDS)
            action[yaw_ids] = np.clip(action[yaw_ids], -HIP_YAW_LIMIT, HIP_YAW_LIMIT)
        self.last_action = action.copy()
        self.data.ctrl[:] = DEFAULT_POSE + action.clip(-ACTION_CLIP, ACTION_CLIP)
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        # sensordata/xquat должны описывать состояние ПОСЛЕ интегрирования:
        # mj_step считает сенсоры до последнего подшага (как в walk_env).
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        return float(self.data.xpos[self.trunk_id, 2])


class TerminalKeys:
    """Одиночные нажатия из терминала (cbreak), стрелки -> up/down/left/right."""

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self):
        import termios
        import tty
        self.enabled = sys.stdin.isatty()
        self.q = queue.Queue()
        if not self.enabled:
            return
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.stop = False
        import threading
        threading.Thread(target=self._reader, daemon=True).start()

    def _read1(self, timeout):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        b = os.read(self.fd, 1)
        return b.decode(errors="ignore") if b else None

    def _reader(self):
        while not self.stop:
            ch = self._read1(0.1)
            if not ch:
                continue
            if ch == "\x1b":
                if self._read1(0.05) == "[":
                    name = self._ARROWS.get(self._read1(0.05) or "")
                    if name:
                        self.q.put(name)
                continue
            self.q.put(ch.lower())

    def get(self):
        keys = []
        while True:
            try:
                keys.append(self.q.get_nowait())
            except queue.Empty:
                return keys

    def close(self):
        if self.enabled:
            import termios
            self.stop = True
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def interactive(duck, vx, vy, wz):
    import mujoco.viewer

    cmd = [vx, vy, wz]

    def report():
        print(f"\rкоманда: vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}   ", end="")

    with mujoco.viewer.launch_passive(duck.model, duck.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = duck.trunk_id
        viewer.cam.distance = 0.8
        viewer.cam.elevation = -20
        keys = TerminalKeys()
        report()
        try:
            while viewer.is_running():
                t0 = time.time()
                for k in keys.get():
                    if k == "up":
                        cmd[0] = min(0.4, cmd[0] + 0.05)
                    elif k == "down":
                        cmd[0] = max(-0.4, cmd[0] - 0.05)
                    elif k == "left":
                        cmd[2] = min(1.0, cmd[2] + 0.1)
                    elif k == "right":
                        cmd[2] = max(-1.0, cmd[2] - 0.1)
                    elif k == " ":
                        cmd[:] = [0.0, 0.0, 0.0]
                    elif k == "q":
                        return
                    report()
                duck.step(np.array(cmd, dtype=np.float32))
                viewer.sync()
                dt = CTRL_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        finally:
            keys.close()


def record(duck, vx, vy, wz, seconds, out, stills, width, height, gif=None, azimuth=135, track=False):
    import imageio.v2 as imageio

    renderer = mujoco.Renderer(duck.model, height, width)
    cam = mujoco.MjvCamera()
    if track:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = duck.trunk_id
    else:
        # Статичная камера: видно реальное продвижение утки по полу.
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.3, 0.0, 0.12]
    cam.distance = 1.2
    cam.elevation = -20
    cam.azimuth = azimuth

    n_steps = int(round(seconds / CTRL_DT))
    cmd = np.array([vx, vy, wz], dtype=np.float32)
    writer = imageio.get_writer(out, fps=25) if out else None
    gif_writer = imageio.get_writer(gif, mode="I", fps=12) if gif else None
    still_times = {int(n_steps * f) for f in (0.15, 0.5, 0.85)} if stills else set()
    start_xy = duck.data.xpos[duck.trunk_id, :2].copy()
    min_z = float("inf")
    print(f"запись {seconds} с, команда vx={vx} vy={vy} wz={wz}")
    for i in range(n_steps):
        z = duck.step(cmd)
        min_z = min(min_z, z)
        if i % 2 == 0:  # 50 Гц симуляция -> 25 fps видео
            renderer.update_scene(duck.data, camera=cam)
            frame = renderer.render()
            if writer:
                writer.append_data(frame)
            if gif_writer and i % 4 == 0:  # 12 fps, уменьшенный кадр
                gif_writer.append_data(frame[::2, ::2])
            if i in still_times and stills:
                path = Path(stills) / f"frame_{i * CTRL_DT:04.1f}s.png"
                imageio.imwrite(path, frame)
                print(f"  кадр {path}")
        if z < 0.07:
            print(f"  ПАДЕНИЕ на {i * CTRL_DT:.1f} с (trunk_z={z:.3f})")
            break
    if writer:
        writer.close()
    if gif_writer:
        gif_writer.close()
    dist = float(np.linalg.norm(duck.data.xpos[duck.trunk_id, :2] - start_xy))
    t = duck.step_count * CTRL_DT
    print(f"итог: {t:.1f} с, путь {dist:.2f} м, min trunk_z {min_z:.3f} м")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--vx", type=float, default=0.15, help="команда вперёд, м/с")
    ap.add_argument("--vy", type=float, default=0.0, help="команда вбок, м/с")
    ap.add_argument("--wz", type=float, default=0.0, help="команда поворота, рад/с")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=None, help="mp4 для записи (headless)")
    ap.add_argument("--gif", type=Path, default=None, help="gif для записи (headless, 12 fps)")
    ap.add_argument("--azimuth", type=float, default=135.0,
                    help="азимут камеры записи, градусы (135 = спереди-справа)")
    ap.add_argument("--track", action="store_true", help="камера следит за уткой")
    ap.add_argument("--no-yaw-bound", action="store_true",
                    help="не ограничивать hip yaw ±10° (для оригинальной политики Pollen)")
    ap.add_argument("--stills", type=Path, default=None, help="каталог для PNG-кадров")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    duck = Momduck(args.scene, args.policy)
    duck.yaw_bound = not args.no_yaw_bound
    if args.out or args.stills or args.gif:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.gif:
            args.gif.parent.mkdir(parents=True, exist_ok=True)
        if args.stills:
            args.stills.mkdir(parents=True, exist_ok=True)
        record(duck, args.vx, args.vy, args.wz, args.seconds,
               args.out, args.stills, args.width, args.height,
               gif=args.gif, azimuth=args.azimuth, track=args.track)
        # EGL на этой машине падает в деструкторе контекста при выходе —
        # завершаем процесс до него, все данные уже на диске.
        sys.stdout.flush()
        os._exit(0)
    else:
        interactive(duck, args.vx, args.vy, args.wz)


if __name__ == "__main__":
    main()
