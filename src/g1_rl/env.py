"""Position-residual control of the Menagerie G1 in native MuJoCo."""
from dataclasses import dataclass
from pathlib import Path
import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


@dataclass
class EnvConfig:
    task: str = "walk"
    model_path: str = "assets/unitree_g1/scene.xml"
    timestep: float = 0.002
    frame_skip: int = 10
    episode_seconds: float = 12.0
    action_scale: float = 0.5
    forward_speed: float = 0.5
    lateral_speed: float = 0.0
    yaw_rate: float = 0.0
    jump_height: float = 0.12
    jump_period: float = 2.0
    reset_noise: float = 0.01


class G1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: EnvConfig | None = None, render_mode=None):
        super().__init__()
        self.cfg = config or EnvConfig()
        c = self.cfg
        if c.task not in ("stand", "walk", "jump"):
            raise ValueError("task must be stand, walk or jump")
        if (c.timestep <= 0 or c.frame_skip < 1 or c.episode_seconds <= 0
                or c.action_scale <= 0 or c.jump_height <= 0 or c.reset_noise < 0):
            raise ValueError("Invalid environment timing, scale or height")
        self.flight_time = 2 * math.sqrt(2 * c.jump_height / 9.81)
        if c.jump_period <= self.flight_time + 0.6:
            raise ValueError("jump_period must leave time for crouching and landing")
        path = Path(c.model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run: uv run scripts/download_model.py")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render mode")
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.model.opt.timestep = c.timestep
        self.data = mujoco.MjData(self.model)
        self.dt = c.timestep * c.frame_skip
        self.metadata = dict(self.metadata, render_fps=round(1 / self.dt))
        self.max_steps = math.ceil(c.episode_seconds / self.dt)
        self.base = self.model.body("pelvis").id
        self.floor = self.model.geom("floor").id
        self.feet = [self.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        names = [f"{side}_{joint}_joint" for side in ("left", "right")
                 for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
        self.act_ids = np.array([self.model.actuator(n).id for n in names])
        joint_ids = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.stand_key = self.model.key("stand").id
        self.home = self.model.key_qpos[self.stand_key].copy()
        # Slightly bent knees create a useful balanced reset pose.
        for side in ("left", "right"):
            for joint, value in (("hip_pitch", -0.15), ("knee", 0.3), ("ankle_pitch", -0.15)):
                self.home[self.model.joint(f"{side}_{joint}_joint").qposadr[0]] = value
        self.home[2] = 0.775
        self.nominal_ctrl = self.home[self.qadr].copy()
        self.action_space = spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)
        # Local velocities, gravity, all joints, command, phase, contacts, previous action, height.
        size = 3 + 3 + 3 + 2 * self.model.nu + 3 + 2 + 2 + 12 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
        self.previous_action = np.zeros(12)
        self.command = np.array([c.forward_speed, c.lateral_speed, c.yaw_rate]) if c.task == "walk" else np.zeros(3)
        self.renderer = None
        self.viewer = None
        self.steps = 0

    def _contacts(self):
        contacts = np.zeros(2)
        bad = False
        for contact in self.data.contact:
            if self.floor not in (contact.geom1, contact.geom2):
                continue
            geom = contact.geom2 if contact.geom1 == self.floor else contact.geom1
            body = self.model.geom_bodyid[geom]
            if body in self.feet:
                contacts[self.feet.index(body)] = 1
            else:
                bad = True
        return contacts, bad

    def _kinematics(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 1)
        return velocity[3:], velocity[:3], rotation.T @ np.array([0., 0., -1.])

    def _phase(self):
        period = self.cfg.jump_period if self.cfg.task == "jump" else 0.8
        return (self.steps * self.dt % period) / period

    def _observation(self):
        lin, ang, gravity = self._kinematics()
        angle = 2 * math.pi * self._phase()
        contacts, _ = self._contacts()
        return np.concatenate([lin, ang, gravity, self.data.qpos[self.qadr] - self.nominal_ctrl,
                               self.data.qvel[self.vadr] * 0.1, self.command,
                               [math.sin(angle), math.cos(angle)], contacts,
                               self.previous_action, [self.data.qpos[2]]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home
        self.data.qpos[self.qadr] += self.np_random.uniform(-self.cfg.reset_noise, self.cfg.reset_noise, self.model.nu)
        self.data.ctrl[:] = self.nominal_ctrl
        self.previous_action[:] = 0
        self.steps = 0
        self.air_steps = 0
        self.qualified_flight = False
        self.landed = False
        self.previous_cycle = 0
        self.peak_height = float(self.home[2])
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {}

    def _jump_target(self):
        t = self.steps * self.dt % self.cfg.jump_period
        launch = 0.6
        if t < launch:
            # Smooth preparatory crouch, returning to nominal height at takeoff.
            return self.home[2] - 0.08 * math.sin(math.pi * t / launch) ** 2, False
        flight = t - launch
        if flight < self.flight_time:
            vz = math.sqrt(2 * 9.81 * self.cfg.jump_height)
            return self.home[2] + vz * flight - 0.5 * 9.81 * flight**2, True
        return self.home[2], False

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite vector of length 12")
        action = np.clip(action, -1, 1)
        ctrl = self.nominal_ctrl.copy()
        ctrl[self.act_ids] += self.cfg.action_scale * action
        self.data.ctrl[:] = np.clip(ctrl, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        # Refresh derived body transforms/contact data for the final integration state.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        lin, ang, gravity = self._kinematics()
        contacts, bad_contact = self._contacts()
        height = float(self.data.qpos[2])
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        terminated = bool(not finite or height < 0.45 or -gravity[2] < 0.45 or bad_contact)
        truncated = self.steps >= self.max_steps
        terms = {
            "upright": 1.0 * np.clip(-gravity[2], 0, 1),
            "velocity": 2.0 * np.exp(-np.sum((lin[:2] - self.command[:2]) ** 2) / 0.25),
            "yaw": 0.5 * np.exp(-(ang[2] - self.command[2]) ** 2 / 0.25),
            "tilt_rate": -0.05 * np.sum(ang[:2] ** 2),
            "effort": -1e-5 * np.sum(self.data.actuator_force ** 2),
            "smooth": -0.05 * np.sum((action - self.previous_action) ** 2),
            "posture": -0.1 * np.sum((self.data.qpos[self.qadr] - self.nominal_ctrl) ** 2),
        }
        target_height = self.home[2]
        if self.cfg.task == "jump":
            target_height, flight = self._jump_target()
            cycle = int(self.steps * self.dt / self.cfg.jump_period)
            if cycle != self.previous_cycle:
                self.air_steps, self.landed = 0, False
                self.qualified_flight = False
                self.previous_cycle = cycle
            airborne = contacts.sum() == 0
            # Count actual flight only after rising above the reference standing height.
            if airborne and height > self.home[2] + 0.025:
                self.air_steps += 1
                self.qualified_flight |= self.air_steps >= 3
            else:
                self.air_steps = 0
            terms["flight"] = 1.0 * float(airborne and flight and height > self.home[2] + 0.025)
            terms["support"] = 0.4 * float(not flight) * float(contacts.mean())
            landed = (not self.landed and self.qualified_flight and contacts.sum() == 2 and not terminated)
            terms["landing"] = 3.0 * float(landed) * np.exp(-lin[2] ** 2)
            self.landed |= landed
        elif self.cfg.task == "walk":
            phase = self._phase()
            desired = np.array([phase < 0.5, phase >= 0.5], dtype=float)
            terms["gait"] = 0.3 * (1 - np.abs(contacts - desired).mean())
            terms["vertical_velocity"] = -0.1 * lin[2] ** 2
        else:
            terms["support"] = 0.4 * contacts.mean()
        terms["height"] = np.exp(-((height - target_height) / 0.08) ** 2)
        self.previous_action[:] = action
        self.peak_height = max(self.peak_height, height)
        reward = float(sum(terms.values()) * self.dt - 3.0 * terminated)
        obs = self._observation()
        if not finite or not np.isfinite(obs).all() or not np.isfinite(reward):
            obs = np.nan_to_num(obs, nan=0., posinf=0., neginf=0.)
            reward, terminated = -3.0, True
        info = {"reward_terms": {k: float(v) for k, v in terms.items()}, "height": height,
                "target_height": float(target_height), "forward_velocity": float(lin[0]),
                "contacts": contacts.tolist(), "peak_height": self.peak_height,
                "landed": self.landed, "fallen": terminated,
                "position_xy": self.data.qpos[:2].tolist()}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.model, self.data)
                self.viewer.cam.distance = 3
            self.viewer.cam.lookat[:] = self.data.xpos[self.base]
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self.data.xpos[self.base]
            camera.distance, camera.azimuth, camera.elevation = 3, 135, -15
            self.renderer.update_scene(self.data, camera=camera)
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
