"""Evaluate saved policies with frozen training normalization."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_rl.env import EnvConfig, G1Env
from g1_rl.device import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, cuda:N or auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--model-path", help="Override asset location when moving a checkpoint")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--gif", type=Path, help="Record the first episode as a 10 fps GIF; headless: MUJOCO_GL=egl")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.gif and args.render:
        parser.error("Choose --gif or --render")
    try:
        device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    config = json.loads((args.checkpoint / "env.json").read_text())
    if args.model_path:
        config["model_path"] = args.model_path
    robot = G1Env(EnvConfig(**config), render_mode="human" if args.render else "rgb_array" if args.gif else None)
    vec = DummyVecEnv([lambda: robot])
    env = VecNormalize.load(args.checkpoint / "vecnormalize.pkl", vec)
    env.training, env.norm_reward = False, False
    env.seed(args.seed)
    model = PPO.load(args.checkpoint / "policy.zip", device=device)
    results = []
    frames = []
    frame_stride = max(1, round(0.1 / robot.dt))
    try:
        obs = env.reset()
        for episode in range(args.episodes):
            total, length, velocities, peak, landings = 0., 0, [], 0., False
            initial_xy = robot.data.qpos[:2].copy()
            single_support = 0
            while True:
                start = time.monotonic()
                if args.gif and episode == 0 and length % frame_stride == 0:
                    from PIL import Image
                    frames.append(Image.fromarray(robot.render()).quantize(colors=128))
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                info = infos[0]
                total += float(rewards[0])
                length += 1
                velocities.append(info["forward_velocity"])
                peak = max(peak, info["peak_height"])
                landings |= info["landed"]
                single_support += sum(info["contacts"]) == 1
                if args.render:
                    time.sleep(max(0., robot.dt - (time.monotonic() - start)))
                if dones[0]:
                    break  # VecEnv has already reset; do not reset again.
            result = {"episode": episode, "return": total, "seconds": length * robot.dt,
                      "mean_forward_velocity": float(np.mean(velocities)), "peak_height": peak,
                      "jump_landing_detected": landings, "fallen": info["fallen"],
                      "distance_xy": float(np.linalg.norm(np.array(info["position_xy"]) - initial_xy)),
                      "mean_absolute_speed_error": float(np.mean(np.abs(np.array(velocities) - robot.command[0]))),
                      "single_support_fraction": single_support / length}
            results.append(result)
            print(json.dumps(result))
    finally:
        env.close()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2))
    if args.gif and frames:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=round(frame_stride * robot.dt * 1000), loop=0)
        print(f"First episode GIF: {args.gif}")


if __name__ == "__main__":
    main()
