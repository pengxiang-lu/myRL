"""PPO training with matched observation statistics and checkpoint metadata."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

from g1_rl.env import EnvConfig, G1Env
from g1_rl.device import resolve_device, device_details, memory_stats


def make_env(config):
    def factory():
        return Monitor(G1Env(EnvConfig(**config)))
    return factory


def save_bundle(model, env, directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "policy.zip")
    env.save(directory / "vecnormalize.pkl")
    (directory / "env.json").write_text(json.dumps(config, indent=2))


class SaveCallback(BaseCallback):
    def __init__(self, output, interval, config):
        super().__init__()
        self.output, self.interval, self.config = output, interval, config
        self.last_save = 0

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.interval:
            save_bundle(self.model, self.training_env,
                        self.output / f"step_{self.num_timesteps}", self.config)
            self.last_save = self.num_timesteps
        infos = self.locals.get("infos", [])
        if infos:
            for key in ("height", "forward_velocity", "peak_height"):
                self.logger.record(f"robot/{key}", sum(i[key] for i in infos) / len(infos))
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/walk.json"))
    parser.add_argument("--task", choices=["stand", "walk", "jump"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Checkpoint directory; continues the SAME task")
    parser.add_argument("--init-from", type=Path, help="Warm start another task, including observation statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="cuda (default), cuda:N, cpu or auto")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")
    try:
        device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    hardware = device_details(device)
    print(f"Training hardware: {json.dumps(hardware)}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    settings = json.loads(args.config.read_text())
    env_config = asdict(EnvConfig(**settings.get("env", {})))
    if args.task:
        env_config["task"] = args.task
    env_config["model_path"] = str(Path(env_config["model_path"]).expanduser().resolve())
    source = args.resume or args.init_from
    if args.resume:
        saved = json.loads((source / "env.json").read_text())
        if args.task and args.task != saved["task"]:
            parser.error("Use --init-from to switch tasks")
        env_config = saved
    count = args.num_envs if args.num_envs is not None else settings.get("num_envs", 4)
    steps = args.steps if args.steps is not None else settings.get("total_timesteps", 5_000_000)
    if count < 1 or steps < 1:
        parser.error("num-envs and steps must be positive")
    output = args.output or Path("runs") / f"{env_config['task']}_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings.get("torch_threads", 1))
    factories = [make_env(env_config) for _ in range(count)]
    vec = DummyVecEnv(factories) if count == 1 else SubprocVecEnv(factories, start_method="spawn")
    vec.seed(args.seed)
    env = VecNormalize.load(source / "vecnormalize.pkl", vec) if source else VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.)
    env.training = True
    env.norm_reward = True
    kwargs = settings.get("ppo", {})
    try:
        if args.resume:
            model = PPO.load(source / "policy.zip", env=env, device=device, tensorboard_log=str(output / "tb"))
        else:
            model = PPO("MlpPolicy", env, seed=args.seed, device=device, verbose=1,
                        tensorboard_log=str(output / "tb"),
                        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}}, **kwargs)
            if args.init_from:
                previous = PPO.load(source / "policy.zip", device=device)
                model.policy.load_state_dict(previous.policy.state_dict())
                del previous
                # Keep observation scaling but start fresh reward-return statistics.
                from stable_baselines3.common.running_mean_std import RunningMeanStd
                env.ret_rms = RunningMeanStd(shape=())
        (output / "run.json").write_text(json.dumps({"env": env_config, "settings": settings,
            "seed": args.seed, "num_envs": count, "requested_steps": steps, "source": str(source),
            "hardware": hardware, "policy_device": str(next(model.policy.parameters()).device)}, indent=2))
        try:
            model.learn(total_timesteps=steps, reset_num_timesteps=not bool(args.resume),
                        callback=SaveCallback(output, settings.get("save_every", 100_000), env_config))
        except KeyboardInterrupt:
            print("Interrupted; saving current policy and normalization statistics.")
        save_bundle(model, env, output / "final", env_config)
        usage = memory_stats(device)
        (output / "memory.json").write_text(json.dumps(usage, indent=2))
        print(f"PyTorch memory (CUDA context/driver excluded): {json.dumps(usage)}")
        print(f"Checkpoint: {output / 'final'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
