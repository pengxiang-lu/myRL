from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from g1_rl.env import EnvConfig, G1Env

MODEL = Path(__file__).resolve().parents[1] / "assets/unitree_g1/scene.xml"


@pytest.fixture
def env():
    if not (MODEL.parent / "source.json").exists():
        pytest.fail("Model incomplete. Run uv run scripts/download_model.py first.")
    instance = G1Env(EnvConfig(model_path=str(MODEL)))
    yield instance
    instance.close()


def test_gym_contract(env):
    check_env(env, warn=True, skip_render_check=True)


def test_seeded_rollout(env):
    first, _ = env.reset(seed=17)
    transition = env.step(np.zeros(12))
    second, _ = env.reset(seed=17)
    repeated = env.step(np.zeros(12))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(transition[0], repeated[0])
    assert transition[1:4] == repeated[1:4]


@pytest.mark.parametrize("task", ["stand", "walk", "jump"])
def test_tasks_finite(env, task):
    env.cfg.task = task
    env.command[:] = [0.5, 0, 0] if task == "walk" else [0, 0, 0]
    env.reset(seed=3)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert set(info["contacts"]).issubset({0., 1.})
        if terminated or truncated:
            env.reset(seed=3)


def test_time_limit_and_fall_distinct(env):
    env.reset(seed=0)
    env.max_steps = 1
    _, _, terminated, truncated, _ = env.step(np.zeros(12))
    assert truncated and not terminated
    env.reset(seed=0)
    env.max_steps = 100
    env.data.qpos[2] = 0.3
    _, _, terminated, truncated, _ = env.step(np.zeros(12))
    assert terminated and not truncated


def test_jump_target_and_no_fake_landing(env):
    env.cfg.task = "jump"
    env.reset(seed=0)
    env.steps = round((0.6 + env.flight_time / 2) / env.dt)
    height, flight = env._jump_target()
    assert flight
    assert height == pytest.approx(env.home[2] + env.cfg.jump_height, abs=0.002)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(12))
    assert not info["landed"]


def test_action_validation(env):
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.full(12, np.nan))
    with pytest.raises(ValueError):
        env.step(np.zeros(29))
