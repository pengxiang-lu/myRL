# G1 行走试训记录 · 2026-11-13

最终策略尚未学会稳定行走。训练链路可运行，但评估中全部回合失衡，不能仅凭前向速度或训练回报判断成功。

## 设置

- 原始 `configs/walk.json`，目标前向速度 0.5 m/s，episode 上限 12 秒。
- 4 个 MuJoCo 环境，CPU PPO；训练 seed=42。
- 首轮 102,400 步，再恢复追加 901,120 步，共 1,003,520 步。实际步数按完整 rollout 向上取整。
- 两段日志中的训练耗时分别为 61 秒、450 秒，合计约 8.5 分钟；不含评估与渲染。
- 每个检查点评估 5 个确定性 episode，固定评估 RNG seed=100，冻结 VecNormalize。
- 本次训练未修改奖励或动作控制参数，构成原始配置的基线试验。此次没有正式训练跳跃策略。

## 结果

速度指标先按 episode 求均值，再对 5 个 episode 等权平均。前向速度使用骨盆局部坐标系，摔倒过程也包含在统计中。

| 采样步数 | 平均存活时间 / s | 平均前向速度 / (m/s) | 平均绝对速度误差 / (m/s) | 摔倒回合 |
| --- | --- | --- | --- | --- |
| 102,400 | 0.780 | 0.187 | 0.563 | 5/5 |
| 202,404 | 1.268 | 0.232 | 0.501 | 5/5 |
| 502,404 | 0.904 | 0.657 | 0.484 | 5/5 |
| 1,003,520 | 1.180 | 0.488 | 0.607 | 5/5 |

最终策略各回合存活时间为 1.42、1.14、1.26、0.96、1.12 秒，均未达到 12 秒上限。第 2 回合的平均速度为负；其余回合的向前运动也不能视为成功步态。

## 复现与查看

在项目根目录运行：

```bash
uv run src/train.py --device cpu --config configs/walk.json --steps 100000 --num-envs 4 --output runs/walk_repro_100k
uv run src/train.py --device cpu --config configs/walk.json --resume runs/walk_repro_100k/final --steps 900000 --num-envs 4 --output runs/walk_repro_1m

uv run src/evaluate.py --checkpoint runs/walk_trial_1m/final --episodes 5 --render
MUJOCO_GL=egl uv run src/evaluate.py --checkpoint runs/walk_trial_1m/final --episodes 5 --gif runs/walk_trial_1m/replay.gif --report runs/walk_trial_1m/reevaluation.json
```

- [最终策略目录](runs/walk_trial_1m/final/)
- [原始逐回合评估数据](runs/walk_trial_1m/evaluation.json)
- [首个回合动画](runs/walk_trial_1m/preview.gif)：约 1.4 秒，10 fps，展示到 episode 终止前的状态。
- [首轮训练日志](runs/walk_trial_100k/train.log)
- [续训日志](runs/walk_trial_1m/train.log)

## 下一步建议

优先降低初始动作探索幅度、先训练站立平衡，再进行低速行走课程；增加防滑与足部抬升约束，并使用独立评估挑选策略。这些是后续实验方向，尚未在本次试验中验证。
