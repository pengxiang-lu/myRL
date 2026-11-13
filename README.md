# myRL

## 一、简介

　　**myRL 是一个基于 PPO 和 MuJoCo 的宇树 G1 强化学习项目**，使用 Gymnasium 封装仿真环境，通过 Stable-Baselines3 训练站立、行走和原地跳跃策略。机器人采用 G1 29 自由度 MJCF 模型，策略控制双腿 12 个关节，上身维持参考姿态。

　　项目使用 **uv + Python 3.12** 管理环境，默认在 NVIDIA GPU 上执行策略网络前向与梯度更新，MuJoCo 物理仿真和 rollout 缓冲区使用 CPU。训练与评估入口分别为 `src/train.py` 和 `src/evaluate.py`。

　　当前项目是可训练的研究基线，每个任务使用独立策略，尚未实现单策略动作切换或实机部署。已有约 100 万步行走试训，但 5 个评估回合平均存活 1.18 秒、全部摔倒，**尚未学会稳定行走或跳跃**。完整结果见 [训练报告](TRAINING_REPORT.md)。

## 二、安装教程

### 2.1 准备工程

　　以下命令以 Linux / Bash 环境为例。取得工程后进入项目根目录；本机目录为：

```bash
cd /home/lpx/python-project/myRL
```

　　工程位于其他位置时替换上述路径。后续安装、模型下载、训练和评估命令均在项目根目录执行，配置与资源路径按此目录解析。GPU 训练需要 NVIDIA GPU 和可用的驱动，可先检查：

```bash
nvidia-smi
```

### 2.2 安装依赖

　　项目使用 `uv` 创建独立虚拟环境，不依赖 Conda。尚未安装 uv 时，可执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

　　准备 Python 3.12，并按照 `uv.lock` 安装依赖：

```bash
uv python install 3.12
uv sync --locked --extra dev
```

　　uv 默认使用项目内的 `.venv`。普通 Python 包使用阿里云 PyPI 镜像；Linux / Windows 的 PyTorch 2.8 使用南京大学 CUDA 12.8 镜像，macOS 使用 CPU 源，具体配置见 `pyproject.toml`。CUDA 运行库随依赖安装，无需另外安装 CUDA Toolkit，但仍需可用的 NVIDIA 驱动。

　　训练默认使用 `--device cuda`。显式请求 CUDA 而设备不可用时，程序会直接报错；CPU 调试需添加 `--device cpu`。`--device auto` 允许自动选择可用设备。小型 MLP 的 PPO 训练不保证 GPU 比 CPU 更快，仿真也可能成为瓶颈。

### 2.3 准备 G1 模型

　　下载固定版本的 G1 XML、网格和许可证，默认保存在 `assets/unitree_g1/`：

```bash
uv run scripts/download_model.py
```

　　模型来自 [Google DeepMind MuJoCo Menagerie / unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_g1)。下载脚本固定上游提交，支持网络重试，并跳过已存在的文件；`source.json` 记录来源，`LICENSE` 保留上游许可。模型资产不加入版本管理，首次使用需要联网下载。

　　使用 `--output` 下载到其他位置时，还需在训练配置的 `env.model_path` 中指定对应的 `scene.xml` 路径。

### 2.4 快速验证

　　模型准备完成后，先检查环境接口，再运行短程 PPO 训练和策略加载：

```bash
uv run pytest -q

uv run src/train.py \
  --config configs/smoke.json \
  --device cuda:0 \
  --output runs/quickstart_smoke

uv run src/evaluate.py \
  --checkpoint runs/quickstart_smoke/final \
  --episodes 2 \
  --report runs/quickstart_smoke/evaluation.json
```

　　没有可用 GPU 时，将训练命令中的设备改为 `--device cpu`。smoke 配置只训练 128 步、每回合上限 1 秒，用于检查仿真、优化更新和检查点保存/加载，不能用于判断策略是否学会任务。`--output` 指定的训练目录必须尚不存在；重复运行时请使用新的目录名。

## 三、训练

### 3.1 站立训练

　　站立任务用于学习平衡，可作为行走和跳跃策略的初始化来源。默认训练 5,000,000 步，使用 4 个仿真进程：

```bash
uv run src/train.py \
  --config configs/stand.json \
  --device cuda:0 \
  --output runs/stand
```

　　站立策略训练完成后，应先按第四节进行评估，确认平衡表现，再用于其他任务的热启动。

### 3.2 行走训练

　　行走任务默认目标前向速度为 0.5 m/s，训练 5,000,000 步。以下命令从站立策略初始化：

```bash
uv run src/train.py \
  --config configs/walk.json \
  --init-from runs/stand/final \
  --device cuda:0 \
  --output runs/walk
```

　　去掉 `--init-from` 即可从零训练。前进速度、侧向速度和转向速度分别由配置中的 `env.forward_speed`、`env.lateral_speed` 和 `env.yaw_rate` 控制。

### 3.3 跳跃训练

　　跳跃任务默认训练 10,000,000 步，目标为原地周期性跳跃：骨盆相对站立参考高度升高 0.12 m，每 2 秒一个周期。

```bash
uv run src/train.py \
  --config configs/jump.json \
  --init-from runs/stand/final \
  --device cuda:0 \
  --output runs/jump
```

　　可通过 `env.jump_height` 和 `env.jump_period` 调整目标高度与周期。奖励包括准备下蹲、弹道高度跟踪、腾空和落地引导；检测到落地并不等同于已验证的跳跃成功。

### 3.4 参数设置与断点恢复

　　训练参数来自 `configs/*.json`，命令行可覆盖常用设置：

| 参数 | 作用 |
| --- | --- |
| `--config` | JSON 配置路径，默认 `configs/walk.json` |
| `--task` | 覆盖配置中的任务，可选 `stand`、`walk`、`jump` |
| `--steps` | 本次训练的采样步数；续训时表示追加步数 |
| `--num-envs` | 并行仿真环境数量，正式配置默认 4 |
| `--device` | 默认 `cuda`，也支持 `cuda:N`、`cpu` 和 `auto` |
| `--seed` | 随机种子，默认 42 |
| `--output` | 新的实验输出目录；省略时按任务和时间自动创建 |
| `--resume` | 恢复同一任务的检查点目录 |
| `--init-from` | 加载已有策略，开始新的训练或切换任务 |

　　例如，指定训练步数和并行环境数量：

```bash
uv run src/train.py \
  --config configs/walk.json \
  --steps 1000000 \
  --num-envs 8 \
  --device cuda:0 \
  --output runs/walk_1m
```

　　从已有检查点继续训练，使用 `--resume` 并指定新的输出目录。下面的 `--steps 1000000` 表示在原有采样步数上**再追加 1,000,000 步**，不是设置累计终止步数：

```bash
uv run src/train.py \
  --config configs/walk.json \
  --resume runs/walk/final \
  --steps 1000000 \
  --device cuda:0 \
  --output runs/walk_more
```

　　`--resume` 恢复检查点中的任务环境、PPO 参数、优化器和归一化统计，支持将先前 CPU 训练的策略恢复到 GPU。`--init-from` 复制网络权重和观测统计，重新创建优化器并重置回报归一化统计，适合从站立切换到行走或跳跃。两者不能同时使用；恢复训练时切换任务应改用 `--init-from`。

　　PPO 按完整 rollout 更新，实际采样数可能向上取整。当前没有自动课程、域随机化或地形生成，速度、跳跃高度等训练难度需要根据评估结果手动调整。

### 3.5 训练日志与检查点

　　使用 TensorBoard 查看回报、回合长度及机器人状态：

```bash
uv run tensorboard --logdir runs
```

　　训练输出按实验目录保存，例如：

```text
runs/walk/
├── run.json
├── memory.json
├── tb/
├── step_<采样步数>/
│   ├── policy.zip
│   ├── vecnormalize.pkl
│   └── env.json
└── final/
    ├── policy.zip
    ├── vecnormalize.pkl
    └── env.json
```

　　`run.json` 记录配置、种子、训练设备和策略参数实际所在设备；`tb/` 保存 TensorBoard 日志。正式配置每约 100,000 步保存一次中间检查点，结束或 Ctrl+C 中断训练时保存 `final/`。恢复或评估时，需完整保留 `policy.zip`、`vecnormalize.pkl` 和 `env.json`，`--checkpoint`、`--resume`、`--init-from` 均传入包含这三份文件的目录。

　　`memory.json` 在训练结束时记录 PyTorch 峰值分配与缓存显存，单位为 MiB，不包含 CUDA 上下文、驱动及其他进程的开销。训练期间可使用 `nvidia-smi` 查看整卡和进程占用。

## 四、评估

　　评估根据检查点的 `env.json` 创建任务环境，冻结观测归一化统计，使用确定性动作和原始奖励。默认使用 CPU；GPU 训练的策略可以直接在 CPU 上评估，也可添加 `--device cuda:0` 使用 GPU。查看完整参数：

```bash
uv run src/evaluate.py --help
```

### 4.1 定量评估

　　运行多个回合，并将逐回合结果保存为 JSON：

```bash
uv run src/evaluate.py \
  --checkpoint runs/walk/final \
  --episodes 10 \
  --seed 100 \
  --device cpu \
  --report runs/walk/evaluation.json
```

　　站立或跳跃任务使用对应检查点即可，例如将 `--checkpoint` 改为 `runs/stand/final` 或 `runs/jump/final`。`--episodes` 默认 5，`--seed` 默认 100。评估会输出每回合的累计奖励、存活时长、平均前向速度、平均绝对速度误差、平面净位移、最高骨盆高度、单足支撑占比，以及是否摔倒、是否检测到跳跃落地。

　　前向速度使用骨盆局部坐标系，失衡过程也包含在统计中。判断行走质量应结合存活时间、速度跟踪和仿真画面，不能仅以平均速度接近目标值作为成功依据。比较不同策略时，应保持评估种子、回合数和任务设置一致。

### 4.2 交互仿真观看

　　在带图形桌面的环境中，使用 MuJoCo viewer 观看策略：

```bash
uv run src/evaluate.py \
  --checkpoint runs/walk/final \
  --render \
  --episodes 3
```

　　观看跳跃时替换为 `--checkpoint runs/jump/final`。无桌面服务器可以直接运行不带 `--render` 的定量评估，或使用下一节的离屏录制。

### 4.3 保存仿真动画

　　在支持 EGL 的环境中，可离屏录制首个评估回合：

```bash
MUJOCO_GL=egl uv run src/evaluate.py \
  --checkpoint runs/walk/final \
  --episodes 5 \
  --gif runs/walk/preview.gif \
  --report runs/walk/evaluation.json
```

　　`--gif` 以约 10 fps 保存**首个回合**的动画，不会为全部回合生成视频，且不能与 `--render` 同时使用。`--report` 则保存全部评估回合的逐回合指标；不指定时，指标只在终端输出。

　　检查点移动到其他机器后，若原模型路径不可用，可添加 `--model-path assets/unitree_g1/scene.xml` 指定本机 G1 场景文件。评估没有断点跳过已完成回合的功能，每次运行都会重新执行指定回合数；需要保留历史结果时，请使用新的报告和动画路径。

## 附：环境设置与工程结构

　　默认物理步长为 0.002 秒，每 10 个物理步执行一次策略控制，对应 50 Hz 控制频率。12 维动作在 `[-1, 1]` 内，表示双腿目标关节位置相对参考姿态的偏移，默认幅度为 0.5 rad；上身 17 个执行器保持参考姿态，并沿用模型的位置伺服与力矩限制。

　　观测包含骨盆局部线速度与角速度、投影重力、29 个关节的位置偏差与速度、速度指令、周期相位、双足接触、上一动作和骨盆高度，使用仿真真值。通用奖励包含直立、速度与转向跟踪、目标高度、姿态、力矩和动作平滑项，行走增加交替支撑奖励，跳跃增加腾空与落地奖励。骨盆过低、倾倒、非足部着地或数值异常会结束回合，时间上限单独作为 truncation 处理。

　　策略网络和价值网络各使用两层 256 单元的 MLP，训练采用 PPO、GAE、观测/奖励归一化与 KL 提前停止。环境参数定义于 `src/g1_rl/env.py` 的 `EnvConfig`，训练超参数位于各 JSON 配置的 `ppo` 字段。

```text
myRL/
├── pyproject.toml             uv 项目与依赖来源
├── uv.lock                    依赖版本锁定
├── configs/                   站立、行走、跳跃和 smoke 配置
├── scripts/
│   └── download_model.py      固定版本 G1 模型下载
├── src/
│   ├── train.py               训练脚本入口
│   ├── evaluate.py            评估脚本入口
│   └── g1_rl/
│       ├── env.py             MuJoCo 环境、奖励和终止判定
│       ├── train.py           PPO 训练、热启动和断点恢复
│       ├── evaluate.py        策略评估、viewer 和 GIF 录制
│       └── device.py          设备检查与显存统计
├── tests/                     环境接口、复现性和设备选择测试
├── assets/unitree_g1/         下载后的 G1 资产
├── runs/                      本机训练与评估产物
└── TRAINING_REPORT.md          已有行走试训记录
```
