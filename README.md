## lerobot-vla
Since LeRobot is getting too cluttered...

## Requirements
- [`uv`](https://docs.astral.sh/uv/)

## Installation
```
git clone https://github.com/Robot-VLA/lerobot-vla.git
cd lerobot-vla
git submodule update --init --recursive

uv sync --prerelease=allow
# or set export UV_PRERELEASE=allow
```

## Usage

### Train
```sh
uv run python src/lerobot/scripts/train.py \
  --dataset.repo_id=brandonyang/StackCube-v1 \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0 \
  --policy.push_to_hub=false \
  --env.type=maniskill \
  --env.task=StackCube-v1 \
  --env.task_description="Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling."
```

### Eval
```sh
uv run python src/lerobot/scripts/eval.py \
  --policy.type=pi0 \
  --policy.pretrained_path=/path/to/pretrained_model \
  --policy.n_action_steps=20 \
  --env.type=maniskill \
  --env.task="StackCube-v1" \
  --env.task_description="Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling." 
```

```sh
# From RoboArena: https://github.com/arhanjain/sim-evals
uv run python src/lerobot/scripts/eval.py \
  --policy.type=pi0 \
  --policy.pretrained_path=/path/to/pretrained_model \
  --policy.n_action_steps=20 \
  --env.type=isaaclab \
  --env.task=DROID \
  --eval.n_episodes=2 \
  --eval.batch_size=2 \
  --env.task_description="put the cube in the bowl"
```

## Pretrained Models

| Model                                                                 | Description                                                                 |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [brandonyang/pi0](https://huggingface.co/brandonyang/pi0)             | Base diffusion `pi0` for fine-tuning, converted from [`pi0_base`](https://github.com/Physical-Intelligence/openpi).                 |
| [brandonyang/pi0_droid](https://huggingface.co/brandonyang/pi0_droid) | `pi0` model pretrained with all Pi data + finetuned on DROID, converted from `pi0_droid`.                             |
| [brandonyang/paligemma_diffusion_droid](https://huggingface.co/brandonyang/paligemma_diffusion_droid) | `pi0` model finetuned on DROID from base paligemma VLM, converted from `paligemma_diffusion_droid`.     |


## Datasets

### Simulation

### [ManiSkill Datasets](https://maniskill.readthedocs.io/en/latest/index.html)

- Control Mode: `pd_ee_delta_pose`
- Observation State: `qpos`
- Demonstration types:
  - MP: Motion Planning
  - RL: RL policy

| Task                                                                 | Demo Type | Cameras      | Obs. Dim | Action Dim |
|----------------------------------------------------------------------|-----------|--------------|----------|------------|
| [PokeCube-v1](https://huggingface.co/datasets/brandonyang/PokeCube-v1)            | RL        | base         | 9        | 7          |
| [LiftPegUpright-v1](https://huggingface.co/datasets/brandonyang/LiftPegUpright-v1) | RL        | base         | 9        | 7          |
| [PushCube-v1](https://huggingface.co/datasets/brandonyang/PushCube-v1)            | RL        | base         | 9        | 7          |
| [PickCube-v1](https://huggingface.co/datasets/brandonyang/PickCube-v1)            | RL        | base         | 9        | 7          |
| [StackCube-v1](https://huggingface.co/datasets/brandonyang/StackCube-v1)          | RL        | base + hand  | 9        | 7          |
| [PegInsertionSide-v1](https://huggingface.co/datasets/brandonyang/PegInsertionSide-v1) | MP        | base + hand  | 9        | 7          |
| [DrawTriangle-v1](https://huggingface.co/datasets/brandonyang/DrawTriangle-v1)    | MP        | base         | 7        | 6          |
| [PushT-v1](https://huggingface.co/datasets/brandonyang/PushT-v1)                  | RL        | base         | 7        | 6          |

Conversion scripts for ManiSkill datasets are available in `src/lerobot/scripts/convert_maniskill_datasets`.

### IsaacLab Datasets
WIP

### Real Robot
- [DROID](https://huggingface.co/datasets/cadene/droid)

## Misc

- Tested with [`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`](https://hub.docker.com/layers/nvidia/cuda/12.4.1-cudnn-devel-ubuntu22.04/images/sha256-0a434eff1826693c1e2a669b20062f9995e73ed3456cdb70416d7ba9c1e3d1f5?context=explore).

### Troubleshooting
- Try installing these packages:
  ```
  sudo apt install cmake build-essential libsm6 libxext6 libxrender1 libxt6 libglu1-mesa mesa-utils libgl1-mesa-glx libglfw3 xvfb libglib2.0-0 libvulkan1 vulkan-tools libglvnd-dev
  ```
- Follow the [ManiSkill trouble shooting](https://maniskill.readthedocs.io/en/v3.0.0b21/user_guide/getting_started/installation.html#troubleshooting) for `vulkan` setup.
- If IsaacLab runs into `vulkan` issues, follow the ManiSkill troubleshooting for `vulkan` setup as well.
- If installation timeout, set `export UV_HTTP_TIMEOUT=90`.

### Apptainer

1. Create `ubuntu_gpu.def`
```def
# ubuntu_gpu.def
Bootstrap: docker
From: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

%post
  apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    ca-certificates \
    vim

  mkdir -p /workspace/repos
  mkdir -p /workspace/datasets

%environment
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

%runscript
  echo "[INFO] Container is ready."
  echo "[INFO] Code is mounted at /workspace/repos and datasets at /workspace/datasets."
```

2. Create sandbox container from `ubuntu_gpu.def`
```bash
module purge
module load apptainer
apptainer build --fakeroot --sandbox ubuntu_gpu/ ubuntu_gpu.def
```

3. Launch interactive container
```bash
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video

# We bind /root because IsaacSim uses /root for caching.
# Make sure the host dirs exist before executing, create dirs if they don't exist.

apptainer shell --nv --fakeroot \
  --containall \
  --no-home \
  --overlay ubuntu_overlay \
  --bind /path/to/repos/:/workspace/repos \
  --bind /path/to/datasets:/workspace/datasets \
  --bind /path/to/apptainer_root:/root \
  ubuntu_gpu/
```

4. Follow installation steps, install to `/workspace/repos/lerobot-vla`.


### Docker

1. Create `Dockerfile`
```dockerfile
# Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    ca-certificates \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Create workspace directories
RUN mkdir -p /workspace/repos /workspace/datasets

# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Default command
CMD echo "[INFO] Container is ready." && \
    echo "[INFO] Code is mounted at /workspace/repos and datasets at /workspace/datasets." && \
    bash
```

2. Build Docker Image
```
docker build -t ubuntu_gpu .
```

3. Run Container with Volume Mounts
```bash
docker run --rm -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video \
  -v /path/to/repos:/workspace/repos \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/docker_root:/root \
  ubuntu_gpu
```

4. Follow installation steps, install to `/workspace/repos/lerobot-vla`.

