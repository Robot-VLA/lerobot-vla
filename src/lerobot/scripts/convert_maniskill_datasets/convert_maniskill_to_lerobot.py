"""
COMPLETE AND UPLOADED
https://huggingface.co/datasets/brandonyang/PokeCube-v1
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/PokeCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --repo_id brandonyang/PokeCube-v1 \
    --task_description "Poke a red cube with a peg and push it to a target goal position."

COMPLETE AND UPLOADED
https://huggingface.co/datasets/brandonyang/LiftPegUpright-v1
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --repo_id brandonyang/LiftPegUpright-v1 \
    --task_description "Move a peg laying on the table to any upright position on the table."

COMPLETE AND UPLOADED
https://huggingface.co/datasets/brandonyang/PushCube-v1
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/PushCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --repo_id brandonyang/PushCube-v1 \
    --task_description "Push and move a cube to a goal region in front of it."

COMPLETE AND UPLOADED
https://huggingface.co/datasets/brandonyang/PickCube-v1
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/PickCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --repo_id brandonyang/PickCube-v1 \
    --task_description "Grasp a red cube and move it to a target goal position."

COMPLETE AND UPLOADED
https://huggingface.co/datasets/brandonyang/StackCube-v1
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/StackCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --repo_id brandonyang/StackCube-v1 \
    --task_description "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling."

COMPLETE AND UPLOADED
https://huggingface.co/datasets/brandonyang/PegInsertionSide-v1
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/PegInsertionSide-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5 \
    --repo_id brandonyang/PegInsertionSide-v1 \
    --task_description "Pick up a orange-white peg and insert the orange end into the box with a hole in it."

IN PROGRESS 1L
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/DrawTriangle-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5 \
    --repo_id brandonyang/DrawTriangle-v1 \
    --task_description "Draw a triangle on the canvas using the robot's end-effector."

# TODO(branyang02): Update ManiSkill or manually add StackPyramid-v1 env.

COMPLETE AND UPLOADED
https://huggingface.co/datasets/brandonyang/PushT-v1
uv run python src/lerobot/scripts/convert_maniskill_datasets/convert_maniskill_to_lerobot.py \
    --file_path /PFS/output/yangyifan/cache/maniskill_datasets/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --repo_id brandonyang/PushT-v1 \
    --task_description "Push the T-shaped object to the target position using the robot's end-effector."
"""

from dataclasses import dataclass

import h5py
import torch
import tyro
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

OBS_SPACE = 9  # qpod
ACTION_SPACE = 7  # pd_ee_delta_pose

DATASET_FEATURES = {
    "observation.images.base_camera": {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "channels"],
        "info": None,
    },
    # "observation.images.hand_camera": {
    #     "dtype": "video",
    #     "shape": (128, 128, 3),
    #     "names": ["height", "width", "channels"],
    #     "info": None,
    # },
    "observation.state": {
        "dtype": "float32",
        "shape": (OBS_SPACE,),
        "names": [f"state_{i}" for i in range(OBS_SPACE)],
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_SPACE,),
        "names": [f"action_{i}" for i in range(ACTION_SPACE)],
    },
}


@dataclass
class Args:
    file_path: str
    repo_id: str
    task_description: str
    fps: int = 20
    push_to_hub: bool = True


def main(args: Args):
    h5_file = h5py.File(args.file_path, "r")

    lerobot_dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=DATASET_FEATURES,
    )

    traj_names = sorted([k for k in h5_file if k.startswith("traj_")])
    traj_names = sorted(traj_names, key=lambda x: int(x.split("_")[1]))

    for traj_name in tqdm(traj_names, desc="Processing trajectories"):
        group = h5_file[traj_name]
        obs = group["obs"]
        actions = group["actions"][:]  # (T, 3)
        qpos = obs["agent"]["qpos"][:]  # (T+1, 7)
        base_rgb = obs["sensor_data"]["base_camera"]["rgb"][:]  # (T+1, H, W, 3)
        # hand_rgb = obs["sensor_data"]["hand_camera"]["rgb"][:]  # (T+1, H, W, 3)

        for t in range(actions.shape[0]):
            frame = {
                "observation.images.base_camera": torch.tensor(base_rgb[t]),
                # "observation.images.hand_camera": torch.tensor(hand_rgb[t]),
                "observation.state": torch.tensor(qpos[t]),
                "action": torch.tensor(actions[t]),
            }
            lerobot_dataset.add_frame(frame, task=args.task_description)
        lerobot_dataset.save_episode()
    if args.push_to_hub:
        lerobot_dataset.push_to_hub(upload_large_folders=True)
    h5_file.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
