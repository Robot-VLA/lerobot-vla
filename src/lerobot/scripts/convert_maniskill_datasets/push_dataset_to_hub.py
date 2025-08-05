"""
This file is used if convert_maniskill_to_lerobot.py fails to push the dataset to the hub.

uv run python src/lerobot/scripts/convert_maniskill_datasets/push_dataset_to_hub.py \
    --repo-id brandonyang/PushT-v1
"""

from dataclasses import dataclass

import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class Args:
    repo_id: str = "brandonyang/PushT-v1"


def main(args: Args):
    dataset = LeRobotDataset(repo_id=args.repo_id)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args=args)
