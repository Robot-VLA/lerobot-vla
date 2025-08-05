## 📦 Available Tasks

| Task                     | Demo Type | Cameras     | Observation Space | Action Space | Link                                                                 |
|--------------------------|-----------|-------------|-------------------|--------------|----------------------------------------------------------------------|
| PokeCube-v1              | RL        | base        | 9                 | 7            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/PokeCube-v1)              |
| LiftPegUpright-v1        | RL        | base        | 9                 | 7            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/LiftPegUpright-v1)        |
| PushCube-v1              | RL        | base        | 9                 | 7            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/PushCube-v1)              |
| PickCube-v1              | RL        | base        | 9                 | 7            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/PickCube-v1)              |
| StackCube-v1             | RL        | base + hand | 9                 | 7            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/StackCube-v1)             |
| PegInsertionSide-v1      | MP        | base + hand | 9                 | 7            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/PegInsertionSide-v1)      |
| DrawTriangle-v1          | MP        | base        | 7                 | 6            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/DrawTriangle-v1)          |
| PushT-v1                 | RL        | base        | 7                 | 6            | [🔗 Dataset](https://huggingface.co/datasets/brandonyang/PushT-v1)                 |

---

## ⚙️ How to Reproduce Datasets

1. **Download raw demo datasets**  
   From [ManiSkill_Demonstrations](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)

2. **Replay trajectories**  
   `replay_trajectory.py`: Convert the datasets to use `pd_ee_delta_pose` for action space and `qpos` for observation space:

3. **Convert to LeRobot format and upload**  
   `convert_maniskill_to_lerobot.py`: Convert the replayed datasets to the `LeRobotDataset` format and upload to HuggingFace:

4. **(Optional) Manual upload fallback**  
   `push_dataset_to_hub.py`: If the previous step fails to upload:
