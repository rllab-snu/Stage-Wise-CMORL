# Stage-Wise CMORL

This is an official GitHub Repository for paper ["Stage-Wise Reward Shaping for Acrobatic Robots: A Constrained Multi-Objective Reinforcement Learning Approach"](https://arxiv.org/abs/2409.15755).

## Requirement

- python==3.7
- torch==1.12.1
- numpy==1.21.5
- isaacgym (https://developer.nvidia.com/isaac-gym)
- IsaacGymEnvs (https://github.com/isaac-sim/IsaacGymEnvs)
- ruamel.yaml
- requests
- pandas
- scipy
- wandb

## Organization
```
Stage-Wise-CMORL/
    └── algos/
    │     └── common/
    │     └── comoppo/
    │     └── student/
    └── assets/
    │     └── go1/
    │     └── h1/
    └── tasks/
    └── utils/
    └── main_student.py
    └── main_teacher.py
```
- `algos/`: contains the implementation of the proposed algorithm
- `assets/`: contains the assets of the robots
- `tasks/`: contains the implementation of the tasks
- `utils/`: contains the utility functions

## Tasks

- GO1 Robot (Quadruped from Unitree)
    - Back-Flip
    - Side-Flip
    - Side-Roll
    - Two-Hand Walk
- H1 Robot (Humanoid from Unitree)
    - Back-Flip
    - Two-Hand Walk

## Training and Evaluation

It is required to train a teacher poicy first, and then train a student policy using the teacher policy.

### Teacher Learning

- training: `python main_teacher.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/comoppo/{task_name}.yaml --wandb --seed 1`
- test: `python main_teacher.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/comoppo/{task_name}.yaml --test --render --seed 1 --model_num {saved_model_num}`

### Student Learning

- training: `python main_student.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/student/{task_name}.yaml --wandb --seed 1`
- test: `python main_student.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/student/{task_name}.yaml --test --render --seed 1 --model_num {saved_model_num}`
