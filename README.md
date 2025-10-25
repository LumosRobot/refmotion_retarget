<h1 align="left">🤖Humanoid_demo_retarget: retargeting to our own humanoids </h1>

Refer to this repository: https://github.com/ZhengyiLuo/PHC.git

# Installation

Follow the steps below to set up the Python environment for this project.

1. Create a Conda Environment.

```bash
conda create -n env_retarget python=3.8
conda activate env_retarget
```

2. Install PyTorch with CUDA Support and install GLFW.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge glfw
```

3. Install Other Python Dependencies.

```bash
pip install -r requirements.txt
```

# Data

## Download the SMPL

Download SMPL paramters . Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the SMPL model parameters from the following sources:

- Official SMPL website: [SMPL (smpl.is.tue.mpg.de)](https://smpl.is.tue.mpg.de/)

- (**Prefered**) SMPL models: https://web.ugreen.cloud/web/#/share/4b1e9c4317bc4548aec7679e5915c56b 提取码：44T6

```
|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
```

Make sure the `.pkl` files are placed directly under `data/smpl/` as shown above.

## Download the AMASS data

Training the agents requires using AMASS data. The `motion_file` parameter receives either an `.npz` file, for a single motion.

We provide example motions to get you started:

- AMASS dataset: https://web.ugreen.cloud/web/#/share/3ab9d6682040464ea973d3f7ed9bbe03 提取码：9X54

# Run Demo 

You have two ways to execute the demo:

1. **Manual Step-by-Step Execution** - Run each script individually (recommended for debugging)
2. **Automated Execution via `run.sh`** – Execute the full pipeline with a single command

## Option 1: Manual Step-by-Step Execution

### Step One: **Fit SMPL Shape to Robot Joints** 

You can fit the SMPL body model to a robot's joint configuration using the following command.

```bash
python fit_smpl_shape.py robot=lumos_lus2_joint27_fitting
```

 After running this script, you will obtain a set of `beta` parameters and a `scale` value such that the resulting SMPL joint positions closely align with the robot's joint configuration.

- Beta parameters (shape coefficients)

- Scale factor

### Step Two:  Motion Retargeting from AMASS Dataset

To fit a specific motion from the AMASS dataset (e.g., from the CMU subset), use the following command:

```bash
python fit_smpl_motion.py robot=lumos_lus2_joint27_fitting +motion_name=CMU_CMU_07_07_01_poses
```

In this example, `CMU_CMU_07_07_01` refers to the file `CMU/CMU/07/07_01_poses.npz`.The motion file used is:

```
humanoid_demo_retarget/data/AMASS/CMU/
└── CMU
    └── 07
        └── 07_01_poses.npz
```

**Note**:  "CMU_CMU_07_07_01_poses" corresponds to the file path above. Replace this with your desired motion file from the AMASS dataset.

### Step Three: Visualization

Render the retargeted motion using MuJoCo and save the results in a structured format:

```bash
python vis_q_mj.py robot=lumos_lus2_joint27_fitting +motion_name=CMU_CMU_07_07_01_poses

```

After running `vis_q_mj.py`, two `.pkl` files will be generated:

- `humanoid_demo_retarget/data/lus2_joint27/fit_motion/CMU_CMU_07_07_01_poses.pkl`
- `humanoid_demo_retarget/data/lus2_joint27/pkl/CMU_CMU_07_07_01_poses.pkl`

The file in the `pkl` folder is the one used by downstream reinforcement learning projects such as `st_gym`.

This `.pkl` file contains the following fields:

```python
dict_keys([
    'LoopMode',
    'FrameDuration',
    'EnableCycleOffsetPosition',
    'EnableCycleOffsetRotation',
    'MotionWeight',
    'Fields',
    'Frames'
])
```

These fields are structured specifically to support the motion input format expected by the `st_gym` environment.

## Option 2: Automated Execution (run.sh)

```bash
# Shape fitting :
./run.sh -r lumos_lus2_joint27 -s

# Full pipeline (shape fitting + motion retargeting + visualization):
./run.sh -r lumos_lus2_joint27 -s -f -v -m CMU_CMU_07_07_01_poses
```

Flags:

- `-r`: Robot name ( (used as YAML config prefix, e.g., `lumos_lus2_joint27`)
- `-s`: Run shape fitting(`fit_smpl_shape.py`)
- `-f`: Run motion fitting(`fit_smpl_motion.py`)
- `-v`: Visualize with MuJoCo(`vis_q_mj.py`)
- `-m`: Specify motion name from AMASS(e.g., `CMU_CMU_07_07_01_poses`, matches AMASS filename)

![Peek 2025-06-10 18-16](./docs/Peek 2025-06-10 18-16.gif)

# YAML files

YAML configuration files (located in `data/cfg/robot/`) define the retargeting setup for each robot.

Example: `lumos_lus2_joint27_fitting.yaml`

Key components:

- `extend_config`: Defines additional virtual joints (e.g., a `head_link` added under `pelvis`) to provide more constraints for motion fitting.
- `joint_matches`: Specifies one-to-one mapping between robot joints and SMPL joints (e.g., `"left_hip_pitch_link"` ↔ `"L_Hip"`).
- `smpl_pose_modifier`: Applies fixed rotation offsets to certain SMPL joints to better match the robot's joint coordinate conventions.

This file is used consistently across all stages of the pipeline to ensure proper retargeting alignment.

# Viewer Shortcuts

We provide a set of predefined keyboard controls for interacting with the MuJoCo viewer during playback:

| Key     | Description                                  |
| ------- | -------------------------------------------- |
| `R`     | Reset playback to the beginning              |
| `Space` | Toggle pause/resume                          |
| `T`     | Switch to the next motion in the loaded list |

# Acknowledgements

This project is based on the PHC (Perpetual Humanoid Control) framework.

If you find this work useful for your research, please cite the following paper:

```mathematica
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}
```

Also consider citing these prior works that have influenced or are used in this project:

```mathematica
@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}

@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

```

This repository is built on top of the following projects:

- Main code framework from: [IsaacGymEnvs](https://github.com/ikostrikov/IsaacGymEnvs)
- SMPL robot code adapted from: [UHC](https://github.com/xyz/UHC)
- SMPL models and layers from: SMPL-X

Please respect and follow the licenses of the above repositories when using this project.

# Issues

- if having problems related to glfw when run visulization, thus set this env variable: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
