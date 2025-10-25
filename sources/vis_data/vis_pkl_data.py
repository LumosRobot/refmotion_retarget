# display pkl folder pkl
import matplotlib.pyplot as plt
import itertools

def plot_pkl(motion_data, motion_names=["dance1_subject2"], fields=["right_knee_link_pos_x_b"]):
    fig, axes = plt.subplots(len(fields), 1, figsize=(15, 3 * len(fields)), sharex=True)

    # 保证 axes 是列表类型
    if len(fields) == 1:
        axes = [axes]

    # 常见线型组合
    linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]
    linestyle_cycle = itertools.cycle(linestyles)

    for motion_name in motion_names:
        linestyle = next(linestyle_cycle)

        traj_names = list(motion_data[motion_name].keys())
        print(f"{motion_name} contains trajectories: {traj_names}")
        data = motion_data[motion_name][traj_names[0]]

        for i, field in enumerate(fields):
            if field not in data["Fields"]:
                print(f"⚠️ Field '{field}' not found in {motion_name}")
                continue

            idx = data["Fields"].index(field)
            start_idx = 1
            end_idx = min(7000, data["Frames"].shape[0])
            axes[i].plot(
                    data["Frames"][start_idx:end_idx, idx],
                    label=motion_name,
                    linestyle=linestyle
                    )
            axes[i].set_title(f"{field}")
            axes[i].grid(True)
            axes[i].legend()

    axes[-1].set_xlabel("Frame Index")
    plt.tight_layout()
    plt.show()

# load dataset for demonstration:
import glob
from collections import OrderedDict
import os
from scipy.spatial.transform import Rotation as sRot
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,json
import torch
from scipy.signal import savgol_filter
# Configure the logging system
import logging
logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
        )
# Create a logger object
logger = logging.getLogger("vis_mj")
############## data files  ##################
aug_traj_num=1
#motion_files = glob.glob(os.getenv("HOME") + "/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/fit_motion/*.pkl")
motion_files = glob.glob(os.getenv("HOME") + "/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/*.pkl")
logger.info(f"Motion file is: {motion_files}")
motion_data = {}
for file in motion_files:
    motion_name = os.path.basename(file)[:-4]
    motion_data[motion_name] = joblib.load(file)
print(f"motion names: {motion_data.keys()}")



fields=["root_pos_x","root_pos_z",
        "root_vel_x_b","root_vel_y_b","root_ang_vel_z_b",
        "left_hip_pitch_joint_dof_vel",
        "left_hip_pitch_joint_dof_pos", "right_hip_pitch_joint_dof_pos",
        "left_hip_roll_joint_dof_pos", "right_hip_roll_joint_dof_pos",
        "left_hip_yaw_joint_dof_pos", "right_hip_yaw_joint_dof_pos",
        "left_knee_joint_dof_pos", "right_knee_joint_dof_pos",
        "left_ankle_roll_joint_dof_pos","right_ankle_roll_joint_dof_pos",
        "torso_joint_dof_pos",
        "left_shoulder_pitch_joint_dof_pos", "right_shoulder_pitch_joint_dof_pos",
        "left_shoulder_roll_joint_dof_pos", "right_shoulder_roll_joint_dof_pos",
        "left_shoulder_yaw_joint_dof_pos", "right_shoulder_yaw_joint_dof_pos",
        "left_elbow_joint_dof_pos", "right_elbow_joint_dof_pos",
        "left_ankle_pitch_joint_dof_pos","right_ankle_pitch_joint_dof_pos",
        ]

fields=[
        "left_hip_pitch_joint_dof_vel", "right_hip_pitch_joint_dof_vel",
        "left_hip_roll_joint_dof_vel", "right_hip_roll_joint_dof_vel",
        "left_hip_yaw_joint_dof_vel", "right_hip_yaw_joint_dof_vel",
        "left_knee_joint_dof_vel", "right_knee_joint_dof_vel",
        "left_ankle_roll_joint_dof_vel","right_ankle_roll_joint_dof_vel",
        "torso_joint_dof_vel",
        "left_shoulder_pitch_joint_dof_vel", "right_shoulder_pitch_joint_dof_vel",
        "left_shoulder_roll_joint_dof_vel", "right_shoulder_roll_joint_dof_vel",
        "left_shoulder_yaw_joint_dof_vel", "right_shoulder_yaw_joint_dof_vel",
        "left_elbow_joint_dof_vel", "right_elbow_joint_dof_vel",
        "left_ankle_pitch_joint_dof_vel","right_ankle_pitch_joint_dof_vel",
        ]

plot_pkl(motion_data, ["dance1_subject2_fps20","dance1_subject2_fps25","dance1_subject2_fps30"], fields=["right_hip_yaw_joint_dof_vel","left_hip_pitch_joint_dof_vel","left_knee_joint_dof_vel"])
