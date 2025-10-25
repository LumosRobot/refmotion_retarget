# Import necessary libraries
import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
import os
import sys
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

# Add the current working directory to the system path
sys.path.append(os.getcwd())

# Configuration for the SMPL robot
robot_cfg = {
    "mesh": False,
    "model": "smpl",
    "upright_start": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}
print("Robot configuration:", robot_cfg)

# Initialize the SMPL robot
smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/smpl")

# Load the input AMASS data
amass_data = joblib.load("insert_your_data")

# Flag to indicate if mirrored data should be generated
double = False

# Define MuJoCo joint names
mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
]

# Initialize a dictionary to store the final motion data
full_motion_dict = {}

# Process each entry in the AMASS data
for key_name in tqdm(amass_data.keys(), desc="Processing AMASS data"):
    smpl_data_entry = amass_data[key_name]

    # Extract pose, translation, and other metadata
    pose_aa = smpl_data_entry['pose_aa'].copy()
    root_trans = smpl_data_entry['trans'].copy()
    beta = smpl_data_entry.get('beta', smpl_data_entry.get('betas', np.zeros(10)))
    beta = beta[0] if len(beta.shape) == 2 else beta
    gender = smpl_data_entry.get("gender", "neutral")
    fps = smpl_data_entry.get("fps", 30.0)

    # Normalize gender information
    gender_number = {"neutral": [0], "male": [1], "female": [2]}.get(gender, [0])

    # Map SMPL joints to MuJoCo joints
    smpl_2_mujoco = [joint_names.index(joint) for joint in mujoco_joint_names if joint in joint_names]
    pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((pose_aa.shape[0], 6))], axis=1)
    pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :]

    # Process the data (original and mirrored if `double` is True)
    for idx in range(2 if double else 1):
        # Convert axis-angle to quaternion
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(pose_aa.shape[0], 24, 4)

        # Load the SMPL robot model
        smpl_local_robot.load_from_skeleton(
            betas=torch.from_numpy(beta[None, :]),
            gender=gender_number,
            objs_info=None
        )
        smpl_local_robot.write_xml("egoquest/data/assets/mjcf/smpl_humanoid_1.xml")

        # Load the skeleton tree
        skeleton_tree = SkeletonTree.from_mjcf("egoquest/data/assets/mjcf/smpl_humanoid_1.xml")

        # Compute root translation offset
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        # Create a new skeleton state
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True
        )

        # Adjust for upright start if required
        if robot_cfg['upright_start']:
            pose_quat_global = (
                sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            ).as_quat().reshape(pose_aa.shape[0], -1, 4)

            # Handle mirrored data if `double` is True
            if idx == 1:
                left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
                pose_quat_global = pose_quat_global[:, left_to_right_index]
                pose_quat_global[..., 0] *= -1
                pose_quat_global[..., 2] *= -1
                root_trans_offset[..., 1] *= -1

            # Update the skeleton state
            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,
                torch.from_numpy(pose_quat_global),
                root_trans_offset,
                is_local=False
            )
            pose_quat = new_sk_state.local_rotation.numpy()

        # Prepare the output dictionary for this entry
        key_name_dump = f"{key_name}_{idx}" if double else key_name
        full_motion_dict[key_name_dump] = {
            'pose_quat_global': pose_quat_global,
            'pose_quat': pose_quat,
            'trans_orig': root_trans,
            'root_trans_offset': root_trans_offset,
            'beta': beta,
            'gender': gender,
            'pose_aa': pose_aa,
            'fps': fps
        }

# Save the processed motion data
joblib.dump(full_motion_dict, "insert_your_data")
print("Processed data saved to insert_your_data")
