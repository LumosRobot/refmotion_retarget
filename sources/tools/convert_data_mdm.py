# Import necessary libraries
import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os
import sys
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.utils.transform_utils import quat_correct

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

# Load the input motion data
res_data = joblib.load("data/mdm/res_run.pk")

# Initialize an empty dictionary to store processed AMASS data
amass_data = {}

# Process each motion sequence in the input data
for i in range(len(res_data['json_file']['thetas'])):
    # Extract pose and translation data
    pose_euler = np.array(res_data['json_file']['thetas'])[i].reshape(-1, 24, 3)
    trans = np.array(res_data['json_file']['root_translation'])[i]
    B = pose_euler.shape[0]

    # Convert Euler angles to axis-angle representation
    pose_aa = sRot.from_euler('XYZ', pose_euler.reshape(-1, 3), degrees=True).as_rotvec().reshape(B, 72)

    # Apply a transformation to the root joint
    transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
    pose_aa[:, :3] = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
    trans = trans.dot(transform.as_matrix().T)
    trans[:, 2] -= (trans[0, 2] - 0.92)

    # Store the processed data
    amass_data[f"{i}"] = {"pose_aa": pose_aa, "trans": trans, "beta": np.zeros(10)}

# Define MuJoCo joint names
mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
]

# Initialize a dictionary to store the final motion data
amass_full_motion_dict = {}

# Process each entry in the AMASS data
for key_name in tqdm(amass_data.keys(), desc="Processing motion data"):
    smpl_data_entry = amass_data[key_name]
    pose_aa = smpl_data_entry['pose_aa']
    root_trans = smpl_data_entry['trans']
    beta = smpl_data_entry['beta']
    gender = smpl_data_entry.get("gender", "neutral")
    fps = 30.0

    # Handle gender information
    gender_number = {"neutral": [0], "male": [1], "female": [2]}.get(gender, [0])

    # Map SMPL joints to MuJoCo joints
    smpl_2_mujoco = [joint_names.index(joint) for joint in mujoco_joint_names if joint in joint_names]
    pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((pose_aa.shape[0], 6))], axis=1)
    pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :]

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

        # Apply filtering to smooth the motion
        root_trans_offset = gaussian_filter1d(root_trans_offset.numpy(), 3, axis=0, mode="nearest")
        root_trans_offset = torch.from_numpy(root_trans_offset)
        pose_quat_global = np.stack([quat_correct(pose_quat_global[:, i]) for i in range(pose_quat_global.shape[1])], axis=1)
        filtered_quats = gaussian_filter1d(pose_quat_global, 2, axis=0, mode="nearest")
        pose_quat_global = filtered_quats / np.linalg.norm(filtered_quats, axis=-1)[..., None]

        # Update the skeleton state
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False
        )
        pose_quat = new_sk_state.local_rotation.numpy()

    # Prepare the output dictionary for this entry
    new_motion_out = {
        'pose_quat_global': pose_quat_global,
        'pose_quat': pose_quat,
        'trans_orig': root_trans,
        'root_trans_offset': root_trans_offset,
        'beta': beta,
        'gender': gender,
        'pose_aa': pose_aa,
        'fps': fps
    }
    amass_full_motion_dict[key_name] = new_motion_out

# Save the processed motion data
joblib.dump(amass_full_motion_dict, "data/mdm/mdm_isaac_run.pkl")
print("Processed data saved to data/mdm/mdm_isaac_run.pkl")
