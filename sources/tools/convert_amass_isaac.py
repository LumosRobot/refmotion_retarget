import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import sys

# Add the current working directory to the system path
sys.path.append(os.getcwd())

# Import necessary modules from the project
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

def run(in_file: str, out_file: str):
    """
    Main function to process AMASS data and convert it into a format compatible with Isaac Gym.

    Args:
        in_file (str): Path to the input AMASS data file.
        out_file (str): Path to save the processed output file.
    """
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
    amass_data = joblib.load(in_file)

    # Define MuJoCo joint names
    mujoco_joint_names = [
        'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
        'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
        'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
    ]

    # Initialize the output dictionary
    amass_full_motion_dict = {}

    # Process each entry in the AMASS data
    for key_name in tqdm(amass_data.keys(), desc="Processing AMASS data"):
        smpl_data_entry = amass_data[key_name]

        # Extract pose, translation, and other metadata
        pose_aa = smpl_data_entry['pose_aa']
        root_trans = smpl_data_entry['trans']
        beta = smpl_data_entry.get('beta', smpl_data_entry.get('betas', np.zeros(10)))
        gender = smpl_data_entry.get('gender', "neutral")
        fps = 30.0

        # Ensure beta is a 1D array
        if len(beta.shape) == 2:
            beta = beta[0]

        # Convert gender to a numeric representation
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
        smpl_local_robot.write_xml("phc/data/assets/mjcf/smpl_humanoid_1.xml")

        # Load the skeleton tree
        skeleton_tree = SkeletonTree.from_mjcf("phc/data/assets/mjcf/smpl_humanoid_1.xml")

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

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,
                torch.from_numpy(pose_quat_global),
                root_trans_offset,
                is_local=False
            )
            pose_quat = new_sk_state.local_rotation.numpy()

        # Prepare the output dictionary for this entry
        new_motion_out = {
            'pose_quat_global': new_sk_state.global_rotation.numpy(),
            'pose_quat': pose_quat,
            'trans_orig': root_trans,
            'root_trans_offset': root_trans_offset.numpy(),
            'beta': beta,
            'gender': gender,
            'pose_aa': pose_aa,
            'fps': fps
        }
        amass_full_motion_dict[key_name] = new_motion_out

    # Ensure the output directory exists
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    # Save the processed data
    joblib.dump(amass_full_motion_dict, out_file)
    print(f"Processed data saved to {out_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert AMASS data to Isaac Gym format.")
    parser.add_argument("--in_file", type=str, required=True, help="Path to the input AMASS data file.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to save the processed output file.")
    args = parser.parse_args()

    # Run the conversion process
    run(in_file=args.in_file, out_file=args.out_file)
