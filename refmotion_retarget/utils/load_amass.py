import glob
import joblib
import numpy as np
from smpl_sim.utils import torch_utils
from easydict import EasyDict
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig
import glob
import os
import logging
from .torch_humanoid_batch import Humanoid_Batch
from scipy.spatial.transform import Rotation as sRot
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger("load_amass")

def load_amass_data(data_path):
    """
    Load AMASS motion data from a .npz file.

    Args:
        data_path (str): Path to the AMASS data file.

    Returns:
        dict: A dictionary containing pose, translation, gender, betas, and framerate.
    """
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if 'mocap_framerate' not in entry_data:
        return None

    return {
        # last two smpl_joints's pose/orentation keep the same with their parent link/joint in kinematic tree
        "pose_aa": np.concatenate([entry_data['poses'][:, :66], np.zeros((entry_data['trans'].shape[0], 6))], axis=-1), # pose param: 23 joint and a root axis-angle
        "gender": entry_data['gender'],
        "trans": entry_data['trans'], # tanslation of root in world frame
        "betas": entry_data['betas'], # shape paramster
        "fps": entry_data['mocap_framerate']
    }



def get_filtered_amass_data(cfg: DictConfig) -> Tuple[Dict[str, str], List[str]]:
    """
    Get AMASS dataset and apply filtering based on configuration.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        Tuple[Dict[str, str], List[str]]:
            - key_name_to_pkls: Dictionary mapping key names to file paths
            - filtered_key_names: List of filtered key names

    Raises:
        ValueError: If amass_root is not specified or no data found
    """
    # Get AMASS root directory
    amass_root = cfg.get("amass_root", None)
    if not amass_root:
        raise ValueError("amass_root is not specified in the config")

    # Find all npz files
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    if not all_pkls:
        raise ValueError(f"No data found in {amass_root}")

    # Create mapping from key names to file paths
    key_name_to_pkls = {
        "_".join(path.split("/")[len(amass_root.split("/")):]).replace(".npz", ""): path
        for path in all_pkls
    }

    # Get all key names
    all_key_names = list(key_name_to_pkls.keys())

    # Apply filtering
    motion_name = cfg.get("motion_name", None)
    filtered_key_names = all_key_names

    if motion_name:
        if motion_name == "filter":
            # Get motion names from filter configuration
            motion_name = cfg.motion_filter.Walk
            # Ensure motion_name is a list
            filtered_key_names = motion_name if isinstance(motion_name, list) else [motion_name]
        else:
            # Filter key names containing the motion_name
            filtered_key_names = [key for key in all_key_names if motion_name in key]

    logger.info(f"Selected data file names: {filtered_key_names}")
    #logger.info(f"Selected amass npz files: {key_name_to_pkls[key] for key in filtered_key_names}")

    return key_name_to_pkls, filtered_key_names


def save_processed_data(all_data: Dict, cfg: DictConfig) -> int:
    """
    Save processed motion data to files.

    Args:
        all_data (Dict): Dictionary containing processed motion data
        cfg (DictConfig): Configuration object

    Returns:
        int: 0 if successful, 1 if failed

    Raises:
        ValueError: If data_root is not specified in config
    """
    # Check if any data was successfully processed
    if not all_data:
        logger.error("No data was successfully processed")
        return 1

    # Validate configuration
    data_root = cfg.get("data_root", None)
    if not data_root:
        raise ValueError("data_root is not specified in the config")

    if not hasattr(cfg, 'robot') or not hasattr(cfg.robot, 'humanoid_type'):
        raise ValueError("robot.humanoid_type is not specified in the config")

    if not hasattr(cfg, 'motion_name'):
        raise ValueError("motion_name is not specified in the config")

    try:
        # Create output directory
        output_dir = f"{data_root}/motions/{cfg.robot.humanoid_type}/fit_motion"
        os.makedirs(output_dir, exist_ok=True)

        # Save as pkl file (combined data)
        pkl_path = f"{output_dir}/{cfg.motion_name}.pkl"
        logger.info(f"Saving combined fit data to {pkl_path} ...")
        joblib.dump(all_data, pkl_path)
        logger.info(f"Successfully saved combined data to {pkl_path}")

        # Save individual npz files
        successful_saves = 0
        for key_name, data in all_data.items():
            try:
                npz_path = f"{output_dir}/{key_name}.npz"
                np.savez(
                    npz_path,
                    dof_names=data["joint_names"],
                    body_names=data["body_names"],
                    dof_positions=data["dof_pos"],
                    dof_velocities=data["dof_vels"],
                    body_positions=data["global_translation"],
                    body_rotations=data["global_rotation"],
                    body_linear_velocities=data["global_velocity"],
                    body_angular_velocities=data["global_angular_velocity"],
                    fps=data["fps"]
                )
                logger.info(f"Saved individual fit data to {npz_path}")
                successful_saves += 1
            except Exception as e:
                logger.error(f"Failed to save {key_name}.npz: {e}")
                continue

        logger.info(f"Successfully processed and saved {successful_saves}/{len(all_data)} motions")

        if successful_saves == 0:
            logger.error("Failed to save any individual motion files")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return 1



def qpos_to_pose(cfg, qpos: [list, np.array], fps: float):

    if isinstance(qpos, list):
        qpos = np.asarray(qpos)

    # root pos
    root_pos = qpos[:, :3]

    # quaternion wxyz → xyzw
    root_rot = qpos[:, 3:7][:, [1, 2, 3, 0]]

    # joint dofs
    dof_pos = qpos[:, 7:]
    frame_num, dof_num = dof_pos.shape

    humanoid_fk = Humanoid_Batch(cfg.robot)
    num_augment = len(cfg.robot.extend_config)

    # allocate
    pose_aa = np.zeros((frame_num, 1 + dof_num + num_augment, 3))

    # convert root rot
    pose_aa[:, 0] = sRot.from_quat(root_rot).as_rotvec()

    # joint axis-angle = axis * angle
    axis = humanoid_fk.dof_axis  # (dof_num, 3)
    pose_aa[:, 1:1+dof_num] = dof_pos[..., None] * axis.cpu().numpy()

    motion_data = EasyDict(
        joint_names=humanoid_fk.joint_names,
        body_names=humanoid_fk.body_names,
        pose_aa=pose_aa,
        fps=fps,
        root_trans=root_pos,
        root_trans_offset=root_pos.copy(),
        root_rot=root_rot,
        dof_pos=dof_pos,
        dof_vels=None
    )

    return motion_data

