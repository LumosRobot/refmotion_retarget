import argparse
import pathlib
import os
import time
import torch

import numpy as np

from utils.gmr import GeneralMotionRetargeting as GMR
from utils.gmr.smpl import load_smplx_file, get_smplx_data_offline_fast

from easydict import EasyDict
from rich import print
import hydra
from omegaconf import DictConfig
from utils.load_amass import get_filtered_amass_data, load_amass_data, save_processed_data

from utils.torch_humanoid_batch import Humanoid_Batch
HERE = pathlib.Path(__file__).parent

def process_motion(key_name, key_name_to_pkls, cfg):
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        key_name_to_pkls[key_name], os.path.join(HERE,cfg.gmr.smplx_folder)
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=cfg.gmr.get("robot"),
    )
    
    qpos_list = []
    human_motion_data = []
    # Start the viewer
    i = 0
    while True:
        if i >= len(smplx_data_frames):
            break
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]
        # retarget
        qpos = retarget.retarget(smplx_data)
        # save qpos
        qpos_list.append(qpos)
        human_motion_data.append(retarget.scaled_human_data)
        i += 1
            
    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # save from wxyz to xyzw
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    # get robot pose aa using humanoid_fk
    humanoid_fk = Humanoid_Batch(cfg.robot)  # Load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    frame_num = len(smplx_data_frames)
    dof_num = dof_pos.shape[1]
    
    pose_aa_robot = np.zeros((frame_num, 1 + dof_num + num_augment_joint, 3))
    pose_aa_robot[:, 0, :] = root_rot[:, :3]  # 根旋转
    pose_aa_robot[:, 1:1+dof_num, :] = humanoid_fk.dof_axis * dof_pos[:, :, np.newaxis]  # 关节轴角

    motion_data = EasyDict({
        "joint_names": humanoid_fk.joint_names,
        "body_names": humanoid_fk.body_names,
        "pose_aa": pose_aa_robot,

        "fps": aligned_fps,
        "root_trans": root_pos,
        "root_trans_offset": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_vels": None,

        #"robot_joints": None, #robot_joints.squeeze().detach().numpy(),
        #"smpl_joints": None,
        "human_motion_data": human_motion_data

    })

    return motion_data
    
@hydra.main(version_base=None, config_path="./../data/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to process AMASS motion data and save the results.
    """
    # Get Amass dataset
    key_name_to_pkls, key_names = get_filtered_amass_data(cfg)
    
    
    # retarget motions
    all_data = {}
    for key_name in key_names:
        data = process_motion(key_name, key_name_to_pkls, cfg)
        if data is not None:
            all_data[key_name] = data
       
    return save_processed_data(all_data, cfg)
            

if __name__ == "__main__":
    main()
