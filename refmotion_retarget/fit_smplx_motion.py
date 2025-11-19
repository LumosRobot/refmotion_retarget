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
from utils.load_amass import get_filtered_amass_data, load_amass_data, save_processed_data, qpos_to_pose

HERE = pathlib.Path(__file__).parent

def process_motion(key_name, key_name_to_pkls, cfg):
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(key_name_to_pkls[key_name], os.path.join(HERE, cfg.gmr.smplx_folder))
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=cfg.gmr.get("robot"),
        use_velocity_limit=False
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
        qpos = retarget.retarget(smplx_data,False)# offset ground 
        # save qpos
        qpos_list.append(qpos)
        human_motion_data.append(retarget.scaled_human_data)
        i += 1
            
    motion_data = qpos_to_pose(cfg, qpos_list, aligned_fps)

    motion_data["human_motion_data"] = human_motion_data

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
