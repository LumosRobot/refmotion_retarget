import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from smpl_sim.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger("fit_smpl_shape")



@hydra.main(version_base=None, config_path="./data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    
    humanoid_fk = Humanoid_Batch(cfg.robot) # load forward kinematics model

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment 
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]

    robot_joint_pick_idx = [ robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    #### Preparing fitting varialbes
    device = torch.device("cpu")
    pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], humanoid_fk.num_bodies , axis = 2), 1, axis = 1)
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float() # shape: 1,1,28,3 of axis-vector
    
    ###### prepare SMPL default pose for humanoid robot
    pose_aa_stand = np.zeros((1, 72)) #23*3+1*3
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
    
    for modifiers in cfg.robot.smpl_pose_modifier:
        modifier_key = list(modifiers.keys())[0]
        modifier_value = list(modifiers.values())[0]
        pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index(modifier_key)] = sRot.from_euler("xyz", eval(modifier_value),  degrees = False).as_rotvec()

    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72)) # initial state pose (aa)
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3]) # smpl model root pos
    beta = torch.zeros([1, 10]) # smpl model root 
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans) # get vertices (6890 vertices locations) and joints (24 joint locations)
    root_trans_offset = joints[:,0]

    fk_return = humanoid_fk.fk_batch(pose_aa_robot[None, ], root_trans_offset[None, 0:1])

    beta_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([beta_new, scale],lr=0.1)
    
    train_iterations=3000
    logger.info("start fitting shapes")
    pbar = tqdm(range(train_iterations))
    for iteration in pbar:
        # verts shape: (1, 6890, 3), (keypoints) joints shape: (1, 24, 3)
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta_new, trans[0:1]) # fitted smpl shape
        joints = (joints - joints[:, 0]) * scale + joints[:,0] # joints[:,0] is root pos
        if len(cfg.robot.extend_config) > 0:
            diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        else:
            diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

        loss = diff.norm(dim = -1).square().sum()

        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # logger.info the fitted shape and scale parameters
    logger.info(f"beta (shape parameter): {beta_new.detach()}")
    logger.info(f"scale: {scale}")

    # display smpl and humnoaid key points
    if cfg.get("vis", False):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt

        # humanoid 
        if len(cfg.robot.extend_config) > 0:
            j3d_humanoid = fk_return.global_translation_extend[0, :, robot_joint_pick_idx, :] .detach().numpy()
        else:
            j3d_humanoid = fk_return.global_translation[0, :, robot_joint_pick_idx, :] .detach().numpy()

        j3d_humanoid = j3d_humanoid - j3d_humanoid[:, 0:1]

        # smpl
        j3d_smpl = joints[:, smpl_joint_pick_idx].detach().numpy()
        j3d_smpl = j3d_smpl - j3d_smpl[:, 0:1]

        # display
        fig = plt.figure()
        idx = 0 # model/humanoid idx
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(90, 0)

        ax.scatter(j3d_smpl[idx, :,0], j3d_smpl[idx, :,1], j3d_smpl[idx, :,2], label='Fitted SMPL Shape', c='red')
        ax.scatter(j3d_humanoid[idx, :,0], j3d_humanoid[idx, :,1], j3d_humanoid[idx, :,2], label='Humanoid Shape', c='blue')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        drange = 1
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.show()

    os.makedirs(f"data/motions/{cfg.robot.humanoid_type}/fit_shape", exist_ok=True)
    joblib.dump((beta_new.detach(), scale), f"data/motions/{cfg.robot.humanoid_type}/fit_shape/shape_optimized_v1.pkl") # V2 has hip joints
    logger.info(f"fit_smpl_shape saving shape.pkl at data/motions/{cfg.robot.humanoid_type}/fit_shape/shape_optimized_v1.pkl")

    return 0


if __name__ == "__main__":
    main()
