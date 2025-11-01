# Import necessary libraries
import glob
import os
import sys
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch
from utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig

# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger("fit_smpl_motion")


# Add the current working directory to the system path
sys.path.append(os.getcwd())

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

def process_motion(key_name, key_name_to_pkls, cfg):
    """
    Process motion data and retarget it to a humanoid robot.

    Args:
        key_name (str): motion keys to process.
        key_name_to_pkls (dict): Mapping of motion keys to file paths.
        cfg (DictConfig): Configuration object.

    Returns:
        dict: Processed motion data.
    """
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot)  # Load forward kinematics model
    smpl_parser = SMPL_Parser(model_path="./../data/smpl", gender="neutral")

    # Load shape and scale parameters
    logger.info(f"fit_smpl_motion loading shape.pkl at data/motions/{cfg.robot.humanoid_type}/fit_shape/shape_optimized_v1.pkl")
    data_root = cfg.get("data_root", None)
    beta_new, scale = joblib.load(f"{data_root}/motions/{cfg.robot.humanoid_type}/fit_shape/shape_optimized_v1.pkl")
    robot_joint_names_augment = humanoid_fk.body_names_augment
    num_augment_joint = len(cfg.robot.extend_config)
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    try:
        amass_data = load_amass_data(key_name_to_pkls[key_name])
    except Exception as e:
        raise(f"Error loading data for {key_name}: {e}")

    if amass_data is None:
        logger.warn(f"there is no data for {key_name}")
        return None

    # Resample motion data to a target framerate
    fps = amass_data["fps"]
    trans = torch.from_numpy(amass_data['trans'])
    pose_aa = torch.from_numpy(amass_data['pose_aa']).float() # shape: frame num, 74 (pose params theta: 23+3+3)
    frame_num = trans.shape[0]

    if frame_num < 120:
        logger.debug(f"Motion {key_name} is too short.")
        return None

    # Compute joint positions and root translation offset
    with torch.no_grad():
        verts, smpl_joints = smpl_parser.get_joints_verts(pose_aa, beta_new, trans) # smpl_joints shape: frame_num, keypoint_num (24), 3
        smpl_joints = (smpl_joints - smpl_joints[:, 0:1]) * scale + smpl_joints[:, 0:1] # verts shape: frame_num, point_num(6890), 3
        smpl_joints[..., 2] -= verts[0, :, 2].min().item()
        root_trans_offset = smpl_joints[:, 0].clone() # shape: (frame_num,3), the pelis location of all frames

    # Compute root orientation
    """
    pose_aa[:,;3] indicate pelvis axis-angle (pose/orinetation)
    The quaternion [0.5, 0.5, 0.5, 0.5] corresponds to a 90° rotation about the axis (1,1,1). It often appears in frame conversions between Z-up ↔ Y-up coordinate systems.
    """
    gt_root_rot_quat = torch.from_numpy(
        (sRot.from_rotvec(pose_aa[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()
    ).float()
    gt_root_rot = torch.from_numpy(
        sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()
    ).float()

    # Initialize optimization variables
    dof_pos = torch.zeros((1, frame_num, humanoid_fk.num_dof, 1))
    dof_pos_new = Variable(dof_pos.clone(), requires_grad=True) # shape: 1, frame_num, joint_num, 1
    root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)# shape: 1,3
    root_trans_new = Variable(torch.zeros(1, 3), requires_grad=True)# shape: 1,3
    optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_trans_new], lr=0.02)

    # Optimization loop
    pbar = tqdm(range(cfg.get("fitting_iterations", 1000)))
    for iteration in pbar:
        pose_aa_robot_new = torch.cat([ #pose param: 3+24+3 axis-angle
            root_rot_new[None, :, None],
            humanoid_fk.dof_axis * dof_pos_new,
            torch.zeros((1, frame_num, num_augment_joint, 3)).to(device)
        ], axis=2)

        fk_return = humanoid_fk.fk_batch(pose_aa_robot_new, root_trans_offset[None, :] + root_trans_new,dt=1.0/fps) # batch of frame_num
        if num_augment_joint > 0:
            robot_joints = fk_return.global_translation_extend[:, :, robot_joint_pick_idx]
        else:
            robot_joints = fk_return.global_translation[:, :, robot_joint_pick_idx]

        diff = robot_joints - smpl_joints[:, smpl_joint_pick_idx]

        # === Loss terms ===
        # 1. Joint position match loss
        joint_loss = diff.norm(dim=-1).mean()

        # 2. DOF regularization (L2)
        dof_reg_loss = 0.01 * torch.mean(torch.square(dof_pos_new))

        # 3. Root smoothness over time (optional: if root_trans is per-frame)
        root_trans_smoothness_loss = ((root_trans_new[:, 1:] - root_trans_new[:, :-1])**2).mean()

        # 4. Temporal smoothness in joint angles
        joint_vel = dof_pos_new[:, 1:] - dof_pos_new[:, :-1]
        joint_vel_loss = 800 * (joint_vel**2).mean()

        # === Combine all losses ===
        loss = joint_loss + dof_reg_loss + root_trans_smoothness_loss + joint_vel_loss
        pbar.set_description_str(f"{iteration} - joint_loss: {joint_loss.item()}, dof_reg_loss: {dof_reg_loss}, root_trans_smoothness_loss: {root_trans_smoothness_loss.item()}, joint_vel_loss: {joint_vel_loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp and smooth joint positions
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        dof_pos_new.data = gaussian_filter_1d_batch(
            dof_pos_new.squeeze().transpose(1, 0)[None, :], kernel_size=5, sigma=0.95
        ).transpose(2, 1)[..., None]

    # Post-process root translation and save results
    root_trans = (root_trans_offset + root_trans_new).clone()
    combined_mesh = humanoid_fk.mesh_fk(pose_aa_robot_new[:, :1].detach(), root_trans[None, :1].detach())
    height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
    root_trans[..., 2] -= height_diff
    smpl_joints[..., 2] -= height_diff
    robot_joints[..., 2] -= height_diff

    fk_return = humanoid_fk.fk_batch(pose_aa_robot_new, root_trans[None,:],dt=1.0/fps, return_full=True) # batch of frame_num
    fk_return = EasyDict({key: value if key=="fps" else value.clone().detach().squeeze()  for key, value in fk_return.items()})

    all_data = EasyDict({
        "joint_names": humanoid_fk.joint_names,
        "body_names": humanoid_fk.body_names,
        "pose_aa": pose_aa_robot_new.squeeze().detach().numpy(),
        "root_trans": root_trans.squeeze().detach().numpy(),
        #"root_trans_offset": root_trans_offset,
        "root_trans_offset": root_trans.clone().detach(),
        "root_rot": sRot.from_rotvec(root_rot_new.detach().numpy()).as_quat(),
        "dof_pos": dof_pos_new.squeeze().detach().numpy(),
        "robot_joints": robot_joints.squeeze().detach().numpy(),
        "smpl_joints": smpl_joints.numpy(),
    })
    all_data.update(fk_return)
    

    return all_data

@hydra.main(version_base=None, config_path="./../data/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to process AMASS motion data and save the results.

    Args:
        cfg (DictConfig): Configuration object.
    """
    amass_root = cfg.get("amass_root", None)
    if not amass_root:
        raise ValueError("amass_root is not specified in the config")

    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    if not all_pkls:
        raise ValueError(f"No data found in {amass_root}")

    key_name_to_pkls = {
        "_".join(path.split("/")[len(amass_root.split("/")):]).replace(".npz", ""): path
        for path in all_pkls
    }
    key_names = list(key_name_to_pkls.keys())

    if cfg.get("motion_name", None):
        motion_name = cfg.motion_name
        if cfg.motion_name=="filter": # geting motion name from filter.yaml file
            motion_name = cfg.motion_filter.Walk
            key_names = motion_name
        else:
            key_names = [key for key in key_names if motion_name in key]
    
    if key_names ==[]:
        logger.error(f"Do not find the matched motion name {motion_name}")
    else:
        logger.info(f"Processing motions: {key_names}")

    all_data={}
    for key_name in key_names:
        data = process_motion(key_name, key_name_to_pkls, cfg)
        if data is not None:
            all_data[key_name] = data

    data_root = cfg.get("data_root", None)
    os.makedirs(f"{data_root}/motions/{cfg.robot.humanoid_type}/fit_motion", exist_ok=True)
    logger.info(f"saving fit data to {data_root}/motions/{cfg.robot.humanoid_type}/fit_motion/{cfg.motion_name}.pkl ...")
    joblib.dump(all_data, f"{data_root}/motions/{cfg.robot.humanoid_type}/fit_motion/{cfg.motion_name}.pkl")

    # saving npz for lus2_amp cfg
    for key_name, data in all_data.items():
        np.savez(
            f"{data_root}/motions/{cfg.robot.humanoid_type}/fit_motion/{key_name}.npz",
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
        logger.info(f"saving fit data to {data_root}/motions/{cfg.robot.humanoid_type}/fit_motion/{key_name}.npz ...")


    return 0


if __name__ == "__main__":
    main()
