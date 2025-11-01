import os
import glob
import sys
import time
import argparse
import json
import os.path as osp
import scipy.ndimage.filters as filters

sys.path.append(os.getcwd())
import torch
import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict
import hydra

# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger("vis_mj")


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        logger.info("Reset")
        time_step = 0
        motion_id = 0
    elif chr(keycode) == " ":
        logger.info("Paused")
        paused = not paused
    elif chr(keycode) == "T":
        logger.info("next")
        motion_id += 1
        motion_id = motion_id% len(motion_data_keys)
        curr_motion_key = motion_data_keys[motion_id]
        logger.info(curr_motion_key)
    #else:
    #    logger.info("not mapped", chr(keycode))
    
    


def save_motion_example(curr_motion, curr_motion_key, motion_file, joint_names, body_names, robot_body_names):

    def safe_get(name, fallback_shape=None):
        if name in curr_motion:
            if isinstance(curr_motion[name], torch.Tensor):
                return curr_motion[name].clone().detach().numpy()
            else:
                return curr_motion[name]
        elif fallback_shape:
            logger.warning(f"'{name}' not found in curr_motion — filling with zeros")
            return np.zeros(fallback_shape, dtype=np.float32)
        else:
            raise ValueError(f"'{name}' missing and fallback_shape is None")


    root_trans = curr_motion['root_trans'] if "root_trans" in curr_motion.keys() else curr_motion['root_trans_offset']
    root_rot = curr_motion["root_rot"]
    dof_pos = curr_motion["dof_pos"] if "dof_pos" in curr_motion.keys() else curr_motion["dof"]
    smpl_joints = curr_motion['smpl_joints']
    fps = curr_motion["fps"]

    num_frames = len(dof_pos)

    # i) Velocities
    root_lin_vel_w = safe_get("global_velocity", (num_frames, 1, 3))[:, 0, :]
    root_ang_vel_w = safe_get("global_angular_velocity", (num_frames, 1, 3))[:, 0, :]

    # Inverse root rotation
    rot_inv = sRot.inv(sRot.from_quat(root_rot))
    root_lin_vel_b = rot_inv.apply(root_lin_vel_w)
    root_ang_vel_b = rot_inv.apply(root_ang_vel_w)

    # ii) Robot body positions and velocities
    robot_bodies = safe_get("robot_joints", (num_frames, robot_body_names, 3))
    body_pos_w = robot_bodies  # (T, N, 3)
    body_vel_w = np.gradient(body_pos_w, axis=0)*fps
    #body_vel_w = filters.gaussian_filter1d(body_vel_w, 0.3, axis=0, mode="nearest")

    body_pos_rel = body_pos_w - root_trans[:, None, :]
    body_pos_b = np.stack([r.apply(p) for r, p in zip(rot_inv, body_pos_rel)], axis=0)

    body_vel_rel = body_vel_w - root_lin_vel_w[:, None, :]
    body_vel_b = np.stack([r.apply(v) for r, v in zip(rot_inv, body_vel_rel)], axis=0)

    # iii) dof velocity
    dof_vel = np.gradient(dof_pos, axis=0)*fps
    #dof_vel = filters.gaussian_filter1d(dof_vel, 0.3, axis=0, mode="nearest")

    # Concatenate features
    all_data = np.concatenate((
        root_trans,
        root_rot,
        root_lin_vel_w,
        root_ang_vel_w,
        root_lin_vel_b,
        root_ang_vel_b,
        dof_pos,
        dof_vel,
        body_pos_w.reshape(body_pos_w.shape[0], -1),
        body_vel_w.reshape(body_vel_w.shape[0], -1),
        body_pos_b.reshape(body_pos_b.shape[0], -1),
        body_vel_b.reshape(body_vel_b.shape[0], -1)
    ), axis=-1)

    #if all_data.shape[0] < 100:
    #    logger.info("Frame count < 100, skipping save")
    #    return

    # Field names
    root_trans_key = ["root_pos_x", "root_pos_y", "root_pos_z"]
    root_rot_key = ["root_rot_x", "root_rot_y", "root_rot_z", "root_rot_w"]
    root_w_vel_key = ["root_vel_x_w", "root_vel_y_w", "root_vel_z_w",
                      "root_ang_vel_x_w", "root_ang_vel_y_w", "root_ang_vel_z_w"]
    root_b_vel_key = ["root_vel_x_b", "root_vel_y_b", "root_vel_z_b",
                      "root_ang_vel_x_b", "root_ang_vel_y_b", "root_ang_vel_z_b"]
    dof_keys = [f"{name}_dof_pos" for name in joint_names] + [f"{name}_dof_vel" for name in joint_names]

    body_keys = [f"{body}_{label}" for body in robot_body_names for label in ["pos_x_w", "pos_y_w", "pos_z_w"]]
    body_keys += [f"{body}_{label}" for body in robot_body_names for label in ["vel_x_w", "vel_y_w", "vel_z_w"]]
    body_keys += [f"{body}_{label}" for body in robot_body_names for label in ["pos_x_b", "pos_y_b", "pos_z_b"]]
    body_keys += [f"{body}_{label}" for body in robot_body_names for label in ["vel_x_b", "vel_y_b", "vel_z_b"]]

    data_fields = root_trans_key + root_rot_key + root_w_vel_key + root_b_vel_key + dof_keys + body_keys

    assert all_data.shape[1] == len(data_fields), \
        f"Mismatch: data columns {all_data.shape[1]} != fields {len(data_fields)}"

    saved_data = {
        "LoopMode": "Wrap",
        "FrameDuration": 1.0 / fps,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
        "Fields": data_fields,
        "Frames": all_data
    }

    # Save
    saving_folder = os.path.join(os.path.dirname(os.path.dirname(motion_file)), "pkl")
    os.makedirs(saving_folder, exist_ok=True)
    save_path = os.path.join(saving_folder, f"{curr_motion_key+'_fps'+str(fps)}.pkl")
    joblib.dump({curr_motion_key: saved_data}, save_path)
    logger.info(f"✅ Saved motion with shape {all_data.shape} and fps: {fps}: {curr_motion_key} → {save_path}")


@hydra.main(version_base=None, config_path="./../data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    device = torch.device("cpu")
    humanoid_xml = os.path.join(cfg.robot.asset.assetRoot, cfg.robot.asset.assetFileName)
    logger.info(f"humanoid xlm is {humanoid_xml}")
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    joint_names = [mj_model.joint(i).name for i in range(1, mj_model.njnt)]
    curr_start, num_motions, motion_id, motion_acc, time_step, paused = 0, 1, 0, set(), 0, False
    
    # geting all key names 
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
        else:
            motion_name = [key for key in key_names if motion_name in key]

        logger.info(f"motion name is {motion_name}")
        motion_file = f"{cfg.data_root}/motions/{cfg.robot.humanoid_type}/fit_motion/{cfg.motion_name}.pkl"

    logger.info(f"Motion file is: {motion_file}")
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())

    logger.info(f"motion keys: {motion_data_keys}")

    # move robot to origin location of env coordination
    logger.info(f"📌 Move the robot to env origin location!")
    for key, data in motion_data.items():

        # reset trans_offset to original point
        init_trans = motion_data[key]["root_trans_offset"][0,:]
        motion_data[key]["root_trans_offset"][:,:2]-=init_trans[:2]
        #motion_data[key]["fps"] = 33
        fps = motion_data[key]["fps"]
        logger.info(f"data name: {key}, fps: {fps}")

        # update fk
        if "global_translation_extend" not in motion_data[key]:
            logger.info("[update_fk] Running humanoid forward kinematics...")
            humanoid_fk = Humanoid_Batch(cfg.robot)  # Load forward kinematics model
            trans = torch.from_numpy(motion_data[key]['root_trans_offset']).float()
            pose_aa = torch.from_numpy(motion_data[key]['pose_aa']).float() # shape: frame num, 74 (pose params theta: 23+3+3)
            fk_return = humanoid_fk.fk_batch(pose_aa[None,:], trans[None,:],dt=1.0/fps, return_full=True) # batch of frame_num

            # update robot joints(keypoint) link position 
            num_augment_joint = len(cfg.robot.extend_config)
            robot_body_names = [i[0] for i in cfg.robot.joint_matches]
            robot_joint_names_augment = humanoid_fk.body_names_augment
            robot_body_names_idx = [robot_joint_names_augment.index(j) for j in robot_body_names]
            if num_augment_joint > 0:
                robot_joints = fk_return.global_translation_extend[:, :, robot_body_names_idx]
            else:
                robot_joints = fk_return.global_translation[:, :, robot_body_names_idx]
            motion_data[key]["robot_joints"] = robot_joints.squeeze().detach()

            fk_return = EasyDict({key: value if key=="fps" else value.clone().detach().squeeze()  for key, value in fk_return.items()})
            motion_data[key].update(fk_return)

            # calibrate robot height
            combined_mesh = humanoid_fk.mesh_fk(pose_aa[None, :1].detach(), trans[None, :1].detach())
            height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
            motion_data[key]["root_trans_offset"][..., 2] -= height_diff
            motion_data[key]["smpl_joints"][..., 2] -= height_diff
            if "robot_joints" in motion_data[key].keys():
                motion_data[key]["robot_joints"][..., 2] -= height_diff

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back, show_left_ui=False, show_right_ui=False) as viewer:
        cam = viewer.cam
        cam.distance = 4.0 ;cam.azimuth = 135; cam.elevation = -10; cam.lookat = [0,0,0]
        cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING;cam.trackbodyid=1;

        # adding gemo to dislay key joint position
        for _ in range(20):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.04, np.array([1, 0, 0, 1]))
        for _ in range(20):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.04, np.array([0, 1, 0, 1]))

        # Close the viewer automatically after 30 wall-seconds.
        curr_motion_key = motion_data_keys[motion_id]
        curr_motion = motion_data[curr_motion_key]
        frame_num = curr_motion['pose_aa'].shape[0]
        fps = curr_motion["fps"]
        logger.info(f"curr_motion_key: {curr_motion_key}, fps: {fps}")
        #import pdb;pdb.set_trace()

        while viewer.is_running():
            cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING
            step_start = time.time()
            root_trans = curr_motion['root_trans'] if "root_trans" in curr_motion.keys() else curr_motion['root_trans_offset']
            root_rot = curr_motion["root_rot"]
            dof_pos = curr_motion["dof_pos"]
            smpl_joints = curr_motion['smpl_joints']
            robot_bodies = curr_motion['robot_joints'] if "robot_joints" in curr_motion.keys() else None
            joint_names = [mj_model.joint(i).name for i in range(1, mj_model.njnt)]
            body_names = [mj_model.body(i).name for i in range(1, mj_model.nbody)]

            dt = 1.0/fps
            curr_frame = int(time_step/dt) % frame_num

            mj_data.qpos[:3] = root_trans[curr_frame]
            mj_data.qpos[3:7] = root_rot[curr_frame][[3, 0, 1, 2]]
            mj_data.qpos[7:] = dof_pos[curr_frame]
        
            mujoco.mj_forward(mj_model, mj_data)

            if paused:
                logger.info(f"Finish and saving data, the total frame is: {curr_frame}")
            else:
                time_step += dt
            
            # visualizing smpl joints
            if True:
                for i in range(smpl_joints.shape[1]):
                    viewer.user_scn.geoms[i].pos = smpl_joints[curr_frame, i]

            # visualizing robot joints
            # get robot joint index
            robot_body_names = [i[0] for i in cfg.robot.joint_matches]
            if robot_bodies is not None:
                for idx in range(len(robot_body_names)):
                    viewer.user_scn.geoms[20+idx].pos = robot_bodies[curr_frame, idx]
            
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            ################################# SAVE DATA ###########################################
            if curr_frame == frame_num -1:
                save_motion_example(curr_motion, curr_motion_key, motion_file, joint_names, body_names,robot_body_names)
                motion_id = (motion_id + 1) % len(motion_data_keys)
                time_step = 0
                logger.info(f"Finish and saving data, the total frame is: {curr_frame}")
                logger.info(f"Waiting 5 seconds for loading next motion key")
                time.sleep(5)# waiting for 5 seconds


            ############################################################################

if __name__ == "__main__":
    main()
