# Importing necessary libraries
import glob  # For file path pattern matching
import os  # For interacting with the operating system
import sys  # For system-specific parameters and functions
import pdb  # For debugging
import os.path as osp  # For path manipulations
sys.path.append(os.getcwd())  # Adding the current working directory to the system path

import torch  # PyTorch for tensor operations
from scipy.spatial.transform import Rotation as sRot  # For rotation transformations
import numpy as np  # For numerical operations
import joblib  # For saving and loading Python objects
from tqdm import tqdm  # For progress bars
import argparse  # For parsing command-line arguments
import cv2  # OpenCV for image processing (not used in this script)
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState  # Skeleton-related utilities
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES  # SMPL joint and bone names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot  # SMPL robot model

# Main function
if __name__ == "__main__":
    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)  # Debug mode flag
    parser.add_argument("--path", type=str, default="")  # Path to AMASS data
    args = parser.parse_args()
    
    # Configuration and initialization
    process_split = "train"  # Data split to process (train, test, or validation)
    upright_start = False  # Whether to start in an upright position
    robot_cfg = {  # Configuration for the SMPL robot
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, 
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False, 
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smpl",
    }

    # Initialize the SMPL robot
    smpl_local_robot = LocalRobot(robot_cfg)

    # Check if the provided path exists
    if not osp.isdir(args.path):
        print("Please specify AMASS data path")
        import ipdb; ipdb.set_trace()  # Debugging breakpoint
        
    # Collect all .npz files recursively from the specified path
    all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)

    # Load occlusion data for AMASS
    amass_occlusion = joblib.load("sample_data/amass_copycat_occlusion_v3.pkl")
    amass_full_motion_dict = {}  # Dictionary to store processed motion data

    # Define data splits for AMASS
    amass_splits = {
        'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT',  'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]
    }
    process_set = amass_splits[process_split]  # Select the appropriate split
    length_acc = []  # Accumulator for lengths of processed sequences

    # Process each .npz file
    for data_path in tqdm(all_pkls):
        bound = 0  # Initialize bound for sequence length
        splits = data_path.split("/")[7:]  # Extract relevant parts of the file path
        key_name_dump = "0-" + "_".join(splits).replace(".npz", "")  # Generate a unique key for the file
        
        # Skip files not in the selected split
        if (not splits[0] in process_set):
            continue
        
        # Handle occlusion issues
        if key_name_dump in amass_occlusion:
            issue = amass_occlusion[key_name_dump]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
                bound = amass_occlusion[key_name_dump]["idxes"][0]  # Adjust bound based on occlusion
                if bound < 10:
                    print("bound too small", key_name_dump, bound)
                    continue
            else:
                print("issue irrecoverable", key_name_dump, issue)
                continue
            
        # Load motion data from the .npz file
        entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
        
        # Skip files without a framerate
        if not 'mocap_framerate' in entry_data:
            continue
        framerate = entry_data['mocap_framerate']

        # Special case for a specific file
        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2
        
        # Downsample the data to 30 FPS
        skip = int(framerate / 30)
        root_trans = entry_data['trans'][::skip, :]
        pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
        betas = entry_data['betas']
        gender = entry_data['gender']
        N = pose_aa.shape[0]
        
        # Adjust bound if necessary
        if bound == 0:
            bound = N
            
        root_trans = root_trans[:bound]
        pose_aa = pose_aa[:bound]
        N = pose_aa.shape[0]
        if N < 10:
            continue
    
        # Map SMPL joints to MuJoCo joints
        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

        # Initialize beta and gender
        beta = np.zeros((16))
        gender_number, beta[:], gender = [0], 0, "neutral"
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        # Create a new skeleton state
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True
        )
        
        # Adjust for upright start if necessary
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()

        # Extract global and local rotations
        pose_quat_global = new_sk_state.global_rotation.numpy()
        pose_quat = new_sk_state.local_rotation.numpy()
        fps = 30

        # Store processed motion data
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

        amass_full_motion_dict[key_name_dump] = new_motion_out
        
    # Save the processed data
    import ipdb; ipdb.set_trace()
    if upright_start:
        joblib.dump(amass_full_motion_dict, "data/amass/amass_train_take6_upright.pkl", compress=True)
    else:
        joblib.dump(amass_full_motion_dict, "data/amass/amass_train_take6.pkl", compress=True)