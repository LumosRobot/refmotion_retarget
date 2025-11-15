import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "../../../data/cfg/robot/ik_configs"
ASSET_ROOT = HERE / "../../.." / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
    "lumos_lus2": ASSET_ROOT / "lumos_lus2" / "mjcf/lus2.xml",
}

IK_CONFIG_DICT = {
    # offline data
    "smplx":{
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "lumos_lus2": IK_CONFIG_ROOT / "smplx_to_lus2.json",
    },
    "bvh_lafan1":{
        "unitree_g1": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",
        "unitree_g1_with_hands": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",
        "booster_t1_29dof": IK_CONFIG_ROOT / "bvh_lafan1_to_t1_29dof.json",
        "fourier_n1": IK_CONFIG_ROOT / "bvh_lafan1_to_n1.json",
        "stanford_toddy": IK_CONFIG_ROOT / "bvh_lafan1_to_toddy.json",
        "engineai_pm01": IK_CONFIG_ROOT / "bvh_lafan1_to_pm01.json",
        "pal_talos": IK_CONFIG_ROOT / "bvh_to_talos.json",
    },
    "bvh_nokov":{
        "unitree_g1": IK_CONFIG_ROOT / "bvh_nokov_to_g1.json",
    },
    "fbx":{
        "unitree_g1": IK_CONFIG_ROOT / "fbx_to_g1.json",
        "unitree_g1_with_hands": IK_CONFIG_ROOT / "fbx_to_g1.json",
    },
    "fbx_offline":{
        "unitree_g1": IK_CONFIG_ROOT / "fbx_offline_to_g1.json",
    },
}


ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "lumos_lus2": "pelvis",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "lumos_lus2": 2.0,
}
