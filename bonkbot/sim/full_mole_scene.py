from pathlib import Path

def get_full_mole_scene_yaml(hammer_file='hammer.sdf', mole_file='fancy_mole.sdf', grid_file='grid.sdf'):
    """
    Returns YAML string of 9x9 grid of moles with an iiwa with a hammer.
    """
    HAMMER_SDF_PATH = Path(__file__).parent / "assets" / hammer_file
    MOLE_SDF_PATH = Path(__file__).parent / "assets" / mole_file
    GRID_SDF_PATH = Path(__file__).parent / "assets" / grid_file

    MOLE_SCENE_YAML = f"""directives:

# ===============================================================
# IIWA
# ===============================================================
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0

# ===============================================================
# Hammer
# ===============================================================
- add_model:
    name: hammer
    file: file://{HAMMER_SDF_PATH}
    default_free_body_pose:
        hammer_link:
            translation: [0, 0, 0]
            rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_weld:
    parent: iiwa::iiwa_link_7
    child: hammer::hammer_link
    X_PC:
        translation: [0, 0, 0.06]
        rotation: !Rpy {{deg: [0, -90, 0] }}

# ===============================================================
# Grid Board + Moles
# ===============================================================
- add_model:
    name: grid_board
    file: file://{GRID_SDF_PATH}

- add_weld:
    parent: world
    child: grid_board::board
    X_PC:
        translation: [0, -0.75, 0]
        rotation: !Rpy {{ deg: [0, 0, 0] }}

# 3x3 mole grid
- add_model:
    name: mole_0_0
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_0_0::socket
    X_PC: {{translation: [-0.2, -0.2, 0.0125]}}

- add_model:
    name: mole_0_1
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_0_1::socket
    X_PC: {{translation: [0.0, -0.2, 0.0125]}}

- add_model:
    name: mole_0_2
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_0_2::socket
    X_PC: {{translation: [0.2, -0.2, 0.0125]}}

- add_model:
    name: mole_1_0
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_1_0::socket
    X_PC: {{translation: [-0.2, 0.0, 0.0125]}}

- add_model:
    name: mole_1_1
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_1_1::socket
    X_PC: {{translation: [0.0, 0.0, 0.0125]}}

- add_model:
    name: mole_1_2
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_1_2::socket
    X_PC: {{translation: [0.2, 0.0, 0.0125]}}

- add_model:
    name: mole_2_0
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_2_0::socket
    X_PC: {{translation: [-0.2, 0.2, 0.0125]}}

- add_model:
    name: mole_2_1
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_2_1::socket
    X_PC: {{translation: [0.0, 0.2, 0.0125]}}

- add_model:
    name: mole_2_2
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_2_2::socket
    X_PC: {{translation: [0.2, 0.2, 0.0125]}}

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
    mole_0_0: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_0_1: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_0_2: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_1_0: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_1_1: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_1_2: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_2_0: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_2_1: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_2_2: !InverseDynamicsDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    """
    return MOLE_SCENE_YAML