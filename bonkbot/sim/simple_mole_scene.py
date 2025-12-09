from pathlib import Path

def get_simple_mole_scene_yaml(hammer_file='hammer.sdf', mole_file='mole.sdf'):
    """
    Returns YAML string of 3 moles in a straight line in front of the iiwa with a hammer.
    """
    HAMMER_SDF_PATH = Path(__file__).parent / "assets" / hammer_file
    MOLE_SDF_PATH = Path(__file__).parent / "assets" / mole_file

    SIMPLE_MOLE_SCENE_YAML = f"""directives:
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
- add_model:
    name: mole_0
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: world
    child: mole_0::socket
    X_PC: {{translation: [.5, -.8, 0]}}
- add_model:
    name: mole_1
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: world
    child: mole_1::socket
    X_PC: {{translation: [0, -.8, 0]}}
- add_model:
    name: mole_2
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: world
    child: mole_2::socket
    X_PC: {{translation: [-.5, -.8, 0]}}

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
    mole_0: !JointStiffnessDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_1: !JointStiffnessDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    mole_2: !JointStiffnessDriver
        gains: 
            mole_slider: {{kp: 100, kd: 10}}
    """
    return SIMPLE_MOLE_SCENE_YAML