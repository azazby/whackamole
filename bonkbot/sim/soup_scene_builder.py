# sim/soup_scene_builder.py
from pathlib import Path
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)
from pydrake.all import (
    DiagramBuilder,
)

HAMMER_SDF_PATH = Path(__file__).parent / "assets" / "hammer.sdf"

# maybe store the YAML here too
SOUP_SCENE_YAML = f"""directives:
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
    name: soup_0
    file: package://manipulation/hydro/005_tomato_soup_can.sdf
- add_weld:
    parent: world
    child: soup_0::base_link_soup
    X_PC:
        translation: [.5, -.8, 0.1]
        rotation: !Rpy {{ deg: [-90, 0, 0] }}
- add_model:
    name: soup_1
    file: package://manipulation/hydro/005_tomato_soup_can.sdf
- add_weld:
    parent: world
    child: soup_1::base_link_soup
    X_PC:
        translation: [0, -.8, 0.1]
        rotation: !Rpy {{ deg: [-90, 0, 0] }}
- add_model:
    name: soup_2
    file: package://manipulation/hydro/005_tomato_soup_can.sdf
- add_weld:
    parent: world
    child: soup_2::base_link_soup
    X_PC:
        translation: [-.5, -.8, 0.1]
        rotation: !Rpy {{ deg: [-90, 0, 0] }}
model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
"""

def add_soup_scene(builder: DiagramBuilder, meshcat=None):
    """
    Given a DiagramBuilder, add the iiwa + 3 soup cans as a HardwareStation.

    Returns:
        station, plant, scene_graph, iiwa_model, soup_models
    """
    scenario = LoadScenario(data=SOUP_SCENE_YAML)
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)

    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    iiwa_model = plant.GetModelInstanceByName("iiwa")
    soup_models = [
        plant.GetModelInstanceByName(f"soup_{i}") for i in range(3)
    ]

    return station, plant, scene_graph, iiwa_model, soup_models
