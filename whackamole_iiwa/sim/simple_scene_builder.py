from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat
)
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)

HAMMER_SDF_PATH = Path(__file__).parent / "assets" / "hammer.sdf"

scenario_yaml = f"""directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
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
    name: soup
    file: package://manipulation/hydro/005_tomato_soup_can.sdf
- add_weld:
    parent: world
    child: soup::base_link_soup
    X_PC:
        translation: [0, -0.7, 0.05]
        rotation: !Rpy {{ deg: [-90, 0, 0] }}
model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
"""

meshcat = StartMeshcat()
scenario = LoadScenario(data=scenario_yaml)
station = MakeHardwareStation(scenario, meshcat=meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)

diagram = builder.Build()

# Create a context for the diagram
root_context = diagram.CreateDefaultContext()
# Get plant context
plant = station.GetSubsystemByName("plant")
plant_context = plant.GetMyContextFromRoot(root_context)

# Get initial positions of the iiwa joints
iiwa = plant.GetModelInstanceByName("iiwa")
iiwa_q0 = plant.GetPositions(plant_context, iiwa)

# Fix the station input port "iiwa.position" to q0
station_context = station.GetMyMutableContextFromRoot(root_context)
station.GetInputPort("iiwa.position").FixValue(station_context, iiwa_q0)

hammer = plant.GetModelInstanceByName("iiwa")
soup = plant.GetModelInstanceByName("soup")

# Simulate
simulator = Simulator(diagram, root_context)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(5.0)  # run for 5 seconds (or longer)
input("Press Enter to exit...")  # keep Meshcat alive so you can look around