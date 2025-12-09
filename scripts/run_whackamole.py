from pydrake.all import (
    DiagramBuilder, StartMeshcat, Simulator, MultibodyPlant, 
    AddMultibodyPlantSceneGraph, RigidTransform,
    LogVectorOutput, MeshcatVisualizer,
)
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)

from bonkbot.state_machine import BonkBotBrain
from bonkbot.hit_force_control import HammerContactForce
from bonkbot.gt_perception import DummyPerception
from bonkbot.sim.full_mole_scene import get_full_mole_scene_yaml
from bonkbot.mole_scheduler import MoleController


def run_whack_a_mole(mole_scene_yaml, num_moles, grid_size=None, 
                     switch_time_range=(1, 5), multi_mode=False, random_switch_time=False, debug=False):
    """
    grid_size for 3x3 grid is 3.
    """
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    
    scenario = LoadScenario(data=mole_scene_yaml)
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)

    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    iiwa_model = plant.GetModelInstanceByName("iiwa")
    mole_models = []
    if grid_size is not None:
        for i in range(grid_size):
            for j in range(grid_size):
                mole_models.append(plant.GetModelInstanceByName(f"mole_{i}_{j}"))
    else:
        for i in range(num_moles):
            mole_models.append(plant.GetModelInstanceByName(f"mole_{i}"))

    # Visualize axes (useful for debugging)
    # Get hammer head frame
    # hammer = plant.GetModelInstanceByName("hammer")
    # hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     frame=hammer_face_frame,
    #     length=0.1,
    # )
        
    # ----- ADD SYTEMS -----
    
    # BonkBotBrain, aka the FSMPlanner
    planner = builder.AddSystem(BonkBotBrain(plant, iiwa_model, mole_models, debug=debug))
    planner.set_name("bonkbot_brain")
    
    # Perception: DummyPerception for now
    perception = builder.AddSystem(DummyPerception(plant, mole_models))
    perception.set_name("perception")

    # Force Sensor
    hammer_body = plant.GetBodyByName("hammer_link", plant.GetModelInstanceByName("hammer"))
    force_sensor = builder.AddSystem(HammerContactForce(plant, hammer_body.index(), [0,0,-1]))

    # Mole Controller
    mole_controller = builder.AddSystem(MoleController(plant, hammer_body.index(), mole_models, 
                                        switch_time_range=switch_time_range, multi_mode=multi_mode, 
                                        random_switch_time=random_switch_time))
    mole_controller.set_name("mole_controller")

    # ----- WIRING -----
    
    # Robot State -> Planner
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), planner.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa.velocity_estimated"), planner.GetInputPort("iiwa_velocity"))

    # Plant State -> Perception
    builder.Connect(station.GetOutputPort("state"), perception.GetInputPort("plant_state"))

    # Station -> Force Sensor -> Planner
    builder.Connect(station.GetOutputPort("contact_results"), force_sensor.GetInputPort("contact_results"))
    builder.Connect(force_sensor.GetOutputPort("F_meas"), planner.GetInputPort("F_meas"))

    # Perception -> Planner
    builder.Connect(perception.GetOutputPort("mole_poses"), planner.GetInputPort("mole_poses"))

    # Planner -> Robot
    builder.Connect(planner.GetOutputPort("iiwa_position_command"),station.GetInputPort("iiwa.position"))

    # Game Controller -> Station
    if grid_size is not None:
        n = 0
        for i in range(grid_size):
            for j in range(grid_size):
                builder.Connect(
                    mole_controller.GetOutputPort(f"mole_{n}_setpoint"),
                    station.GetInputPort(f"mole_{i}_{j}.desired_state")
                )
                n += 1
    else:
        for i in range(num_moles):
            builder.Connect(
                mole_controller.GetOutputPort(f"mole_{i}_setpoint"),
                station.GetInputPort(f"mole_{i}.desired_state")
            )
        
    # Station -> Game Controller
    builder.Connect(station.GetOutputPort("contact_results"), mole_controller.GetInputPort("contact_results"))
    builder.Connect(station.GetOutputPort("state"), mole_controller.GetInputPort("plant_state"))

    # ----- VISUALIZATION -----
    diagram = builder.Build()
    # RenderDiagram(diagram, max_depth=1)
    simulator = Simulator(diagram)

    # Run Simulation
    simulator.set_target_realtime_rate(1.0)
    meshcat.AddButton("Stop Simulation", "Escape")
    meshcat.StartRecording()
    print("Starting simulation...")
    dt = 0.2    
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        context = simulator.get_context()
        simulator.AdvanceTo(context.get_time() + dt)
    print("Stopping simulation and finalizing recording...")
    meshcat.StopRecording()
    meshcat.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")
    print("Done.")


if __name__ == "__main__":
    full_mole_scene_yaml = get_full_mole_scene_yaml()
    run_whack_a_mole(full_mole_scene_yaml, num_moles=9, grid_size=3, 
                     switch_time_range=(0.5, 6), multi_mode=False, random_switch_time=True,
                     debug=False)