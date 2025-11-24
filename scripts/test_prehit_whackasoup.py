# scripts/test_prehit_whackasoup.py
from bonkbot.sim.soup_scene_builder import add_soup_scene
from bonkbot.planning.prehit_planner import plan_prehit
from bonkbot.strategy.scripted_strategy import (make_scripted_strategy,
                                                compile_strategy_sequence_to_segments,
                                                concatenate_joint_trajectories)

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    ConstantVectorSource,
    TrajectorySource
)
from manipulation.meshcat_utils import AddMeshcatTriad


def main():
    meshcat = StartMeshcat()

    # Create builder
    builder = DiagramBuilder()

    # Add scene (iiwa + 3 soups)
    station, plant, scene_graph, iiwa_model, soup_models = add_soup_scene(
        builder, meshcat=meshcat
    )
    context = station.CreateDefaultContext()
    plant_context = station.GetSubsystemContext(plant, context)

    # Get initial positions of the iiwa joints
    iiwa_q0 = plant.GetPositions(plant_context, iiwa_model)
    q_rest = iiwa_q0.copy()  # rest position to return to after each hit
    # default set iiwa joint input to initial position for now
    # iiwa_src = builder.AddSystem(ConstantVectorSource(iiwa_q0))
    # builder.Connect(iiwa_src.get_output_port(), station.GetInputPort("iiwa.position"))

    # Get hammer head frame
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)

    # Get pose of iiwa link 7 in hammer face frame
    l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model)
    X_HL7 = plant.CalcRelativeTransform(plant_context, hammer_face_frame, l7_frame)
    
    # Create scripted StrategyOutput sequence for all soups
    strategy_msgs = make_scripted_strategy(
        plant=plant,
        station_context=context,
        iiwa_model=iiwa_model,
        soup_models=soup_models,
        hammer_face_frame=hammer_face_frame,
    )

    # Convert StrategyOutputs -> list of JointTrajectory segments
    segments = compile_strategy_sequence_to_segments(
        plant=plant,
        strategy_msgs=strategy_msgs,
        q_rest=q_rest,
    )

    # Concatenate segments into a single trajectory
    q_traj_global = concatenate_joint_trajectories(segments)

    # Set up trajectory source to provide the joint trajectory
    iiwa_src = builder.AddSystem(TrajectorySource(q_traj_global))
    builder.Connect(iiwa_src.get_output_port(), station.GetInputPort("iiwa.position"))

    # Build the diagram 
    diagram = builder.Build()

    simulator = Simulator(diagram)

    # Run simulation
    simulator.set_target_realtime_rate(1.0)
    # simulator.AdvanceTo(5.0)  # run for 5 seconds (or longer)

    # Record simulation
    meshcat.StartRecording()
    simulator.AdvanceTo(q_traj_global.end_time())
    meshcat.StopRecording()
    meshcat.PublishRecording()

    input("Press Enter to exit...")  # keep Meshcat alive so you can look around


if __name__ == "__main__":
    main()