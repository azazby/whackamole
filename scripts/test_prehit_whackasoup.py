# scripts/test_prehit_whackasoup.py
from bonkbot.sim.soup_scene_builder import add_soup_scene
from bonkbot.planning.prehit_planner import plan_prehit
from bonkbot.messages import JointTrajectory, PrehitPlan

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    RigidTransform,
    RotationMatrix,
    StartMeshcat,
    ConstantVectorSource,
    BasicVector,
    TrajectorySource

)
from manipulation.meshcat_utils import AddMeshcatTriad


def get_prehit_pose(X_WO: RigidTransform, 
                    X_HL7: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    """
    Given the object pose X_WO, compute a prehit pose in hammer face frame and iiwa link 7 frame.
    
    Parameters:
        X_WO: pose of target object in world frame
        X_HL7: pose of iiwa link 7 in hammer face frame
    """
    p_OH = np.array([0.0, -0.2, 0.0])
    R_OH = RotationMatrix.MakeYRotation(-np.pi/2) @ RotationMatrix.MakeXRotation(-np.pi/2)
    X_OH = RigidTransform(R_OH, p_OH) # pose of hammer face from object frame
    X_WH_prehit = X_WO @ X_OH # prehit pose of hammer face 
    X_WL7_prehit = X_WH_prehit @ X_HL7
    return X_WH_prehit, X_WL7_prehit


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
    # default set iiwa joint input to initial position for now
    # iiwa_src = builder.AddSystem(ConstantVectorSource(iiwa_q0))
    # builder.Connect(iiwa_src.get_output_port(), station.GetInputPort("iiwa.position"))

    # Choose which soup to hit (hardcode for now)
    SOUP_ID = 2
    soup_model = soup_models[SOUP_ID]
    soup_body = plant.GetBodyByName("base_link_soup", soup_model)
    X_WSoup = plant.EvalBodyPoseInWorld(plant_context, soup_body)

    # Get hammer head frame
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)

    # Get pose of iiwa link 7 in hammer face frame
    l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model)
    X_HL7 = plant.CalcRelativeTransform(plant_context, hammer_face_frame, l7_frame)
    
    # Get pre-hit pose frame
    X_WH_prehit, X_WL7_prehit = get_prehit_pose(X_WSoup, X_HL7)
    # Visualize target prehit frames
    AddMeshcatTriad(
        meshcat,
        path="hammer_prehit_pose_triad",  
        length=0.1,
        radius=0.005,
        X_PT=X_WH_prehit,                 
    )

    # Create prehit plan
    prehit_plan = plan_prehit(plant, iiwa_q0, SOUP_ID, X_WH_prehit)
    if prehit_plan is None:
        print("Planning failed!")
        return
    joint_traj = prehit_plan.traj # JointTrajectory object
    q_traj = joint_traj.q_traj  # PiecewisePolynomial object

    # Set up trajectory source to provide the joint trajectory
    iiwa_src = builder.AddSystem(TrajectorySource(q_traj))
    builder.Connect(iiwa_src.get_output_port(), station.GetInputPort("iiwa.position"))

    # Build the diagram 
    diagram = builder.Build()

    simulator = Simulator(diagram)

    # Run simulation
    simulator.set_target_realtime_rate(1.0)
    # simulator.AdvanceTo(5.0)  # run for 5 seconds (or longer)
    # input("Press Enter to exit...")  # keep Meshcat alive so you can look around

    # Record simulation
    meshcat.StartRecording()
    simulator.AdvanceTo(q_traj.end_time())
    meshcat.StopRecording()
    meshcat.PublishRecording()

    input("Press Enter to exit...")  # keep Meshcat alive so you can look around



if __name__ == "__main__":
    main()