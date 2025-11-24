from bonkbot.messages import StrategyOutput, JointTrajectory, PrehitPlan
from bonkbot.planning.prehit_planner import plan_prehit

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    RigidTransform,
    RotationMatrix,
    StartMeshcat,
    ConstantVectorSource,
    BasicVector,
    TrajectorySource,
)
from pydrake.trajectories import PiecewisePolynomial


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


def make_scripted_strategy(
    plant,
    station_context,
    iiwa_model,
    soup_models,
    hammer_face_frame,
) -> list[StrategyOutput]:
    """
    Offline 'strategy' for the test: go to prehit for each soup in order.
    """
    plant_context = plant.GetMyContextFromRoot(station_context)

    strategy_msgs: list[StrategyOutput] = []
    t = 0.0  # dummy timestamps just to populate the field for now

    for soup_id, soup_model in enumerate(soup_models):
        # 1) Compute prehit pose in world frame for this soup
        soup_body = plant.GetBodyByName("base_link_soup", soup_model)
        X_WSoup = plant.EvalBodyPoseInWorld(plant_context, soup_body)

        l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model)
        X_HL7 = plant.CalcRelativeTransform(plant_context, hammer_face_frame, l7_frame)

        X_WH_prehit, _ = get_prehit_pose(X_WSoup, X_HL7)

        # 2) Wrap that into a StrategyOutput
        msg = StrategyOutput(
            timestamp=t,
            mode="plan_to_prehit",
            target_mole_id=soup_id,
            target_prehit_pose_W=X_WH_prehit,
            post_hit_behavior="return_to_ready",
        )
        strategy_msgs.append(msg)
        t += 1.0   # arbitrary; not used by this test

    return strategy_msgs



def make_retreat_trajectory(
    q_start: np.ndarray,
    q_rest: np.ndarray,
    duration: float = 2.0,
) -> JointTrajectory:
    times = [0.0, duration]
    Q = np.column_stack([q_start, q_rest])
    q_traj = PiecewisePolynomial.FirstOrderHold(times, Q)
    return JointTrajectory(start_time=0, duration=duration, q_traj=q_traj)



def compile_strategy_sequence_to_segments(
    plant,
    strategy_msgs: list[StrategyOutput],
    q_rest: np.ndarray,
) -> list[JointTrajectory]:
    """
    For this test:
    - For each StrategyOutput(mode='plan_to_prehit'):
        1) Plan a prehit trajectory from q_current.
        2) Plan a retreat-to-rest trajectory afterward.
    """
    segments: list[JointTrajectory] = []
    q_current = q_rest.copy()

    for msg in strategy_msgs:
        if msg.mode != "plan_to_prehit":
            # For now, we just ignore other modes
            continue

        assert msg.target_mole_id is not None
        assert msg.target_prehit_pose_W is not None

        soup_id = msg.target_mole_id
        X_WH_prehit = msg.target_prehit_pose_W  # RigidTransform

        # Create Prehit plan
        prehit_plan: PrehitPlan = plan_prehit(plant, q_current, soup_id, X_WH_prehit)
        if prehit_plan is None:
            print(f"Planning failed for soup {soup_id}")
            continue

        segments.append(prehit_plan.traj)

        # Update q_current to end of prehit trajectory
        q_traj = prehit_plan.traj.q_traj # PiecewisePolynomial
        q_current = q_traj.value(q_traj.end_time()).ravel()

        # Add Post-hit behavior. For now, we always go back to ready/rest
        if msg.post_hit_behavior == "return_to_ready":
            retreat_traj = make_retreat_trajectory(q_start=q_current, q_rest=q_rest)
            segments.append(retreat_traj)
            q_current = q_rest.copy()

        # if 'chain_next_target', we would skip the retreat
    return segments


def concatenate_joint_trajectories(segments: list[JointTrajectory]) -> PiecewisePolynomial:
    """
    Assumes each q_traj starts at time 0; we re-time and stitch them.
    """
    if not segments:
        raise ValueError("No segments to concatenate.")

    knot_times: list[float] = []
    knot_values: list[np.ndarray] = []

    t_offset = 0.0

    for seg in segments:
        q_traj = seg.q_traj
        segment_breaks = q_traj.get_segment_times()
        for t_local in segment_breaks:
            t_global = t_offset + t_local
            q = q_traj.value(t_local).ravel()

            if knot_times and np.isclose(t_global, knot_times[-1]):
                knot_values[-1] = q
            else:
                knot_times.append(t_global)
                knot_values.append(q)

        t_offset += q_traj.end_time()

    Q = np.column_stack(knot_values)
    print(knot_times)
    return PiecewisePolynomial.FirstOrderHold(knot_times, Q)
