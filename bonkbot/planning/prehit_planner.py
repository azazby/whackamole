# planning/prehit_planner.py

from dataclasses import dataclass
import numpy as np

from pydrake.math import RigidTransform
from pydrake.multibody.plant import MultibodyPlant

from pydrake.multibody import inverse_kinematics
from pydrake.all import (
    PiecewisePolynomial,
    RotationMatrix,
    Solve,
    SolutionResult
)
from bonkbot.messages import PrehitPlan, JointTrajectory, JointVector, Time

def solve_ik(plant: MultibodyPlant, 
             plant_context, 
             X_WH:RigidTransform, 
             q_guess: JointVector=None, 
             pos_tol: float=1e-3, theta_bound: float=1e-2) -> tuple[JointVector, bool]:
    """
    Solve IK to find iiwa joint configuration that achieves the desired hammer face pose X_WH.
    """
    # Get the hammer face frame 
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_face_frame = plant.GetFrameByName("hammer_face",hammer)
    W = plant.world_frame()

    if q_guess is None:
        iiwa = plant.GetModelInstanceByName("iiwa")
        iiwa_q0 = plant.GetPositions(plant_context, iiwa)

    ik = inverse_kinematics.InverseKinematics(plant, plant_context)

    # Extract desired position/orientation from X_WH
    R_WH_desired = X_WH.rotation()
    p_WH_desired = X_WH.translation()

    # Position constraint: place origin of hammer face at p_WH_desired (within a small box)
    ik.AddPositionConstraint(
        frameA=W,
        p_BQ=np.zeros(3),
        frameB=hammer_face_frame,
        p_AQ_lower=p_WH_desired - pos_tol,
        p_AQ_upper=p_WH_desired + pos_tol,
    )

    # Orientation constraint: align world and hammer_face_frame up to some tolerance
    ik.AddOrientationConstraint(
        frameAbar=W,
        R_AbarA=RotationMatrix(),
        frameBbar=hammer_face_frame,
        R_BbarB=R_WH_desired,
        theta_bound=theta_bound,
    )

    # Add collision constraint to ensure collision free pre-hit pose
    # ik.AddMinimumDistanceConstraint(0.02)

    prog = ik.prog()
    prog.SetInitialGuess(ik.q(), q_guess)
    result = Solve(prog)

    if result.get_solution_result() != SolutionResult.kSolutionFound:
        return result.GetSolution(ik.q()), False
    return result.GetSolution(ik.q()), True


def make_joint_space_position_trajectory(path, timestep=1.0) -> PiecewisePolynomial:
    """
    Given a list of joint configurations, create a PiecewisePolynomial trajectory that linearly interpolates between them.
    """
    traj = None
    times = [timestep * i for i in range(len(path))]
    Q = np.column_stack(path)
    traj = PiecewisePolynomial.FirstOrderHold(times, Q)
    return traj


def plan_prehit(
    plant: MultibodyPlant,
    q_current: JointVector,
    target_mole_id: int,
    target_prehit_pose_W: RigidTransform,
) -> PrehitPlan:
    """
    Given current joint config + desired prehit hammer pose,
    return a time-parameterized joint trajectory to get there.
    """    
    # Solve IK for the target prehit pose q_goal
    q_goal, success = solve_ik(plant, plant.CreateDefaultContext(), target_prehit_pose_W, q_guess=q_current)
    if not success:
        print(f"IK failed to find a solution for target mole {target_mole_id}")
        return None

    # Make linear joint space position trajectory from q_current to q_goal
    path = [q_current, q_goal]
    traj = make_joint_space_position_trajectory(path)

    # Make PrehitPlan message
    prehit_plan = PrehitPlan(
        target_mole_id=target_mole_id,
        traj=JointTrajectory(
            q_traj=traj,
            start_time=Time(0.0),
            duration=Time(traj.end_time()),
        ),
        q_goal=q_goal,
        target_prehit_pose_W=target_prehit_pose_W,
    )

    return prehit_plan