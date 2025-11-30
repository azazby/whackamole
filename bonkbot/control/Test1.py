# admittance_call.py
import numpy as np

from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform

from manipulation.scenarios import LoadScenario, MakeHardwareStation

from whack_force_admittance import (
    AdmittanceParams,
    run_hit_experiment,
)

# ----------------------------------------------------------------------
# Project-specific helpers you already have in the notebook
# (you may want to move them into their own module)
# ----------------------------------------------------------------------

def get_prehit_pose(X_WMole: RigidTransform, X_HL7: RigidTransform):
    """
    Your existing function that, given:
      - X_WMole: pose of the mole/soup in world
      - X_HL7:   pose of iiwa link 7 in hammer-face frame
    returns:
      - X_WH_prehit: hammer-face pre-hit pose in world
      - X_WL7_prehit: link-7 pre-hit pose in world

    Here we just assume it exists already and import/use it.
    """
    # TODO: replace with actual implementation or import
    raise NotImplementedError("Use your real get_prehit_pose here.")


def solve_ik_for_prehit(plant, plant_context, hammer_face_frame, X_WH_prehit, iiwa_model_instance, q_guess):
    """
    Wrap your existing solve_ik(...) call for the pre-hit hammer pose.
    Returns q_goal (7-dim pre-hit joint config).
    """
    # TODO: replace with your real IK function.
    # Example of calling your existing solve_ik:
    # q_goal, optimal = solve_ik(
    #     plant, plant_context,
    #     X_WH_prehit,
    #     q_guess=q_guess,
    #     pos_tol=1e-6,
    #     theta_bound=1e-3,
    # )
    # return q_goal

    raise NotImplementedError("Use your real solve_ik here.")


def get_mole_pose_in_world(plant, plant_context, mole_index):
    """
    Return X_WMole for the selected mole (1..9).
    You likely already have body/frame names for each mole
    in your scenario YAML.

    Example: if your moles are models named 'mole_1', ..., 'mole_9'.
    """
    mole_name = f"mole_{mole_index}"
    mole_model = plant.GetModelInstanceByName(mole_name)
    mole_body = plant.GetBodyByName("base_link_mole", mole_model)
    X_WMole = plant.EvalBodyPoseInWorld(plant_context, mole_body)
    return X_WMole


def compute_X_HL7(plant, plant_context, hammer_face_frame, iiwa_model_instance):
    """
    Compute X_HL7 = pose of iiwa link 7 in hammer-face frame.
    You already did this in the notebook as:

        l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
        X_HL7 = plant.CalcRelativeTransform(context, hammer_face_frame, l7_frame)

    Here we just wrap it.
    """
    l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model_instance)
    X_HL7 = plant.CalcRelativeTransform(
        plant_context,
        hammer_face_frame,
        l7_frame,
    )
    return X_HL7

# ----------------------------------------------------------------------
# Main entry: run admittance hit on any mole
# ----------------------------------------------------------------------

def run_admittance_hit_on_mole(
    scenario_yaml,
    mole_index: int,
    F_des: float = 200.0,
    T_final: float = 8.0,
    plot: bool = True,
):
    """
    High-level function: given a mole index (1..9), run a full
    admittance-based hit experiment on that mole.

    This is the function your whack-a-mole supervisor could call
    whenever it decides “hit mole k now”.
    """

    # 1. Build station / plant / builder
    scenario = LoadScenario(data=scenario_yaml)
    station = MakeHardwareStation(scenario, meshcat=None)  # or pass meshcat
    builder = DiagramBuilder()
    builder.AddSystem(station)

    plant = station.GetSubsystemByName("plant")

    # Temporary context for kinematics / IK setup
    temp_context = station.CreateDefaultContext()
    temp_plant_context = plant.GetMyContextFromRoot(temp_context)

    # 2. Grab iiwa + hammer info
    iiwa = plant.GetModelInstanceByName("iiwa")
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_body = plant.GetBodyByName("hammer_link", hammer)
    hammer_body_index = hammer_body.index()
    hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)

    # 3. Compute X_HL7 (link 7 in hammer-face frame)
    X_HL7 = compute_X_HL7(plant, temp_plant_context, hammer_face_frame, iiwa)

    # 4. Get current iiwa joints (start pose)
    q_current = plant.GetPositions(temp_plant_context, iiwa)

    # 5. Get pose of the selected mole in world
    X_WMole = get_mole_pose_in_world(plant, temp_plant_context, mole_index)

    # 6. Compute hammer pre-hit pose from mole pose + X_HL7
    X_WH_prehit, X_WL7_prehit = get_prehit_pose(X_WMole, X_HL7)

    # 7. Solve IK for q_prehit
    q_prehit = solve_ik_for_prehit(
        plant,
        temp_plant_context,
        hammer_face_frame,
        X_WH_prehit,
        iiwa_model_instance=iiwa,
        q_guess=q_current,
    )

    # 8. Set admittance parameters
    params = AdmittanceParams(
        F_des=F_des,
        M_a=1.0,
        D_a=40.0,
        K_a=200.0,
        K_null=5.0,
        hit_duration=5.0,
        retract_duration=1.5,
        approach_timestep=1.0,
        v_pre=1.0,
    )

    # 9. Call your one-shot helper from whack_force_admittance
    diagram, simulator, hit_handles = run_hit_experiment(
        builder=builder,
        station=station,
        plant=plant,
        iiwa_model_instance=iiwa,
        hammer_body_index=hammer_body_index,
        hammer_face_frame=hammer_face_frame,
        q_current=q_current,
        X_WSoup=X_WMole,        # same as "soup" here: the mole/board pose
        X_WH_prehit=X_WH_prehit,
        q_prehit=q_prehit,
        params=params,
        T_final=T_final,
        plot=plot,
    )

    return diagram, simulator, hit_handles


if __name__ == "__main__":
    # Example usage:
    # Assume you have a scenario YAML dict with all 9 moles loaded
    from my_project_scenarios import whackamole_scenario_yaml

    # Hit mole 5 with default force, show plots
    run_admittance_hit_on_mole(
        scenario_yaml=whackamole_scenario_yaml,
        mole_index=5,
        F_des=200.0,
        T_final=8.0,
        plot=True,
    )
