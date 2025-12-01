"""
multi_hits.py

Sketch of how to run multi-hit Whack-a-Mole using the force admittance
module defined in whack_force_admittance.py.

Supports:
  - OFFLINE mode  (pre-programmed mole sequence)
  - LIVE mode     (mole index decided at runtime by a perception system)

Switch mode by toggling USE_LIVE_PERCEPTION below.
"""

import numpy as np

from whack_force_admittance import (
    AdmittanceParams,
    initialize_whack_system,
    configure_hit_for_target,
    reset_admittance_state_for_new_hit,
    plot_hit_results,
)

# ---------------------------------------------------------------------------
# Mode switch
# ---------------------------------------------------------------------------

# If False: use a fixed, offline sequence of mole indices (e.g. [0, 4, 8]).
# If True:  mole index is provided at runtime by a (placeholder) live perception system.
USE_LIVE_PERCEPTION = False


# ---------------------------------------------------------------------------
# Scenario + target helpers (YOU fill these in)
# ---------------------------------------------------------------------------

def load_scenario_yaml():
    """
    TODO: Replace this with your actual scenario loading.

    Options:
      - Read YAML from file:
            with open("path/to/your_scenario.yaml", "r") as f:
                scenario_yaml = f.read()
      - Or import a string from another module.

    Returns:
        scenario_yaml: str
            Full YAML text describing the Drake scenario.
    """
    # Placeholder: raise if you forget to implement
    raise NotImplementedError("Implement load_scenario_yaml() to return your scenario YAML string.")


def get_mole_sequence():
    """
    OFFLINE MODE ONLY.

    Decide the order of moles to hit in a pre-programmed sequence.

    Returns:
        mole_indices: list[int]
            Example: [0, 4, 8] means: first hit mole 0, then mole 4, then mole 8.

    Format:
        - Each mole index is an int, typically 0..8 for a 3x3 whack-a-mole grid.
    """
    # Placeholder: you can hard-code a test sequence or hook into your game logic
    return [0, 4, 8]


def get_next_mole_from_live_perception(current_time, previous_mole_index=None):
    """
    LIVE MODE ONLY.

    This is a placeholder for your **live perception integration**.
    In the real system, this might:
      - Read from a queue/ROS topic,
      - Query a vision model,
      - Or listen to game logic that decides which mole popped up.

    Inputs:
        current_time: float
            Current simulation time (seconds). You can use this to decide when
            new moles appear.
        previous_mole_index: int or None
            (Optional) Last mole that was hit; useful if you want to avoid
            hitting the same mole twice in a row.

    Returns:
        mole_index: int or None
            - int:    index of the mole to hit next (0..8, etc.)
            - None:   no mole to hit right now (keep waiting / simulating).

    NOTE:
        Right now this is a toy stub:
          - it "activates" mole 0 at t >= 0.5,
          - then mole 4 at t >= 3.0,
          - then mole 8 at t >= 6.0,
          - returns None otherwise.
        Replace this with your real perception hook.
    """
    # Example: simple time-based schedule pretending to be "live" decisions
    if current_time < 0.5:
        return None
    elif current_time < 3.0:
        return 0
    elif current_time < 6.0:
        return 4
    elif current_time < 9.0:
        return 8
    else:
        return None


def get_target_for_mole(
    mole_index,
    plant,
    iiwa_model_instance,
    hammer_face_frame,
    root_context,
):
    """
    Compute the soup/board pose and pre-hit pose for a given mole.

    This is where you plug in:
      - perception (to locate the active mole in camera frame),
      - kinematics (to convert that into a world-frame target),
      - IK to compute a pre-hit joint config q_prehit, etc.

    Inputs:
        mole_index: int
            Which mole (e.g. 0..8).
        plant: pydrake.multibody.plant.MultibodyPlant
        iiwa_model_instance: pydrake.multibody.tree.ModelInstanceIndex
            The model instance for the iiwa arm.
        hammer_face_frame: pydrake.multibody.tree.Frame
            Frame attached to the hammer face.
        root_context: pydrake.systems.framework.Context
            Root diagram context (used to get the plant subcontext).

    Returns:
        X_WSoup: pydrake.math.RigidTransform
            Pose of the board/mole contact surface in world frame at hit.
            (In the simple toy example, you can just use the board frame pose.)
        X_WH_prehit: pydrake.math.RigidTransform
            Pose of the hammer face at the pre-hit configuration (in world frame).
        q_prehit: np.ndarray, shape (7,)
            Joint angles for the iiwa at the pre-hit pose, in radians.

    NOTE:
        Right now, this uses a VERY SIMPLE placeholder:
          - uses the board body (base_link_soup) as "soup",
          - uses the *current* hammer pose as pre-hit,
          - sets q_prehit = current joint positions.
        This will run, but you should replace it with your actual prehit + IK pipeline.
    """
    # Get the plant subcontext
    plant_context = plant.GetMyContextFromRoot(root_context)

    # --- 1) Get the board / soup pose in world ---------------------------
    # TODO: If you have separate mole bodies/frames, change the name here,
    #       e.g. "mole_0", "mole_1", etc. based on mole_index.
    soup_body = plant.GetBodyByName("base_link_soup")
    X_WSoup = plant.EvalBodyPoseInWorld(plant_context, soup_body)

    # --- 2) Get current hammer face pose ---------------------------------
    X_WH_current = hammer_face_frame.CalcPoseInWorld(plant_context)

    # --- 3) Get current iiwa joint positions -----------------------------
    q_current = plant.GetPositions(plant_context, iiwa_model_instance)

    # TODO: Replace the lines below with your actual pre-hit IK:
    #   Example structure (pseudo-code):
    #       X_WH_prehit = compute_prehit_pose_for_mole(mole_index, X_WSoup, ...)
    #       q_prehit = solve_ik_for_pre_hit(X_WH_prehit, plant, iiwa_model, ...)
    #
    # For now, we just use the current pose as pre-hit, so approach = zero.
    X_WH_prehit = X_WH_current
    q_prehit = q_current.copy()

    return X_WSoup, X_WH_prehit, q_prehit


def compute_segment_end_time(
    t_now,
    traj_duration,
    hit_duration,
    retract_duration,
    extra_margin=0.2,
):
    """
    Compute the absolute time until which we should advance the simulator
    for a single hit segment.

    Inputs:
        t_now: float
            Current simulation time (seconds).
        traj_duration: float
            Duration of the approach trajectory (seconds).
        hit_duration: float
            Duration of the hit phase (seconds).
        retract_duration: float
            Duration of the retract phase (seconds).
        extra_margin: float
            Small extra time after retract to let things settle.

    Returns:
        t_final_segment: float
            Absolute simulation time to AdvanceTo(...) for this hit.
    """
    return t_now + traj_duration + hit_duration + retract_duration + extra_margin


# ---------------------------------------------------------------------------
# Offline + Live multi-hit routines
# ---------------------------------------------------------------------------

def run_offline_sequence(
    simulator,
    plant,
    iiwa,
    hammer_face_frame,
    hit_handles,
    params: AdmittanceParams,
):
    """
    OFFLINE MODE:
      - Use a fixed list of mole indices (e.g. [0, 4, 8])
      - For each mole, configure and execute one hit (approach + hit + retract).
    """
    root_context = simulator.get_mutable_context()

    # 1) Decide which moles to hit
    mole_indices = get_mole_sequence()

    for mole_index in mole_indices:
        print(f"\n=== Offline: starting hit for mole {mole_index} ===")

        t_now = root_context.get_time()
        plant_context = plant.GetMyContextFromRoot(root_context)
        q_current = plant.GetPositions(plant_context, iiwa)

        # 2) Get target soup + pre-hit for THIS mole
        X_WSoup, X_WH_prehit, q_prehit = get_target_for_mole(
            mole_index=mole_index,
            plant=plant,
            iiwa_model_instance=iiwa,
            hammer_face_frame=hammer_face_frame,
            root_context=root_context,
        )

        # 3) Configure admittance controller for THIS mole
        traj, t_hit_start, n_hat, J_pinv = configure_hit_for_target(
            hit_handles=hit_handles,
            plant=plant,
            iiwa_model_instance=iiwa,
            hammer_face_frame=hammer_face_frame,
            X_WSoup=X_WSoup,
            X_WH_prehit=X_WH_prehit,
            q_current=q_current,
            q_prehit=q_prehit,
            params=params,
            t_now=t_now,
        )

        # Reset [s, s_dot] for this hit
        reset_admittance_state_for_new_hit(
            hit_ctrl=hit_handles.hit_ctrl,
            root_context=root_context,
            v_pre=params.v_pre,
        )

        # 4) Run simulation until this hit finishes
        traj_duration = traj.end_time()
        t_final_segment = compute_segment_end_time(
            t_now=t_now,
            traj_duration=traj_duration,
            hit_duration=params.hit_duration,
            retract_duration=params.retract_duration,
            extra_margin=0.2,
        )

        print(f"  Advancing simulation from t={t_now:.3f} to t={t_final_segment:.3f}")
        simulator.AdvanceTo(t_final_segment)


def run_live_perception_loop(
    simulator,
    plant,
    iiwa,
    hammer_face_frame,
    hit_handles,
    params: AdmittanceParams,
    max_hits: int = 5,
    max_time: float = 15.0,
):
    """
    LIVE MODE:
      - Repeatedly query a (placeholder) live perception function to see if
        there's a mole to hit.
      - When a mole is available, configure and execute one hit.
      - Stop after `max_hits` hits or when `max_time` seconds have elapsed.
    """
    root_context = simulator.get_mutable_context()

    hits_done = 0
    last_mole_index = None

    while hits_done < max_hits:
        t_now = root_context.get_time()
        if t_now >= max_time:
            print("\nLive mode: reached max_time, stopping.")
            break

        # Ask perception: which mole to hit *right now*?
        mole_index = get_next_mole_from_live_perception(
            current_time=t_now,
            previous_mole_index=last_mole_index,
        )

        if mole_index is None:
            # No target yet → advance a little bit and check again
            simulator.AdvanceTo(t_now + 0.05)
            continue

        print(f"\n=== Live: perception selected mole {mole_index} at t={t_now:.3f} ===")
        last_mole_index = mole_index

        # Get latest q_current
        plant_context = plant.GetMyContextFromRoot(root_context)
        q_current = plant.GetPositions(plant_context, iiwa)

        # Get target soup + pre-hit for THIS mole
        X_WSoup, X_WH_prehit, q_prehit = get_target_for_mole(
            mole_index=mole_index,
            plant=plant,
            iiwa_model_instance=iiwa,
            hammer_face_frame=hammer_face_frame,
            root_context=root_context,
        )

        # Configure admittance controller
        traj, t_hit_start, n_hat, J_pinv = configure_hit_for_target(
            hit_handles=hit_handles,
            plant=plant,
            iiwa_model_instance=iiwa,
            hammer_face_frame=hammer_face_frame,
            X_WSoup=X_WSoup,
            X_WH_prehit=X_WH_prehit,
            q_current=q_current,
            q_prehit=q_prehit,
            params=params,
            t_now=t_now,
        )

        reset_admittance_state_for_new_hit(
            hit_ctrl=hit_handles.hit_ctrl,
            root_context=root_context,
            v_pre=params.v_pre,
        )

        # Advance through approach + hit + retract for this mole
        traj_duration = traj.end_time()
        t_final_segment = compute_segment_end_time(
            t_now=t_now,
            traj_duration=traj_duration,
            hit_duration=params.hit_duration,
            retract_duration=params.retract_duration,
            extra_margin=0.2,
        )

        print(f"  Advancing simulation from t={t_now:.3f} to t={t_final_segment:.3f}")
        simulator.AdvanceTo(t_final_segment)

        hits_done += 1

    print(f"\nLive mode: total hits performed = {hits_done}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_multi_hit_demo():
    """
    Example script:
      - Initialize whack-a-mole station + admittance controller.
      - Run multiple hits in OFFLINE or LIVE mode.
      - Plot results at the end.
    """
    # 1) Load scenario YAML
    scenario_yaml = load_scenario_yaml()

    # 2) Set up admittance parameters (tuning)
    params = AdmittanceParams(
        F_des=200.0,          # desired normal force [N]
        M_a=1.0,              # virtual mass
        D_a=40.0,             # virtual damping
        K_a=200.0,            # virtual stiffness along n_hat
        K_null=5.0,           # joint-space posture spring gain
        hit_duration=5.0,     # time in HIT phase [s]
        retract_duration=1.5, # time in RETRACT phase [s]
        approach_timestep=1.0,
        v_pre=1.0,            # initial velocity along n_hat at impact
    )

    # 3) Initialize whack-a-mole system + admittance core
    state = initialize_whack_system(scenario_yaml, params)

    diagram = state["diagram"]
    simulator = state["simulator"]
    station = state["station"]
    plant = state["plant"]
    iiwa = state["iiwa"]
    hammer_face_frame = state["hammer_face_frame"]
    hit_handles = state["hit_handles"]
    params = state["params"]  # same object; included for completeness

    # 4) Choose mode: OFFLINE or LIVE
    if USE_LIVE_PERCEPTION:
        print("Running multi-hit demo in LIVE perception mode...")
        run_live_perception_loop(
            simulator=simulator,
            plant=plant,
            iiwa=iiwa,
            hammer_face_frame=hammer_face_frame,
            hit_handles=hit_handles,
            params=params,
            max_hits=5,     # you can change this
            max_time=15.0,  # you can change this
        )
    else:
        print("Running multi-hit demo in OFFLINE (pre-programmed) mode...")
        run_offline_sequence(
            simulator=simulator,
            plant=plant,
            iiwa=iiwa,
            hammer_face_frame=hammer_face_frame,
            hit_handles=hit_handles,
            params=params,
        )

    # 5) Plot results from all hits (force, s, s_dot, q)
    plot_hit_results(hit_handles, plot=True)


if __name__ == "__main__":
    run_multi_hit_demo()
