from whack_force_admittance import (
    configure_hit_for_target,
    reset_admittance_state_for_new_hit,
)

def perform_hit_on_mole(state, mole_index: int, extra_time: float = 1.0):
    simulator = state["simulator"]
    station   = state["station"]
    plant     = state["plant"]
    iiwa      = state["iiwa"]
    hammer_face_frame = state["hammer_face_frame"]
    hit_handles = state["hit_handles"]
    params   = state["params"]

    # Current sim time & context
    root_context = simulator.get_mutable_context()
    t_now = root_context.get_time()

    # Current iiwa joints
    plant_ctx = plant.GetMyMutableContextFromRoot(root_context)
    q_current = plant.GetPositions(plant_ctx, iiwa)

    # Get mole pose
    temp_plant_context = plant_ctx  # or reuse; they’re the same view
    X_WMole = get_mole_pose_in_world(plant, temp_plant_context, mole_index)

    # Compute X_HL7 & prehit
    X_HL7 = compute_X_HL7(plant, temp_plant_context, hammer_face_frame, iiwa)
    X_WH_prehit, X_WL7_prehit = get_prehit_pose(X_WMole, X_HL7)

    # Solve IK to get q_prehit
    q_prehit = solve_ik_for_prehit(
        plant,
        temp_plant_context,
        hammer_face_frame,
        X_WH_prehit,
        iiwa_model_instance=iiwa,
        q_guess=q_current,
    )

    # Configure controller internals for this hit
    traj, t_hit_start, n_hat, J_pinv = configure_hit_for_target(
        hit_ctrl=hit_handles.hit_ctrl,
        plant=plant,
        iiwa_model_instance=iiwa,
        hammer_face_frame=hammer_face_frame,
        X_WSoup=X_WMole,
        X_WH_prehit=X_WH_prehit,
        q_current=q_current,
        q_prehit=q_prehit,
        params=params,
        t_now=t_now,
    )

    # Reset admittance state [s, s_dot] for new hit
    reset_admittance_state_for_new_hit(hit_handles.hit_ctrl, root_context, params.v_pre)

    # Decide how long to simulate after t_now
    T_hit_total = traj.end_time() + params.hit_duration + params.retract_duration + extra_time
    simulator.AdvanceTo(t_now + T_hit_total)
