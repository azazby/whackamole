from whack_force_admittance import (
    AdmittanceParams,
    run_hit_experiment,
)

# ... you already have:
# station, plant, builder
# iiwa, hammer_body_index, hammer_face_frame
# iiwa_q0, X_WSoup, X_WH_prehit, q_goal (pre-hit IK)

params = AdmittanceParams(
    F_des=200.0,
    M_a=1.0,
    D_a=40.0,
    K_a=200.0,
    K_null=5.0,
    hit_duration=5.0,
    retract_duration=1.5,
    approach_timestep=1.0,
    v_pre=1.0,
)

T_final = 8.0  # e.g. 1s approach + 5s hit + 1.5s retract + margin

diagram, simulator, hit_handles = run_hit_experiment(
    builder=builder,
    station=station,
    plant=plant,
    iiwa_model_instance=iiwa,
    hammer_body_index=hammer_body_index,
    hammer_face_frame=hammer_face_frame,
    q_current=iiwa_q0,
    X_WSoup=X_WSoup,
    X_WH_prehit=X_WH_prehit,
    q_prehit=q_goal,
    params=params,
    T_final=T_final,
    plot=True,   # set False to suppress plots
)
