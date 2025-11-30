# --- Imports ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from pydrake.systems.framework import LeafSystem, BasicVector, DiagramBuilder
from pydrake.common.value import AbstractValue
from pydrake.multibody.plant import ContactResults
from pydrake.multibody.tree import JacobianWrtVariable  # used later for J_pinv
from pydrake.systems.primitives import LogVectorOutput  # used later for logging
from pydrake.trajectories import PiecewisePolynomial     # used by trajectory helper
from pydrake.systems.analysis import Simulator
from manipulation.scenarios import LoadScenario, MakeHardwareStation





# --- Contact force extraction system --------------------------------------
class HammerContactForce(LeafSystem):
    """
    Reads ContactResults from the plant and outputs the scalar contact force
    on the hammer body along n_hat (in world frame).
    """

    def __init__(self, plant, hammer_body_index, n_hat):
        super().__init__()
        self._plant = plant
        self._hammer_body_index = hammer_body_index
        # Ensure n_hat is a unit vector
        self._n_hat = n_hat / np.linalg.norm(n_hat)

        # Abstract input: ContactResults from the plant
        self.DeclareAbstractInputPort(
            "contact_results",
            AbstractValue.Make(ContactResults())
        )

        # Scalar output: F_meas (normal force along n_hat)
        self.DeclareVectorOutputPort(
            "F_meas",
            BasicVector(1),
            self.CalcOutput,
        )
    
    def set_direction(self, n_hat):
        """Update the projection direction n_hat (unit vector in world)."""
        self._n_hat = n_hat / np.linalg.norm(n_hat)


    def CalcOutput(self, context, output):
        # Read ContactResults from the input port
        contact_results = self.get_input_port(0).Eval(context)

        # Accumulate net contact force on the hammer body, expressed in world
        F_W = np.zeros(3)

        for i in range(contact_results.num_point_pair_contacts()):
            info = contact_results.point_pair_contact_info(i)
            f_Bc_W = np.array(info.contact_force())  # force on B, expressed in W
            bodyA = info.bodyA_index()
            bodyB = info.bodyB_index()

            if bodyA == self._hammer_body_index:
                # Force on A is equal and opposite to that on B
                F_W -= f_Bc_W
            if bodyB == self._hammer_body_index:
                F_W += f_Bc_W

        # Project net force onto n_hat to get scalar normal force
        F_meas = float(self._n_hat.dot(F_W))
        output.SetAtIndex(0, F_meas)


class HitSequenceAdmittance(LeafSystem):
    """
    3-phase controller with 1D admittance + joint-space posture spring:

      Phase 1 (approach): follow joint-space trajectory traj(t) for ALL joints.
      Phase 2 (hit): 1D admittance along n_hat using ALL joints to realize
                     the motion; also apply posture spring toward q_prehit.
      Phase 3 (retract): same as Phase 2 but with F_des_eff = 0 so
                         the virtual spring K_a pulls s → 0 (retract hammer).

    State: [s, s_dot] = [displacement along n_hat, velocity along n_hat]
    Inputs:
      0: F_meas (scalar)    - measured force along n_hat
      1: q_meas (7-vector)  - measured iiwa joint positions
    Outputs:
      0: q_cmd (7-vector)        - joint position command
      1: admittance_state (2-d)  - [s, s_dot]
    """

    def __init__(self, traj_approach, t_hit_start, hit_duration,
                 q_prehit, J_pinv, n_hat,
                 F_des,
                 M_a=1.0, D_a=40.0, K_a=200.0,
                 K_null=5.0,
                 retract_duration=1.5):
        super().__init__()

        self._traj = traj_approach
        self._traj_duration = traj_approach.end_time()
        self._t_traj_start = 0.0  # initial approach starts at t = 0
        self._t_hit_start = float(t_hit_start)
        self._hit_duration = float(hit_duration)

        self._q_prehit = q_prehit.copy().reshape(-1)   # 7
        self._J_pinv = J_pinv.copy()                   # (7, 3)
        self._n_hat = n_hat / np.linalg.norm(n_hat)

        self._F_des_hit = float(F_des)
        self._M_a = float(M_a)
        self._D_a = float(D_a)
        self._K_a = float(K_a)        # used mainly in retract
        self._K_null = float(K_null)  # posture spring gain
        self._retract_duration = float(retract_duration)

        # Continuous state: [s, s_dot]
        self.DeclareContinuousState(2)

        # Input 0: measured force along n_hat
        self.DeclareVectorInputPort("F_meas", BasicVector(1))
        # Input 1: measured joint positions (q_meas)
        self.DeclareVectorInputPort("q_meas", BasicVector(len(self._q_prehit)))

        # Output 0: joint position command q_cmd
        self.DeclareVectorOutputPort(
            "q_cmd",
            BasicVector(len(self._q_prehit)),
            self.CalcOutput,
        )

        # Output 1: internal admittance state [s, s_dot]
        self.DeclareVectorOutputPort(
            "admittance_state",
            BasicVector(2),
            self.CalcStateOutput,
        )
        
    def configure_new_hit(
        self,
        t_hit_start,
        q_prehit,
        J_pinv,
        n_hat,
        F_des,
        hit_duration,
        retract_duration,
        traj_approach,
    ):
        """Update internal parameters for a new hit without rebuilding the system."""
        self._t_hit_start = float(t_hit_start)
        self._hit_duration = float(hit_duration)
        self._retract_duration = float(retract_duration)
        self._q_prehit = q_prehit.copy().reshape(-1)
        self._J_pinv = J_pinv.copy()
        self._n_hat = n_hat / np.linalg.norm(n_hat)
        self._F_des_hit = float(F_des)

        self._traj = traj_approach
        self._traj_duration = traj_approach.end_time()
        # The approach should run over [t_traj_start, t_hit_start] in absolute time.
        # Since we choose t_hit_start = t_traj_start + traj_duration, we get:
        self._t_traj_start = self._t_hit_start - self._traj_duration



    def DoCalcTimeDerivatives(self, context, derivatives):
        t = context.get_time()
        x = context.get_continuous_state_vector().CopyToVector()
        s = x[0]
        v = x[1]

        if t < self._t_hit_start:
            # Phase 1: approach – keep admittance state frozen
            ds = 0.0
            dv = 0.0

        else:
            tau = t - self._t_hit_start
            F_meas = self.get_input_port(0).Eval(context)[0]

            if tau <= self._hit_duration:
                # Phase 2: HIT – true force control
                F_des_eff = self._F_des_hit
                F_err = F_des_eff - F_meas
            else:
                # Phase 3: RETRACT – ignore measured force, just spring back to s = 0
                F_err = 0.0   # no F_meas in retract

            ds = v
            dv = (F_err - self._D_a * v - self._K_a * s) / self._M_a

        der = derivatives.get_mutable_vector()
        der[0] = ds
        der[1] = dv

    def CalcOutput(self, context, output):
        t = context.get_time()
        x = context.get_continuous_state_vector().CopyToVector()
        s = x[0]

        if t < self._t_hit_start:
            # Phase 1: approach – follow joint-space trajectory in "local" time
            t_rel = t - self._t_traj_start
            # Clamp just in case
            if t_rel < 0.0:
                t_rel = 0.0
            if t_rel > self._traj_duration:
                t_rel = self._traj_duration
            q_cmd = self._traj.value(t_rel).flatten()


        else:
            tau = t - self._t_hit_start

            if tau <= self._hit_duration:
                # Phase 2: HIT – admittance + posture spring
                dx_W = self._n_hat * s
                dq   = self._J_pinv @ dx_W

                q_meas = self.get_input_port(1).Eval(context)
                q_err  = self._q_prehit - q_meas
                q_null = self._K_null * q_err

                q_cmd = self._q_prehit + dq + q_null

            elif tau <= self._hit_duration + self._retract_duration:
                # Phase 3: RETRACT – s is being driven toward 0, still use posture spring
                dx_W = self._n_hat * s
                dq   = self._J_pinv @ dx_W

                q_meas = self.get_input_port(1).Eval(context)
                q_err  = self._q_prehit - q_meas
                q_null = self._K_null * q_err

                q_cmd = self._q_prehit + dq + q_null

            else:
                # Phase 4: DONE – lock exactly at pre-hit pose
                q_cmd = self._q_prehit

        output.SetFromVector(q_cmd)

    def CalcStateOutput(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)  # [s, s_dot]

# ---------------------------------------------------------------------------
# Admittance configuration + handles
# ---------------------------------------------------------------------------

@dataclass
class AdmittanceParams:
    """Tuning parameters for the 1D admittance + posture spring."""
    F_des: float = 200.0          # desired normal force [N]
    M_a: float = 1.0              # virtual mass
    D_a: float = 40.0             # virtual damping
    K_a: float = 200.0            # virtual stiffness along n_hat
    K_null: float = 5.0           # joint-space posture spring gain
    hit_duration: float = 5.0     # how long to track F_des [s]
    retract_duration: float = 1.5 # how long to let spring retract [s]
    approach_timestep: float = 1.0# dt between waypoints in approach traj
    v_pre: float = 1.0            # initial velocity along n_hat at contact
                                  # (you’ll set this in the controller context)
    

@dataclass
class HitAdmittanceHandles:
    """
    Convenience bundle returned by the builder function so the notebook
    can access the main systems, trajectory, n_hat, J_pinv, and loggers.
    """
    hit_ctrl: HitSequenceAdmittance
    force_sys: HammerContactForce
    traj_approach: 'PiecewisePolynomial'
    t_hit_start: float
    loggers: dict        # e.g. {"force": logger, "adm": logger, "q": logger, "v": logger}
    n_hat: np.ndarray
    J_pinv: np.ndarray


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_joint_space_position_trajectory(path, timestep=1.0):
    """
    Given a list of joint-waypoints [q0, q1, ...], returns a
    PiecewisePolynomial that linearly interpolates between them in time.

    times: [0, dt, 2*dt, ...]
    Q:     stacked as columns, shape (nq, N)
    """
    times = [timestep * i for i in range(len(path))]
    Q = np.column_stack(path)
    return PiecewisePolynomial.FirstOrderHold(times, Q)


def compute_n_hat(X_WSoup, X_WH_prehit):
    """
    Compute unit hit direction n_hat in world frame, pointing
    from hammer pre-hit position toward the soup center.

    X_WSoup:   pose of soup in world
    X_WH_prehit: pose of hammer face at pre-hit in world
    """
    p_soup_W = X_WSoup.translation()
    p_hammer_prehit_W = X_WH_prehit.translation()
    n = p_soup_W - p_hammer_prehit_W
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        # Fallback direction (e.g. world -z) if poses coincide
        n = np.array([0.0, 0.0, -1.0])
        norm = np.linalg.norm(n)
    return n / norm


def compute_J_pinv_at_prehit(plant, iiwa_model_instance, hammer_face_frame, q_prehit):
    """
    Compute the translational velocity Jacobian Jv_WQ at the hammer-face
    origin, at the pre-hit configuration q_prehit, and return its
    Moore–Penrose pseudoinverse J_pinv (7x3).

      v_WQ = Jv_WQ * v    ⇒   dq ≈ J_pinv * dx_W
    """
    J_context = plant.CreateDefaultContext()
    plant.SetPositions(J_context, iiwa_model_instance, q_prehit)

    p_HQ_H = np.zeros(3)  # origin of hammer-face frame

    Jv_WQ = plant.CalcJacobianTranslationalVelocity(
        J_context,
        JacobianWrtVariable.kV,
        hammer_face_frame,    # frame B = hammer face
        p_HQ_H,               # point Q in frame B
        plant.world_frame(),  # measure in world
        plant.world_frame(),  # express in world
    )

    J_pinv = np.linalg.pinv(Jv_WQ)  # shape (7,3) for iiwa
    return J_pinv


# ---------------------------------------------------------------------------
# Builder: add contact-force + hit admittance pipeline to a diagram builder
# ---------------------------------------------------------------------------

def build_hit_admittance_pipeline(
    builder,
    station,
    plant,
    iiwa_model_instance,
    hammer_body_index,
    hammer_face_frame,
    q_current,
    X_WSoup,
    X_WH_prehit,
    q_prehit,
    params: AdmittanceParams,
) -> HitAdmittanceHandles:
    """
    Adds the full hit-admittance pipeline to `builder`:

      - joint-space approach trajectory from q_current -> q_prehit
      - hit direction n_hat
      - J_pinv at pre-hit
      - HammerContactForce to extract scalar F_meas
      - HitSequenceAdmittance controller
      - loggers for F_meas, admittance state, q, v

    Returns a HitAdmittanceHandles bundle with useful references.

    NOTE: this assumes:
      station.get_output_port(18) = ContactResults
      station.get_output_port(3)  = iiwa.position_measured
      station.get_output_port(4)  = iiwa.velocity_estimated
      station.GetInputPort("iiwa.position") is the iiwa position command.
    """

    # 1. Approach trajectory in joint space
    path = [q_current, q_prehit]
    traj = make_joint_space_position_trajectory(
        path,
        timestep=params.approach_timestep,
    )
    t_hit_start = traj.end_time()

    # 2. Hit direction n_hat (world) from hammer pre-hit -> soup
    n_hat = compute_n_hat(X_WSoup, X_WH_prehit)

    # 3. Jacobian pseudoinverse at pre-hit (hammer-face origin)
    J_pinv = compute_J_pinv_at_prehit(
        plant,
        iiwa_model_instance,
        hammer_face_frame,
        q_prehit,
    )

    # 4. Contact force extraction system
    force_sys = builder.AddSystem(
        HammerContactForce(plant, hammer_body_index, n_hat)
    )
    # ContactResults from station → HammerContactForce
    builder.Connect(
        station.get_output_port(18),   # 'contact_results'
        force_sys.get_input_port(0),
    )

    # Log scalar normal force F_meas
    force_logger = LogVectorOutput(
        force_sys.get_output_port(0),  # F_meas
        builder,
    )

    # 5. Hit + admittance + retract controller
    hit_ctrl = builder.AddSystem(
        HitSequenceAdmittance(
            traj_approach=traj,
            t_hit_start=t_hit_start,
            hit_duration=params.hit_duration,
            q_prehit=q_prehit,
            J_pinv=J_pinv,
            n_hat=n_hat,
            F_des=params.F_des,
            M_a=params.M_a,
            D_a=params.D_a,
            K_a=params.K_a,
            K_null=params.K_null,
            retract_duration=params.retract_duration,
        )
    )

    # Measured force → controller (input 0)
    builder.Connect(
        force_sys.get_output_port(0),
        hit_ctrl.get_input_port(0),
    )

    # Measured joint positions → controller (input 1)
    builder.Connect(
        station.get_output_port(3),  # 'iiwa.position_measured'
        hit_ctrl.get_input_port(1),
    )

    # Controller output q_cmd → iiwa position command
    builder.Connect(
        hit_ctrl.get_output_port(0),
        station.GetInputPort("iiwa.position"),
    )

    # 6. Log admittance internal state [s, s_dot]
    adm_logger = LogVectorOutput(
        hit_ctrl.get_output_port(1),  # "admittance_state"
        builder,
    )

    # Log measured joint positions
    q_logger = LogVectorOutput(
        station.get_output_port(3),   # 'iiwa.position_measured'
        builder,
    )

    # Log estimated joint velocities
    v_logger = LogVectorOutput(
        station.get_output_port(4),   # 'iiwa.velocity_estimated'
        builder,
    )

    # 7. Package handles for the caller
    loggers = {
        "force": force_logger,
        "adm": adm_logger,
        "q": q_logger,
        "v": v_logger,
    }

    return HitAdmittanceHandles(
        hit_ctrl=hit_ctrl,
        force_sys=force_sys,
        traj_approach=traj,
        t_hit_start=t_hit_start,
        loggers=loggers,
        n_hat=n_hat,
        J_pinv=J_pinv,
    )

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_hit_results(hit_handles: HitAdmittanceHandles, plot: bool = True):
    """
    Plot force, admittance state [s, s_dot], and joint positions using
    the loggers stored in hit_handles.

    If plot=False, this function is a no-op.
    Call this AFTER running the simulation (i.e., after AdvanceTo).
    """
    if not plot:
        return

    # Unpack loggers
    force_logger = hit_handles.loggers.get("force", None)
    adm_logger   = hit_handles.loggers.get("adm", None)
    q_logger     = hit_handles.loggers.get("q", None)

    # --- Plot measured normal force ---
    if force_logger is not None:
        t_force = force_logger.sample_times()
        data_F  = force_logger.data()   # shape (1, N) if scalar
        F_meas  = data_F[0, :]

        plt.figure()
        plt.plot(t_force, F_meas)
        plt.xlabel("t [s]")
        plt.ylabel("F_meas [N]")
        plt.title("Measured normal force along n_hat")

    # --- Plot admittance state [s, s_dot] ---
    if adm_logger is not None:
        t_adm = adm_logger.sample_times()
        adm   = adm_logger.data()       # shape (2, N)
        s     = adm[0, :]
        s_dot = adm[1, :]

        plt.figure()
        plt.plot(t_adm, s)
        plt.xlabel("t [s]")
        plt.ylabel("s [m]")
        plt.title("Admittance displacement along n_hat")

        plt.figure()
        plt.plot(t_adm, s_dot)
        plt.xlabel("t [s]")
        plt.ylabel("s_dot [m/s]")
        plt.title("Admittance velocity along n_hat")

    # --- Plot joint positions ---
    if q_logger is not None:
        t_q = q_logger.sample_times()
        Q   = q_logger.data()           # shape (nq, N)

        plt.figure()
        for i in range(Q.shape[0]):
            plt.plot(t_q, Q[i, :], label=f"q{i+1}")
        plt.xlabel("t [s]")
        plt.ylabel("joint angle [rad]")
        plt.title("iiwa joint positions (measured)")
        plt.legend()

    # Optional: show all figures
    plt.show()

# ---------------------------------------------------------------------------
# One-shot convenience: build, run, (optionally) plot
# ---------------------------------------------------------------------------

def run_hit_experiment(
    builder,
    station,
    plant,
    iiwa_model_instance,
    hammer_body_index,
    hammer_face_frame,
    q_current,
    X_WSoup,
    X_WH_prehit,
    q_prehit,
    params: AdmittanceParams,
    T_final: float,
    plot: bool = True,
):
    """
    One-shot helper:

      1. Adds the hit-admittance pipeline to `builder`.
      2. Builds the diagram.
      3. Sets initial plant + admittance state.
      4. Runs the simulator to T_final.
      5. Calls plot_hit_results(..., plot=plot).

    Returns: (diagram, simulator, hit_handles)

    IMPORTANT:
      - Do NOT call builder.Build() yourself before this.
      - Call this after you've already created `station`, `plant`, `iiwa`, etc.
    """

    # 1. Build the hit-admittance pipeline (adds systems + loggers to builder)
    hit_handles = build_hit_admittance_pipeline(
        builder=builder,
        station=station,
        plant=plant,
        iiwa_model_instance=iiwa_model_instance,
        hammer_body_index=hammer_body_index,
        hammer_face_frame=hammer_face_frame,
        q_current=q_current,
        X_WSoup=X_WSoup,
        X_WH_prehit=X_WH_prehit,
        q_prehit=q_prehit,
        params=params,
    )

    # 2. Build the full diagram
    diagram = builder.Build()

    # 3. Create simulator and get mutable root context
    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()

    # 4. Initialize iiwa joint positions to q_current (if desired)
    plant_ctx = plant.GetMyMutableContextFromRoot(root_context)
    plant.SetPositions(plant_ctx, iiwa_model_instance, q_current)

    # 5. Initialize admittance state [s, s_dot] = [0, v_pre]
    hit_ctx = hit_handles.hit_ctrl.GetMyMutableContextFromRoot(root_context)
    x_hit = hit_ctx.get_mutable_continuous_state_vector()
    x_hit[0] = 0.0            # s = 0 → at pre-hit pose
    x_hit[1] = params.v_pre   # s_dot = v_pre → initial velocity along n_hat

    # 6. Run the simulation
    simulator.AdvanceTo(T_final)

    # 7. Plot results (if requested)
    plot_hit_results(hit_handles, plot=plot)

    return diagram, simulator, hit_handles


def configure_hit_for_target(
    hit_handles: HitAdmittanceHandles,
    plant,
    iiwa_model_instance,
    hammer_face_frame,
    X_WSoup,
    X_WH_prehit,
    q_current,
    q_prehit,
    params: AdmittanceParams,
    t_now: float,
):
    hit_ctrl = hit_handles.hit_ctrl
    force_sys = hit_handles.force_sys

    # Approach traj
    path = [q_current, q_prehit]
    traj = make_joint_space_position_trajectory(path, timestep=params.approach_timestep)
    t_hit_start = t_now + traj.end_time()

    # Direction & Jacobian
    n_hat = compute_n_hat(X_WSoup, X_WH_prehit)
    J_pinv = compute_J_pinv_at_prehit(plant, iiwa_model_instance, hammer_face_frame, q_prehit)

    # Update BOTH systems to use the new direction
    force_sys.set_direction(n_hat)
    hit_ctrl.configure_new_hit(
        t_hit_start=t_hit_start,
        q_prehit=q_prehit,
        J_pinv=J_pinv,
        n_hat=n_hat,
        F_des=params.F_des,
        hit_duration=params.hit_duration,
        retract_duration=params.retract_duration,
        traj_approach=traj,
    )

    # Optional: keep handles in sync
    hit_handles.traj_approach = traj
    hit_handles.t_hit_start = t_hit_start
    hit_handles.n_hat = n_hat
    hit_handles.J_pinv = J_pinv

    return traj, t_hit_start, n_hat, J_pinv



def build_hit_admittance_core(
    builder, station, plant, iiwa_model_instance, hammer_body_index, hammer_face_frame, params: AdmittanceParams
):
    # Use q_current = zeros or plant defaults for dummy setup
    temp_ctx = station.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(temp_ctx)
    q_current = plant.GetPositions(plant_ctx, iiwa_model_instance)

    # Dummy initial prehit (same as current)
    q_prehit = q_current.copy()
    X_WSoup_dummy = plant.EvalBodyPoseInWorld(plant_ctx, plant.world_body())
    X_WH_prehit_dummy = X_WSoup_dummy  # dummy; will be overwritten


    handles = build_hit_admittance_pipeline(
        builder=builder,
        station=station,
        plant=plant,
        iiwa_model_instance=iiwa_model_instance,
        hammer_body_index=hammer_body_index,
        hammer_face_frame=hammer_face_frame,
        q_current=q_current,
        X_WSoup=X_WSoup_dummy,
        X_WH_prehit=X_WH_prehit_dummy,
        q_prehit=q_prehit,
        params=params,
    )

    return handles


def reset_admittance_state_for_new_hit(hit_ctrl, root_context, v_pre: float):
    hit_ctx = hit_ctrl.GetMyMutableContextFromRoot(root_context)
    x_hit = hit_ctx.get_mutable_continuous_state_vector()
    x_hit[0] = 0.0        # s = 0
    x_hit[1] = v_pre      # s_dot = v_pre


def initialize_whack_system(scenario_yaml, params: AdmittanceParams):
    scenario = LoadScenario(data=scenario_yaml)
    station = MakeHardwareStation(scenario, meshcat=None)
    builder = DiagramBuilder()
    builder.AddSystem(station)
    plant = station.GetSubsystemByName("plant")

    # Get iiwa + hammer
    iiwa = plant.GetModelInstanceByName("iiwa")
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_body = plant.GetBodyByName("hammer_link", hammer)
    hammer_body_index = hammer_body.index()
    hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)

    # Build and wire the admittance controller (with dummy initial config)
    
    hit_handles = build_hit_admittance_core(
        builder=builder,
        station=station,
        plant=plant,
        iiwa_model_instance=iiwa,
        hammer_body_index=hammer_body_index,
        hammer_face_frame=hammer_face_frame,
        params=params,
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()

    return {
        "diagram": diagram,
        "simulator": simulator,
        "station": station,
        "plant": plant,
        "iiwa": iiwa,
        "hammer_face_frame": hammer_face_frame,
        "hammer_body_index": hammer_body_index,
        "hit_handles": hit_handles,
        "params": params,
    }


