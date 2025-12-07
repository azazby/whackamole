#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    InverseDynamicsController,
    MeshcatPoseSliders,
    Context,
    RigidTransform,
    RotationMatrix,
    Sphere,
    Rgba,
    LeafSystem,
    RollPitchYaw,
    MultibodyPlant,
    Parser,
    Solve,
    SolutionResult,
    AddMultibodyPlantSceneGraph,
    AddFrameTriadIllustration,
    FixedOffsetFrame,
    TrajectorySource,
    Trajectory,
    PiecewisePose,
    PiecewisePolynomial,
    ConstantValueSource,
    AbstractValue
)
from pydrake.multibody import inverse_kinematics
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsIntegrator,
)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.primitives import LogVectorOutput
from pydrake.systems.framework import BasicVector

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)
from manipulation.utils import RenderDiagram, running_as_notebook
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.multibody.plant import ContactResults
from pydrake.multibody.tree import JacobianWrtVariable

from pathlib import Path
import sys
from typing import Any, Dict, List
from pydrake.common.yaml import yaml_load

# Ensure repo root is on sys.path so `import bonkbot` works even when run directly.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Perception utilities
from bonkbot.perception.whack_perception import (
    initialize_whack_perception_system,
    MoleDetection,
    perceive_up_moles,
    get_mole_world_poses,
    run_multi_snapshots as wp_run_multi_snapshots,
)
from bonkbot.control.whack_force_admittance import (
    AdmittanceParams,
    build_hit_admittance_pipeline,
    plot_hit_results,
    reset_admittance_state_for_new_hit,
    configure_hit_for_target,
)
from bonkbot.perception import whack_perception as wp

import random
import numpy as np


# In[2]:

# -----------------------------------------------------------------------------
# Perception helpers (mirrors bonkbot/perception/perception_test.py)
# -----------------------------------------------------------------------------

# What mole pose(s) to print from the plant state: "base", "top", or "both".
POSE_REPORT_MODE = "both"
# Which source to use for poses: "plant" (ground truth) or "camera".
POSE_SOURCE = "camera"
# Whether to print camera-vs-plant differences.
REPORT_DIFFERENCES = True
# Camera placement method: "ring" (default) or "notebook" (mirrors screenshot directives)
CAMERA_METHOD = "ring"
# Offset from mole link frame to top surface (matches mole.sdf visual offset).
MOLE_TOP_OFFSET_M = np.array([0.0, 0.0, 0.10])

# Save raw camera snapshots (RGB + depth) each time we query perception.
SAVE_CAMERA_SNAPSHOTS = True
SNAPSHOT_DIR = Path("camera_snaps")
# Whether to let perception choose which mole to hit; falls back to plant ground
# truth for mole_1_1 if no detection is available.
USE_PERCEPTION_FOR_TARGET = True
# Draw simple camera markers so they are visible in the main Meshcat session.
SHOW_PERCEPTION_CAMERAS = True


def _load_base_scenario_dict() -> Dict[str, Any]:
    """
    Load the YAML scenario string from this file and convert it into a dict
    suitable for manipulation.scenarios.LoadScenario.
    """
    scenario_dict = yaml_load(data=scenario_string)
    if not isinstance(scenario_dict, dict):
        raise RuntimeError(
            "Expected scenario_string to parse into a dict, but got a different type."
        )
    return scenario_dict


def build_perception_system(meshcat=None) -> Dict[str, Any]:
    """
    Build the whack-perception station/diagram for the simple scene defined here.

    Returns the `system_handles` dict that whack_perception expects:
        {
            "diagram": Diagram,
            "simulator": Simulator,
            "root_context": Context,
            "station": HardwareStation,
            "plant": MultibodyPlant,
            "camera_ports": List[CameraPorts],
        }
    """
    base_scenario = _load_base_scenario_dict()
    return initialize_whack_perception_system(
        base_scenario_dict=base_scenario,
        meshcat=meshcat,
        camera_method=CAMERA_METHOD,
    )


def run_perception_snapshot(
    t_capture: float = 1.0,
    pose_report_mode: str = POSE_REPORT_MODE,
    pose_source: str = POSE_SOURCE,
    meshcat=None,
) -> List[MoleDetection]:
    """
    One-shot perception pass that mirrors perception_test.run_single_snapshot.
    """
    detections, _ = detect_moles_with_perception(meshcat=meshcat, t_capture=t_capture)
    if pose_source.lower() == "plant":
        # Mirror perception_test: print plant poses if requested.
        handles = build_perception_system(meshcat=meshcat)
        mole_poses = get_mole_world_poses(handles)
        print("\nWorld poses of moles (plant state):")
        for (i, j), X_WM in sorted(mole_poses.items()):
            p = X_WM.translation()
            rpy = X_WM.rotation().ToRollPitchYaw().vector()
            print(
                f"  mole_{i}_{j}: p_W = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], "
                f"rpy = [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
            )
    else:
        print("\nWorld poses of moles (from camera detections):")
        if not detections:
            print("  none (no detections)")
        else:
            for det in detections:
                p_W = det.X_W_mole.translation()
                rpy = det.X_W_mole.rotation().ToRollPitchYaw().vector()
                print(
                    f"  mole_{det.mole_index}: p_W = [{p_W[0]:.3f}, {p_W[1]:.3f}, {p_W[2]:.3f}], "
                    f"rpy_W = [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
                )
    return detections


def run_perception_multi(
    t_end: float = 20.0,
    step: float = 1.0,
    pose_report_mode: str = POSE_REPORT_MODE,
    pose_source: str = POSE_SOURCE,
    meshcat=None,
) -> None:
    """
    Multi-snapshot perception loop; convenience wrapper mirroring perception_test.
    """
    wp_run_multi_snapshots(
        build_system_fn=build_perception_system,
        t_end=t_end,
        step=step,
        pose_source=pose_source,
        pose_report_mode=pose_report_mode,
        top_offset=MOLE_TOP_OFFSET_M,
        save_snapshots=SAVE_CAMERA_SNAPSHOTS,
        snapshot_dir=SNAPSHOT_DIR,
        start_meshcat=(meshcat is None),
        report_differences=REPORT_DIFFERENCES,
    )


def draw_cameras(meshcat, camera_configs):
    """
    Draw camera boxes in Meshcat so they show up in the main sim visualization.
    """
    if meshcat is None or not camera_configs:
        return
    try:
        from pydrake.geometry import Box
    except ImportError:
        return
    cam_box = Box(0.08, 0.05, 0.05)
    cam_color = Rgba(0.15, 0.15, 0.18, 0.9)
    for cfg in camera_configs:
        obj_path = f"perception_cameras/{cfg.name}"
        X_WC = RigidTransform(RollPitchYaw(cfg.rpy_W), cfg.position_W)
        meshcat.SetObject(obj_path, cam_box, cam_color)
        meshcat.SetTransform(obj_path, X_WC)


def detect_moles_with_perception(meshcat=None, t_capture: float = 1.0):
    """
    Build perception on the current scenario, run one snapshot, and return
    detections plus handles. Uses the existing meshcat so camera markers
    appear in the same viewer.
    """
    from bonkbot.perception import whack_perception as wp

    # Enable camera markers in the perception diagram if requested.
    wp.DRAW_CAMERA_MARKERS = SHOW_PERCEPTION_CAMERAS

    system_handles = build_perception_system(meshcat=meshcat)
    camera_configs = system_handles.get("camera_configs", [])
    if SHOW_PERCEPTION_CAMERAS:
        draw_cameras(meshcat, camera_configs)

    detections = perceive_up_moles(
        system_handles=system_handles,
        camera_intrinsics=system_handles["camera_intrinsics"],
        t_capture=t_capture,
    )

    # Optionally dump snapshots using whack_perception helper if available.
    if SAVE_CAMERA_SNAPSHOTS:
        try:
            from bonkbot.perception.whack_perception import dump_camera_snapshots

            dump_camera_snapshots(
                system_handles, t_stamp=t_capture, snapshot_dir=SNAPSHOT_DIR
            )
        except Exception:
            pass

    return detections, system_handles


# Start meshcat for visualization
meshcat = StartMeshcat()
print("Click the link above to open Meshcat in your browser!")
# Draw static camera markers in the main Meshcat viewer
try:
    camera_configs = wp.get_camera_configs(CAMERA_METHOD)
    draw_cameras(meshcat, camera_configs)
except Exception:
    pass


# In[ ]:


class RestingMoleController(LeafSystem):
    """
    Keeps all moles at a fixed position.
    For rest_height = 0.0, the mole base sits flush with the socket.
    """

    def __init__(self, plant, mole_joints, rest_height=0.):
        super().__init__()

        self.plant = plant
        self.mole_joints = mole_joints
        self.rest_height = rest_height   # height for ALL moles (usually 0.0)

        # Input: full plant state [q; v]
        self.state_in = self.DeclareVectorInputPort(
            "x", plant.num_multibody_states()
        )

        # One actuation output per mole
        self.mole_out = {}
        for ij, joint in mole_joints.items():
            self.mole_out[ij] = self.DeclareVectorOutputPort(
                f"mole_{ij[0]}_{ij[1]}_u", 1,
                (lambda ctx, out, ij=ij: self._CalcMoleControl(ctx, out, ij))
            )

    def _CalcMoleControl(self, context, output, ij):
        # Desired height is constant for all moles
        target = self.rest_height

        # Extract plant state
        x = self.state_in.Eval(context)
        nq = self.plant.num_positions()
        q = x[:nq]
        v = x[nq:]

        joint = self.mole_joints[ij]
        qj = q[joint.position_start()]
        vj = v[joint.velocity_start()]

        # PD control
        kp = 600.0
        kd = 12.0
        u = kp * (target - qj) - kd * vj

        output.SetFromVector([u])


class PoppingMoleController(LeafSystem):
    """
    A controller that keeps all moles resting at joint height = 0.0 (base touching socket)
    and repeatedly selects ONE mole at random to rise to a commanded height.
    """

    def __init__(self, plant, mole_joints,
                 rise_height=0.09,
                 rest_height=0.0,
                 min_up=1.5, max_up=1.5):
        super().__init__()

        self.plant = plant
        self.mole_joints = mole_joints
        self.rise_height = rise_height
        self.rest_height = rest_height

        self.min_up = min_up
        self.max_up = max_up

        # ---------------------------
        # State machine
        # ---------------------------
        # Pick the first active mole
        self.active = random.choice(list(mole_joints.keys()))
        # Set the time when we will switch to the next mole
        self.next_switch_time = random.uniform(min_up, max_up)

        # TODO: Track whether each mole is still allowed to pop up
        self.mole_alive_idx = self.DeclareAbstractState(
            AbstractValue.Make({ij: True for ij in self.mole_joints})
        )

        # Next time to switch the active mole
        self.next_switch_idx = self.DeclareAbstractState(
            AbstractValue.Make(random.uniform(min_up, max_up))
        )

        # Input: full plant state [q; v]
        self.state_in = self.DeclareVectorInputPort(
            "x", plant.num_multibody_states()
        )

        # Output: one actuation per mole
        self.mole_out = {}
        for ij, joint in mole_joints.items():
            self.mole_out[ij] = self.DeclareVectorOutputPort(
                f"mole_{ij[0]}_{ij[1]}_u", 1,
                (lambda ctx, out, ij=ij: self._CalcMoleControl(ctx, out, ij))
            )

    # --------------------------------------------------------------
    # Helper: pick a new mole different from the current one
    # --------------------------------------------------------------
    def _choose_new_mole(self):
        keys = list(self.mole_joints.keys())
        keys.remove(self.active)
        return random.choice(keys)

    # --------------------------------------------------------------
    # Main PD control calculation
    # --------------------------------------------------------------
    def _CalcMoleControl(self, context, output, ij):

        t = context.get_time()

        # --------------------------------------------
        # Random state switching: one mole at a time
        # --------------------------------------------
        if len(self.mole_joints) > 1:
            if t >= self.next_switch_time and ij == self.active:
                self.active = self._choose_new_mole()
                self.next_switch_time = t + random.uniform(self.min_up, self.max_up)

        # --------------------------------------------
        # Target height:
        #   active mole → rise_height
        #   others      → rest_height (0.0)
        # --------------------------------------------
        target = self.rise_height if ij == self.active else self.rest_height

        # --------------------------------------------
        # Extract q and v for this mole
        # --------------------------------------------
        x = self.state_in.Eval(context)
        nq = self.plant.num_positions()
        q = x[:nq]
        v = x[nq:]

        joint = self.mole_joints[ij]
        qj = q[joint.position_start()]
        vj = v[joint.velocity_start()]

        # --------------------------------------------
        # PD control
        # --------------------------------------------
        kp = 600.0
        kd = 12.0
        u = kp * (target - qj) - kd * vj

        output.SetFromVector([u])


# In[51]:


HAMMER_SDF_PATH = "/workspaces/whackamole/bonkbot/sim/assets/hammer.sdf"
GRID_SDF_PATH = "/workspaces/whackamole/bonkbot/sim/assets/grid_board.sdf"
MOLE_SDF_PATH = "/workspaces/whackamole/bonkbot/sim/assets/mole.sdf"

scenario_string = f"""directives:

# ===============================================================
# IIWA
# ===============================================================
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
    default_joint_positions:
      iiwa_joint_1: [-1.57]
      iiwa_joint_2: [0.1]
      iiwa_joint_3: [0]
      iiwa_joint_4: [-1.2]
      iiwa_joint_5: [0]
      iiwa_joint_6: [1.6]
      iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0

# ===============================================================
# Hammer
# ===============================================================
- add_model:
    name: hammer
    file: file://{HAMMER_SDF_PATH}
    default_free_body_pose:
      hammer_link:
        translation: [0, 0, 0]
        rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_weld:
    parent: iiwa::iiwa_link_7
    child: hammer::hammer_link
    X_PC:
      translation: [0, 0, 0.06]
      rotation: !Rpy {{deg: [0, -90, 0] }}

# ===============================================================
# Grid Board + Moles
# ===============================================================
- add_model:
    name: grid_board
    file: file://{GRID_SDF_PATH}

- add_weld:
    parent: world
    child: grid_board::board
    X_PC:
      translation: [0, -0.75, 0]
      rotation: !Rpy {{ deg: [0, 0, 0] }}

# 3x3 mole grid
- add_model:
    name: mole_0_0
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_0_0::socket
    X_PC: {{translation: [-0.2, -0.2, 0.0125]}}

- add_model:
    name: mole_0_1
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_0_1::socket
    X_PC: {{translation: [0.0, -0.2, 0.0125]}}

- add_model:
    name: mole_0_2
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_0_2::socket
    X_PC: {{translation: [0.2, -0.2, 0.0125]}}

- add_model:
    name: mole_1_0
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_1_0::socket
    X_PC: {{translation: [-0.2, 0.0, 0.0125]}}

- add_model:
    name: mole_1_1
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_1_1::socket
    X_PC: {{translation: [0.0, 0.0, 0.0125]}}

- add_model:
    name: mole_1_2
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_1_2::socket
    X_PC: {{translation: [0.2, 0.0, 0.0125]}}

- add_model:
    name: mole_2_0
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_2_0::socket
    X_PC: {{translation: [-0.2, 0.2, 0.0125]}}

- add_model:
    name: mole_2_1
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_2_1::socket
    X_PC: {{translation: [0.0, 0.2, 0.0125]}}

- add_model:
    name: mole_2_2
    file: file://{MOLE_SDF_PATH}
- add_weld:
    parent: grid_board::board
    child: mole_2_2::socket
    X_PC: {{translation: [0.2, 0.2, 0.0125]}}

model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
"""


# In[52]:


def get_prehit_pose(X_WO: RigidTransform, X_HL7: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
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


def get_mole_prehit_pose(X_WO: RigidTransform, X_HL7: RigidTransform):
    """
    Computes a prehit hammer pose over an object (mole).
    Align hammer z-axis downward, keep yaw fixed, position above mole.
    """
    # --- Desired prehit offset relative to mole center ---
    p_OH = np.array([0.0, 0.0, 0.12])   # 12 cm above mole, tune as needed

    # --- Build a stable orientation ---
    # z-axis: straight down in world
    z = np.array([0, 0, -1])

    # x-axis: choose a fixed world direction
    x = np.array([1, 0, 0])

    # Orthonormalize
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    R_WH = RotationMatrix(np.column_stack((x, y, z)))

    # --- Build world-frame hammer pose ---
    p_WH = X_WO.translation() + p_OH
    X_WH_prehit = RigidTransform(R_WH, p_WH)

    # --- Corresponding iiwa link 7 pose ---
    X_WL7_prehit = X_WH_prehit @ X_HL7

    return X_WH_prehit, X_WL7_prehit


def solve_ik(plant, plant_context, X_WH, q_guess=None, pos_tol=1e-3, theta_bound=1e-2):
    # Get the hammer face frame 
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_face_frame = plant.GetFrameByName("hammer_face",hammer)
    W = plant.world_frame()

    # Moles introduce extra DOFs
    iiwa = plant.GetModelInstanceByName("iiwa")

    # If user did not provide a guess, use iiwa’s current positions
    if q_guess is None:
        q_guess = plant.GetPositions(plant_context, iiwa)

    full_q_guess = plant.GetPositions(plant_context).copy()
    full_q_guess[0 : 7] = q_guess

    # Set up IK problem
    ik = inverse_kinematics.InverseKinematics(plant, plant_context)

    # Extract desired position/orientation from X_WG
    R_WH_desired = X_WH.rotation()
    p_WH_desired = X_WH.translation()

    # Position constraint: place origin of hammer face at p_WH_desired (within a small box)
    # Offset the hammer so it stays above the target (e.g., top of a mole)
    strike_offset = np.array([0, 0, 0])

    ik.AddPositionConstraint(
        frameA=W,
        p_BQ=np.zeros(3),        # target point in world
        frameB=hammer_face_frame,
        p_AQ_lower=(p_WH_desired + strike_offset) - pos_tol,
        p_AQ_upper=(p_WH_desired + strike_offset) + pos_tol,
    )

    # Orientation constraint: align world and hammer_face_frame up to some tolerance
    ik.AddOrientationConstraint(
        frameAbar=W,
        R_AbarA=RotationMatrix(),      # identity
        frameBbar=hammer_face_frame,
        R_BbarB=R_WH_desired,
        theta_bound=theta_bound,              # rad
    )

    # Add collision constraint to ensure collision free pre-hit pose
    # ik.AddMinimumDistanceConstraint(0.02)

    prog = ik.prog()
    prog.SetInitialGuess(ik.q(), full_q_guess)
    result = Solve(prog)

    if result.get_solution_result() != SolutionResult.kSolutionFound:
        return result.GetSolution(ik.q()), False
    return result.GetSolution(ik.q()), True


def make_joint_space_position_trajectory(path, timestep=1.0):
    traj = None
    times = [timestep * i for i in range(len(path))]
    Q = np.column_stack(path)
    traj = PiecewisePolynomial.FirstOrderHold(times, Q)
    return traj


# In[53]:


class HammerContactForce(LeafSystem):
    """
    Reads ContactResults from the plant and outputs the scalar contact force
    on the hammer body along n_hat (in world frame).
    """
    def __init__(self, plant, hammer_body_index, n_hat):
        super().__init__()
        self._plant = plant
        self._hammer_body_index = hammer_body_index
        self._n_hat = n_hat / np.linalg.norm(n_hat)

        # Abstract input: ContactResults
        self.DeclareAbstractInputPort(
            "contact_results",
            AbstractValue.Make(ContactResults())
        )

        # Scalar output: F_meas
        self.DeclareVectorOutputPort(
            "F_meas", BasicVector(1),
            self.CalcOutput
        )

    def CalcOutput(self, context, output):
        contact_results = self.get_input_port(0).Eval(context)
        F_W = np.zeros(3)

        for i in range(contact_results.num_point_pair_contacts()):
            info = contact_results.point_pair_contact_info(i)
            f_Bc_W = np.array(info.contact_force())
            bodyA = info.bodyA_index()
            bodyB = info.bodyB_index()

            if bodyA == self._hammer_body_index:
                F_W -= f_Bc_W    # equal/opposite on A
            if bodyB == self._hammer_body_index:
                F_W += f_Bc_W

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
    Output:
      q_cmd (7-vector)      - joint position command
    """
    def __init__(self, traj_approach, t_hit_start, hit_duration,
                 q_prehit, J_pinv, n_hat,
                 F_des,
                 M_a=1.0, D_a=40.0, K_a=200.0,
                 K_null=5.0,
                 retract_duration=1.5):
        super().__init__()

        self._traj = traj_approach
        self._t_hit_start = float(t_hit_start)
        self._hit_duration = float(hit_duration)

        self._q_prehit = q_prehit.copy().reshape(-1)   # 7
        self._J_pinv = J_pinv.copy()                   # (7, 3)
        self._n_hat = n_hat / np.linalg.norm(n_hat)

        self._F_des_hit = float(F_des)
        self._M_a = float(M_a)
        self._D_a = float(D_a)
        self._K_a = float(K_a)     # used mainly in retract
        self._K_null = float(K_null)  # posture spring gain
        self._retract_duration = float(retract_duration)

        # Continuous state: [s, s_dot]
        self.DeclareContinuousState(2)

        # Input 0: measured force along n_hat
        self.DeclareVectorInputPort("F_meas", BasicVector(1))
        # Input 1: measured joint positions (q_meas)
        self.DeclareVectorInputPort("q_meas", BasicVector(len(self._q_prehit)))

        # Output: joint position command q_cmd
        self.DeclareVectorOutputPort(
            "q_cmd", BasicVector(len(self._q_prehit)),
            self.CalcOutput
        )
                # Output 1: internal admittance state [s, s_dot]
        self.DeclareVectorOutputPort(
            "admittance_state", BasicVector(2),
            self.CalcStateOutput
        )

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
                F_err = 0.0   # <--- KEY CHANGE: no F_meas in retract

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
            # Phase 1: approach
            q_cmd = self._traj.value(t).flatten()

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


# In[61]:


scenario = LoadScenario(data=scenario_string)
station = MakeHardwareStation(scenario, meshcat=meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)
plant = station.GetSubsystemByName("plant")

temp_context = station.CreateDefaultContext()
temp_plant_context = plant.GetMyContextFromRoot(temp_context)

# # Build mole joints dictionary
mole_joints = {}
for i in range(3):
    for j in range(3):
        model_name = f"mole_{i}_{j}"
        instance = plant.GetModelInstanceByName(model_name)
        joint = plant.GetJointByName("mole_slider", instance)
        mole_joints[(i, j)] = joint

# Add mole controller
controller = builder.AddSystem(
    PoppingMoleController(
        plant,
        mole_joints,
        rise_height=0.09,
        rest_height=0.0,
        min_up=3,
        max_up=3
    )
)

# Connect plant state → controller
builder.Connect(
    station.GetOutputPort("state"),
    controller.state_in
)

# Connect each mole controller output → station actuation port
for i in range(3):
    for j in range(3):
        builder.Connect(
            controller.mole_out[(i, j)],
            station.GetInputPort(f"mole_{i}_{j}_actuation")
        )


# In[62]:


X_WGinitial = plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("iiwa_link_7"))
print(X_WGinitial)

# Get hammer head frame
hammer = plant.GetModelInstanceByName("hammer")
hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)
hammer_body = plant.GetBodyByName("hammer_link", hammer)
hammer_body_index = hammer_body.index()
X_WHammer = hammer_face_frame.CalcPoseInWorld(temp_plant_context)

def select_target_mole_pose(plant_context, meshcat=None) -> tuple[int, RigidTransform]:
    """
    Pick the mole pose to hit based on current plant state.
    Finds moles whose prismatic joint height > threshold, picks the highest;
    if multiple, chooses the one closest to the current hammer face.
    Returns (mole_index, X_WMole). If none are up, returns the last target or center mole.
    """
    height_thresh = 0.01
    up_moles = []
    for (i, j), joint in mole_joints.items():
        height = plant.GetPositions(plant_context)[joint.position_start()]
        if height > height_thresh:
            inst = plant.GetModelInstanceByName(f"mole_{i}_{j}")
            body = plant.GetBodyByName("mole", inst)
            X = plant.EvalBodyPoseInWorld(plant_context, body)
            up_moles.append((3 * i + j, X))

    if up_moles:
        max_z = max(X.translation()[2] for _, X in up_moles)
        eps = 1e-3
        candidates = [(idx, X) for idx, X in up_moles if X.translation()[2] >= max_z - eps]
        p_ref = hammer_face_frame.CalcPoseInWorld(plant_context).translation()
        idx, X_sel = min(candidates, key=lambda item: np.linalg.norm(item[1].translation() - p_ref))
        print(f"[perception] Target mole index: {idx}")
        return idx, X_sel

    # default to center mole
    inst = plant.GetModelInstanceByName("mole_1_1")
    body = plant.GetBodyByName("mole", inst)
    X = plant.EvalBodyPoseInWorld(plant_context, body)
    print("[perception] No moles up; holding center pre-hit.")
    return 4, X


# Get mole pose (perception if available, otherwise default mole_1_1 ground truth)
_, X_WMole = select_target_mole_pose(temp_plant_context, meshcat=meshcat)

# # get pose of iiwa link 7 in hammer face frame
iiwa = plant.GetModelInstanceByName("iiwa")
l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
X_HL7 = plant.CalcRelativeTransform(temp_plant_context, hammer_face_frame, l7_frame)

# get pre-hit pose frame
X_WH_prehit, X_WL7_prehit = get_mole_prehit_pose(X_WMole, X_HL7)
print("X_WH_prehit",X_WH_prehit)
print("X_WL7_prehit",X_WL7_prehit)

# visualize extra prehit frames
AddMeshcatTriad(
    meshcat,
    path="hammer_prehit_pose_triad",  
    length=0.1,
    radius=0.005,
    X_PT=X_WH_prehit,                 
)
AddMeshcatTriad(
    meshcat,
    path="iiwa_prehit_pose_triad",  
    length=0.1,
    radius=0.005,
    X_PT=X_WL7_prehit,                 
)

# Get initial positions of the iiwa joints
iiwa_q0 = plant.GetPositions(temp_plant_context, iiwa)
print("iiwa q0", iiwa_q0)

# solve ik for goal joint config
q_goal, optimal = solve_ik(plant, temp_plant_context, X_WH_prehit, q_guess=iiwa_q0,
                    pos_tol=1e-6, theta_bound=1e-3)
print(q_goal, optimal)

# linear joint space position trajectory
path = [iiwa_q0, q_goal[0:7,]]
traj = make_joint_space_position_trajectory(path)
# iiwa_src = builder.AddSystem(TrajectorySource(traj))
# builder.Connect(iiwa_src.get_output_port(), station.GetInputPort("iiwa.position"))


# In[63]:


# === Hit direction and hit parameters ===
# World positions
p_mole_W = X_WMole.translation()          # mole center in world
p_hammer_prehit_W = X_WH_prehit.translation()  # hammer pre-hit position in world

# Direction from hammer pre-hit -> mole
n_hat = p_mole_W - p_hammer_prehit_W      # vector pointing TOWARDS mole
n_hat = n_hat / np.linalg.norm(n_hat)

print("p_mole_W:", p_mole_W)
print("p_hammer_prehit_W:", p_hammer_prehit_W)
print("n_hat (hammer -> mole):", n_hat)

# Desired hit parameters
F_des = 200.0           # [N] desired normal force (tune)
v_pre = 1               # [m/s] initial velocity along n_hat at contact (tune)
hit_duration = 0.5      # [s] time to maintain the hit (tune)


# In[64]:


# === Compute Jacobian at pre-hit pose ===
J_context = plant.CreateDefaultContext()
plant.SetPositions(J_context, iiwa, q_goal[0:7,])

p_HQ_H = np.zeros(3)

# Full translational Jacobian: v_WQ = Jv_WQ * v
Jv_WQ_full = plant.CalcJacobianTranslationalVelocity(
    J_context,
    JacobianWrtVariable.kV,
    hammer_face_frame,   # hammer face frame
    p_HQ_H,              # origin of hammer face
    plant.world_frame(), # measure in world
    plant.world_frame(), # express in world
)
print("Jv_WQ_full shape:", Jv_WQ_full.shape)  # (3, 7)

# Extract iiwa portion of Jacobian
Jv_WQ = Jv_WQ_full[:, :7]
print("IIWA Jacobian shape:", Jv_WQ.shape)   # (3, 7)

# Full pseudoinverse: dx (3) -> dq (7)
J_pinv = np.linalg.pinv(Jv_WQ)      # shape (7, 3)
print("J_pinv shape:", J_pinv.shape)


# In[65]:


# === Add force and admittance systems via whack_force_admittance ===
params = AdmittanceParams(
    F_des=120.0,
    M_a=1.0,
    D_a=50.0,
    K_a=200.0,
    K_null=5.0,
    hit_duration=hit_duration,
    retract_duration=1.5,
    approach_timestep=max(traj.end_time(), 0.5),
    v_pre=0.0,
)

hit_handles = build_hit_admittance_pipeline(
    builder=builder,
    station=station,
    plant=plant,
    iiwa_model_instance=iiwa,
    hammer_body_index=hammer_body_index,
    hammer_face_frame=hammer_face_frame,
    q_current=iiwa_q0,
    X_WSoup=X_WMole,            # target mole pose as "soup"
    X_WH_prehit=X_WH_prehit,
    q_prehit=q_goal[0:7,],
    params=params,
)

diagram = builder.Build()

# Run simulation (multi-hit loop driven by perception)
simulator = Simulator(diagram)

ctx = simulator.get_mutable_context()

# Initialize iiwa joints to current q0
plant_ctx = plant.GetMyMutableContextFromRoot(ctx)
plant.SetPositions(plant_ctx, iiwa, iiwa_q0)

# Initialize admittance state [s, s_dot] = [0, v_pre]
reset_admittance_state_for_new_hit(hit_handles.hit_ctrl, ctx, v_pre=v_pre)

# Compute final time for approach + hit + retract
t_hit_start = hit_handles.t_hit_start  # initial, will be overwritten per hit

diagram.ForcedPublish(ctx)

meshcat.StartRecording()

# Multi-hit loop parameters
MAX_HITS = 5
MAX_TIME = 20.0
hits_done = 0
extra_margin = 0.2

last_idx = 4
last_prehit = X_WH_prehit

def _update_triad(path: str, X: RigidTransform):
    """Update an existing Meshcat triad transform."""
    if meshcat is not None:
        try:
            meshcat.SetTransform(path, X)
        except Exception:
            pass

while hits_done < MAX_HITS and ctx.get_time() < MAX_TIME:
    t_now = ctx.get_time()
    idx, X_WMole = select_target_mole_pose(plant_context=plant_ctx, meshcat=meshcat)

    # Get hammer → L7 transform (fixed)
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)
    iiwa = plant.GetModelInstanceByName("iiwa")
    l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
    X_HL7 = plant.CalcRelativeTransform(plant_ctx, hammer_face_frame, l7_frame)

    # Compute pre-hit pose and IK
    X_WH_prehit, X_WL7_prehit = get_mole_prehit_pose(X_WMole, X_HL7)
    q_current = plant.GetPositions(plant_ctx, iiwa)
    q_goal, _ = solve_ik(plant, plant_ctx, X_WH_prehit, q_guess=q_current,
                    pos_tol=1e-6, theta_bound=1e-3)

    # Update triads for visualization
    _update_triad("hammer_prehit_pose_triad", X_WH_prehit)
    _update_triad("iiwa_prehit_pose_triad", X_WL7_prehit)

    # If same target and pose as last loop, dwell briefly to avoid jerky resets
    if idx == last_idx and np.allclose(X_WH_prehit.translation(), last_prehit.translation(), atol=1e-3):
        simulator.AdvanceTo(min(t_now + 0.05, MAX_TIME))
        continue

    # Reconfigure admittance pipeline for this target
    traj, t_hit_start, n_hat, J_pinv = configure_hit_for_target(
        hit_handles=hit_handles,
        plant=plant,
        iiwa_model_instance=iiwa,
        hammer_face_frame=hammer_face_frame,
        X_WSoup=X_WMole,
        X_WH_prehit=X_WH_prehit,
        q_current=q_current,
        q_prehit=q_goal[0:7,],
        params=params,
        t_now=t_now,
    )

    # Reset admittance state for new hit
    reset_admittance_state_for_new_hit(hit_handles.hit_ctrl, ctx, v_pre=params.v_pre)

    # Advance through approach + hit + retract
    t_final = t_now + traj.end_time() + params.hit_duration + params.retract_duration + extra_margin
    simulator.AdvanceTo(t_final)
    hits_done += 1
    last_idx = idx
    last_prehit = X_WH_prehit

meshcat.StopRecording()
meshcat.PublishRecording()

# Plot force/admittance/joint traces using helper
plot_hit_results(hit_handles, root_context=ctx, plot=True, save_dir=Path("hit_plots"))

# In[67]:

# Optional: perception sanity check (disabled by default to keep runtime short).
RUN_PERCEPTION_SNAPSHOT = False
if RUN_PERCEPTION_SNAPSHOT:
    detections = run_perception_snapshot(t_capture=1.0, meshcat=meshcat)
    print("\nPerception snapshot results:")
    if not detections:
        print("  No 'up' moles detected.")
    else:
        for det in detections:
            p_W = det.X_W_mole.translation()
            rpy = det.X_W_mole.rotation().ToRollPitchYaw().vector()
            print(
                f"  mole_{det.mole_index}: "
                f"p_W = [{p_W[0]:.3f}, {p_W[1]:.3f}, {p_W[2]:.3f}], "
                f"rpy = [{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}]"
            )


# meshcat.Delete()


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=2500cebb-ccf6-40e5-a2c5-6c1beb3b7769' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
