'''
You still need to:

Fill in CAMERA_INTRINSICS with real numbers

Define MOLE_CENTERS_B from your actual CAD / board

Add a small top-level perceive_up_moles(...) wrapper

Tune thresholds and test in simulation

'''


# --- Imports for camera configuration --------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from copy import deepcopy
import os
from pathlib import Path

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.all import StartMeshcat
from pydrake.systems.sensors import (
    CameraConfig as DrakeCameraConfig,
    CameraInfo,
    RgbdSensor,
)
from pydrake.systems.primitives import ConstantVectorSource
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.common.yaml import yaml_dump
from pydrake.common import schema as drake_schema
from pydrake.geometry import RenderEngineVtkParams, MakeRenderEngineVtk
from pydrake.geometry import (
    RenderCameraCore,
    DepthRenderCamera,
    DepthRange,
    ClippingRange,
    Rgba,
    Box,
)
from pydrake.perception import DepthImageToPointCloud, BaseField, PointCloud
import numpy as np
from bonkbot.sim.simple_scene_builder_notebook import PoppingMoleController


# --- Mole layout in board frame B -----------------------------------------
# TODO: Replace these with the actual board layout from your CAD / SDF.
# Example: 3x3 grid, spacing 0.20 m (matches simple_scene_builder_notebook),
# centered at the origin in B.
MOLE_SPACING = 0.20  # [m] center-to-center in x and y
MOLE_GRID_OFFSET_X = -MOLE_SPACING   # so indices 0..2 span -S,0,+S
MOLE_GRID_OFFSET_Y = -MOLE_SPACING

MOLE_CENTERS_B = []
for iy in range(3):        # row
    for ix in range(3):    # col
        x_B = MOLE_GRID_OFFSET_X + ix * MOLE_SPACING
        y_B = MOLE_GRID_OFFSET_Y + iy * MOLE_SPACING
        z_B = 0.0          # on the board surface
        MOLE_CENTERS_B.append([x_B, y_B, z_B])

MOLE_CENTERS_B = np.array(MOLE_CENTERS_B)  # shape (9, 3)


# --- Camera configuration ---------------------------------------------------

@dataclass
class CameraConfig:
    """
    Simple config for one RGB-D camera.

    Fields:
      name          : string name (used in scenario / port names)
      parent_frame  : name of the frame this camera is attached to (usually "world")
      position_W    : 3D position of the camera in world frame [x, y, z]
      rpy_W         : roll, pitch, yaw of the camera in world frame (radians)
                      (used later to build X_WC or scenario.yaml entries)
    """
    name: str
    parent_frame: str
    position_W: np.ndarray
    rpy_W: np.ndarray


# ---- High-level knobs you can change --------------------------------------

# How many RGB-D cameras you want around the board.
NUM_CAMERAS = 4   # <-- change this to 1, 2, 3, 4, ...
# Draw extra Meshcat markers? False to rely on camera_box model visuals.
DRAW_CAMERA_MARKERS = False

# Approximate board center in WORLD frame (matches grid_board weld in scenario).
# The board is welded at translation [0, -0.75, 0.25] in world.
BOARD_CENTER_W = np.array([0.0, -0.75, 0.25])

# Radius of the camera ring around the board, and camera height.
CAMERA_RADIUS = 1.2   # [m] distance from board center in XY plane
CAMERA_HEIGHT = 1.0   # [m] height above world origin
# Optional yaw offset to rotate the ring and avoid occlusion (e.g., behind iiwa).
CAMERA_RING_YAW_OFFSET = np.pi / 4.0


def compute_camera_rpy_looking_at_board(position_W: np.ndarray,
                                        board_center_W: np.ndarray,
                                        tilt_down: float = 0.0) -> np.ndarray:
    """
    Build an orientation that points the camera +Z axis toward the board center.
    Drake's RenderCamera convention: +Z looks forward, +X to the right,
    +Y down in the image.
    """
    z_axis = board_center_W - position_W
    z_axis = z_axis / np.linalg.norm(z_axis)
    # Choose an "up" reference and build x/y
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = RotationMatrix(np.column_stack([x_axis, y_axis, z_axis]))
    rpy = RollPitchYaw(R).vector()
    # Optional extra tilt about camera X to pitch down/up if desired.
    rpy[0] += tilt_down
    return rpy


def _maybe_draw_camera_models(meshcat, camera_configs: list[CameraConfig]):
    """
    Add simple camera boxes into Meshcat for visualization. Safe no-op if
    meshcat is None. Dimensions / color roughly mirror the camera_box.sdf
    used in the reference notebook so the visuals look familiar.
    """
    if meshcat is None or not DRAW_CAMERA_MARKERS:
        return
    cam_box = Box(0.08, 0.05, 0.05)  # depth, width, height (approx camera_box.sdf)
    cam_color = Rgba(0.15, 0.15, 0.18, 0.9)
    for cfg in camera_configs:
        obj_path = f"perception_cameras/{cfg.name}"
        X_WC = RigidTransform(RollPitchYaw(cfg.rpy_W), cfg.position_W)
        meshcat.SetObject(obj_path, cam_box, cam_color)
        meshcat.SetTransform(obj_path, X_WC)


def make_default_camera_configs() -> list[CameraConfig]:
    """
    Create NUM_CAMERAS configs on a circle around the board center.

    Cameras are placed evenly around a circle of radius CAMERA_RADIUS
    at height CAMERA_HEIGHT, and oriented roughly toward the board center.
    """
    configs: list[CameraConfig] = []

    for k in range(NUM_CAMERAS):
        theta = CAMERA_RING_YAW_OFFSET + 2.0 * np.pi * k / NUM_CAMERAS  # angle around the board
        x = BOARD_CENTER_W[0] + CAMERA_RADIUS * np.cos(theta)
        y = BOARD_CENTER_W[1] + CAMERA_RADIUS * np.sin(theta)
        z = CAMERA_HEIGHT
        position_W = np.array([x, y, z])

        rpy_W = compute_camera_rpy_looking_at_board(position_W, BOARD_CENTER_W)

        configs.append(
            CameraConfig(
                name=f"cam{k}",
                parent_frame="world",
                position_W=position_W,
                rpy_W=rpy_W,
            )
        )

    return configs


def make_notebook_camera_configs() -> list[CameraConfig]:
    """
    Use the ring formation, but keep the notebook-style camera visuals
    (camera_box models) added via directives.
    """
    return make_default_camera_configs()


def get_camera_configs(method: str = "ring") -> list[CameraConfig]:
    """
    method == "ring"      -> cameras on a circle around BOARD_CENTER_W
    method == "notebook"  -> three fixed cameras like the screenshots
    """
    method = (method or "ring").lower()
    if method == "notebook":
        return make_notebook_camera_configs()
    return make_default_camera_configs()


# This is the main list you'll use everywhere else in the perception code by default.
CAMERA_CONFIGS = make_default_camera_configs()

#_________________________#

# Optional default intrinsics (if all cameras share the same model).
# These correspond to a 640x480 pinhole camera with ~45 deg vertical FOV.
DEFAULT_INTRINSICS = {"fx": 580.0, "fy": 580.0, "cx": 320.0, "cy": 240.0}
CAMERA_INTRINSICS = {
    cfg.name: DEFAULT_INTRINSICS
    for cfg in CAMERA_CONFIGS
}
RENDERER_NAME = "vtk"


def camera_configs_to_scenario_block(
    camera_configs: List[CameraConfig],
    width: int = 640,
    height: int = 480,
) -> Dict[str, Dict[str, Any]]:
    """
    Convert CameraConfig objects into the mapping expected by
    manipulation.station.Scenario.cameras (as YAML-ready dicts).
    """
    cameras_block: Dict[str, Dict[str, Any]] = {}

    for cfg in camera_configs:
        cam_entry = {
            "name": cfg.name,
            "rgb": True,
            "depth": True,
            "width": width,
            "height": height,
            "X_PB": {
                "base_frame": cfg.parent_frame,
                "translation": cfg.position_W.tolist(),
            },
        }
        cameras_block[cfg.name] = cam_entry

    return cameras_block

def make_scenario_with_cameras(
    base_scenario_dict: Dict[str, Any],
    camera_configs: list[CameraConfig],
) -> Dict[str, Any]:
    """
    Take an existing scenario dict (iiwa + hammer + board, etc.),
    append camera entries based on CAMERA_CONFIGS, and return a new dict.
    Also add simple visual camera models via directives so they show up in Meshcat.
    """
    scenario_dict = deepcopy(base_scenario_dict)

    # Ensure there's a "cameras" mapping
    if "cameras" not in scenario_dict or scenario_dict["cameras"] is None:
        scenario_dict["cameras"] = {}
    elif isinstance(scenario_dict["cameras"], list):
        scenario_dict["cameras"] = {
            c.get("name", f"cam{idx}"): c for idx, c in enumerate(scenario_dict["cameras"])
        }

    # Build camera blocks and merge (no explicit renderer; use station default)
    cam_block: Dict[str, Dict[str, Any]] = {}
    directives: list = scenario_dict.get("directives", [])

    for cfg in camera_configs:
        cam_block[cfg.name] = {
            "name": cfg.name,
            "rgb": True,
            "depth": True,
            "X_PB": {
                "base_frame": f"{cfg.name}::base",
                "translation": [0.0, 0.0, 0.0],
            },
        }

        # Add visual camera models via directives (visible in Meshcat)
        X_PF = {
            "base_frame": cfg.parent_frame,
            "translation": cfg.position_W.tolist(),
        }

        directives.extend([
            {
                "add_frame": {
                    "name": f"{cfg.name}_origin",
                    "X_PF": X_PF,
                }
            },
            {
                "add_model": {
                    "name": cfg.name,
                    "file": "package://manipulation/camera_box.sdf",
                }
            },
            {
                "add_weld": {
                    "parent": f"{cfg.name}_origin",
                    "child": f"{cfg.name}::base",
                }
            },
        ])

    scenario_dict["cameras"].update(cam_block)
    scenario_dict["directives"] = directives

    return scenario_dict

@dataclass
class CameraPorts:
    """
    Holds the Drake output ports for a single RGB-D camera in the station.
    We don't hard-type the ports; we just store the handles.

    Fields:
      name       : camera name (e.g. "cam0")
      rgb_port   : output port for the color image
      depth_port : output port for the float depth image
      X_WC_port  : output port for the pose of the camera in world frame (if available)
      X_WC_static: RigidTransform for the camera pose (if no port is exported)
    """
    name: str
    rgb_port: Any
    depth_port: Any
    X_WC_port: Any | None = None
    X_WC_static: RigidTransform | None = None

def initialize_whack_perception_system(
    base_scenario_dict: Dict[str, Any],
    meshcat=None,
    camera_method: str = "ring",
):
    """
    Build the Drake station/diagram for the whack-a-mole scene with
    NUM_CAMERAS RGB-D cameras, and return handles for:
      - diagram, simulator, station, plant
      - camera_ports: list[CameraPorts]
    """

    print("[perception] Loading base scenario...")
    camera_configs = get_camera_configs(camera_method)
    intrinsics_map = {cfg.name: DEFAULT_INTRINSICS for cfg in camera_configs}

    # Build Scenario from base YAML, then add cameras as Drake configs.
    scenario_with_cams = make_scenario_with_cameras(
        base_scenario_dict,
        camera_configs=camera_configs,
    )
    scenario = LoadScenario(data=yaml_dump(scenario_with_cams))

    # Allow VTK to render headlessly if possible.
    os.environ.setdefault("PYDRAKE_ALLOW_VTK_HEADLESS", "1")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    # Skip pyvirtualdisplay startup (which hangs without Xvfb) by providing a dummy
    # DISPLAY when none is set; Drake will still render headless via OSMesa.
    os.environ.setdefault("DISPLAY", ":0")

    # Keep Meshcat visuals alive and make sure illustration is published.
    scenario.visualization.delete_on_initialization_event = False
    scenario.visualization.publish_illustration = True

    print("[perception] Added cameras; building station...")
    # 3. Build station from scenario
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    print("[perception] Station built.")

    # 3a. Ensure a depth-capable renderer exists (for RGB-D cameras).
    try:
        scene_graph = station.GetSubsystemByName("scene_graph")
        engine = MakeRenderEngineVtk(RenderEngineVtkParams())
        if not scene_graph.HasRendererName("vtk"):
            scene_graph.AddRenderer("vtk", engine)
            print("[perception] Added VTK renderer to scene_graph.")
        else:
            print("[perception] VTK renderer already present.")
    except Exception as e:
        print(f"[perception] Warning: could not add VTK renderer (may already exist): {e}")

    # 4. Diagram + plant
    builder = DiagramBuilder()
    builder.AddSystem(station)
    print("[perception] Station added to builder.")
    plant = station.GetSubsystemByName("plant")

    # Feed a fixed iiwa position command so the velocity interpolator input is satisfied.
    iiwa_instance = plant.GetModelInstanceByName("iiwa")
    q0 = plant.GetPositions(plant.CreateDefaultContext(), iiwa_instance)
    iiwa_const = builder.AddSystem(ConstantVectorSource(q0))
    builder.Connect(iiwa_const.get_output_port(), station.GetInputPort("iiwa.position"))

    # Add popping mole controller so moles move as in the notebook simulation.
    mole_joints: Dict[Tuple[int, int], Any] = {}
    for i in range(3):
        for j in range(3):
            model_name = f"mole_{i}_{j}"
            instance = plant.GetModelInstanceByName(model_name)
            joint = plant.GetJointByName("mole_slider", instance)
            mole_joints[(i, j)] = joint

    controller = builder.AddSystem(
        PoppingMoleController(
            plant,
            mole_joints,
            rise_height=0.09,
            rest_height=0.0,
            min_up=3,
            max_up=3,
        )
    )
    builder.Connect(station.GetOutputPort("state"), controller.state_in)
    for i in range(3):
        for j in range(3):
            builder.Connect(
                controller.mole_out[(i, j)],
                station.GetInputPort(f"mole_{i}_{j}_actuation"),
            )

    # 5. Add explicit RgbdSensors + DepthImageToPointCloud for each camera config.
    #    We ignore any station-provided camera ports to ensure we only use VTK depth.
    camera_ports: List[CameraPorts] = []
    pc_ports: Dict[str, Any] = {}

    scene_graph = station.GetSubsystemByName("scene_graph")
    query_port = station.GetOutputPort("query_object")
    world_frame_id = scene_graph.world_frame_id()

    # Ensure VTK renderer is registered exactly once.
    try:
        scene_graph.AddRenderer(RENDERER_NAME, MakeRenderEngineVtk(RenderEngineVtkParams()))
        print(f"[perception] Added {RENDERER_NAME} renderer to scene_graph.")
    except Exception:
        # Renderer may already exist; ignore.
        pass

    for cfg in camera_configs:
        intr = intrinsics_map.get(cfg.name, DEFAULT_INTRINSICS)
        X_WB = RigidTransform(
            RollPitchYaw(cfg.rpy_W).ToRotationMatrix(),
            cfg.position_W,
        )

        # Make sure VTK renderer exists in SceneGraph before creating sensors.
        sensor = builder.AddSystem(
            RgbdSensor(
                parent_id=world_frame_id,
                X_PB=X_WB,
                depth_camera=DepthRenderCamera(
                    RenderCameraCore(
                        RENDERER_NAME,
                        CameraInfo(
                            width=640,
                            height=480,
                            focal_x=intr["fx"],
                            focal_y=intr["fy"],
                            center_x=intr["cx"],
                            center_y=intr["cy"],
                        ),
                        ClippingRange(0.1, 5.0),
                        RigidTransform(),
                    ),
                    DepthRange(0.1, 5.0),
                ),
                show_window=False,
            )
        )
        sensor.set_name(cfg.name)
        builder.Connect(query_port, sensor.query_object_input_port())

        pc_sys = builder.AddSystem(
            DepthImageToPointCloud(
                camera_info=CameraInfo(
                    width=640,
                    height=480,
                    focal_x=intr["fx"],
                    focal_y=intr["fy"],
                    center_x=intr["cx"],
                    center_y=intr["cy"],
                ),
                fields=BaseField.kXYZs | BaseField.kRGBs,
            )
        )
        pc_sys.set_name(f"{cfg.name}.pc")
        builder.Connect(sensor.depth_image_32F_output_port(), pc_sys.depth_image_input_port())
        builder.Connect(sensor.color_image_output_port(), pc_sys.color_image_input_port())
        builder.Connect(sensor.body_pose_in_world_output_port(), pc_sys.GetInputPort("camera_pose"))

        pc_ports[cfg.name] = pc_sys

        camera_ports.append(
            CameraPorts(
                name=cfg.name,
                rgb_port=sensor.color_image_output_port(),
                depth_port=sensor.depth_image_32F_output_port(),
                X_WC_port=sensor.body_pose_in_world_output_port(),
                X_WC_static=X_WB,
            )
        )

    # Draw camera boxes in Meshcat if available.
    _maybe_draw_camera_models(meshcat, camera_configs)

    # 6. Build diagram + simulator
    diagram = builder.Build()
    print("[perception] Diagram built.")
    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()
    print("[perception] Simulator created.")

    return {
        "diagram": diagram,
        "simulator": simulator,
        "root_context": root_context,
        "station": station,
        "plant": plant,
        "camera_ports": camera_ports,
        "point_cloud_ports": pc_ports,
        "camera_configs": camera_configs,
        "camera_intrinsics": intrinsics_map,
    }

@dataclass
class CameraMeasurement:
    """
    One RGB-D snapshot from a single camera.

    Fields:
      name    : camera name (e.g. "cam0")
      rgb     : Drake ImageRgba8U (or similar) from the station port
      depth   : Drake ImageDepth32F  (float depth in meters)
      X_WC    : RigidTransform (pose of camera C in world frame W)
      P_W     : (N, 3) numpy array of 3D points in world frame (optional, can be filled later)
    """
    name: str
    rgb: Any
    depth: Any
    X_WC: Any
    P_W: np.ndarray | None = None


def depth_to_points_C(
    depth_image,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """
    Convert a Drake ImageDepth32F (or similar) into a set of 3D points in
    the camera frame C.

    Returns:
      P_C: (N, 3) numpy array of [x_C, y_C, z_C] points.

    Notes:
      - depth_image is assumed to have shape (height, width)
        and contain depth in meters.
      - We ignore pixels with depth <= 0.
    """
    # Convert Drake image to numpy array if needed
    H = depth_image.height()
    W = depth_image.width()
    data_array = np.array(depth_image.data, copy=False)
    if data_array.size == H * W:
        D = data_array.reshape((H, W)).astype(np.float32)
    elif data_array.ndim == 3 and data_array.shape[0] == H:
        D = data_array[:, :, 0].astype(np.float32)
    else:
        byte_count = len(depth_image.data)
        expected_float_bytes = H * W * 4  # float32
        expected_uint16_bytes = H * W * 2
        if byte_count == expected_float_bytes:
            D = np.frombuffer(depth_image.data, dtype=np.float32).reshape((H, W))
        elif byte_count == expected_uint16_bytes:
            D_mm = np.frombuffer(depth_image.data, dtype=np.uint16).reshape((H, W))
            D = D_mm.astype(np.float32) / 1000.0  # convert mm to meters
        else:
            raise RuntimeError(f"Unexpected depth image buffer size: {byte_count} bytes")


    # Build a grid of pixel coordinates
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten
    u_flat = u_coords.reshape(-1)
    v_flat = v_coords.reshape(-1)
    d_flat = D.reshape(-1) * depth_scale  # [m]

    # Mask out invalid depths
    valid = np.isfinite(d_flat) & (d_flat > 0.0)
    u = u_flat[valid]
    v = v_flat[valid]
    d = d_flat[valid]

    # Back-project:
    # x_C = (u - cx) / fx * d
    # y_C = (v - cy) / fy * d
    # z_C = d
    x_C = (u - cx) / fx * d
    y_C = (v - cy) / fy * d
    z_C = d

    P_C = np.stack([x_C, y_C, z_C], axis=1)  # (N, 3)
    return P_C

def points_C_to_W(P_C: np.ndarray, X_WC) -> np.ndarray:
    """
    Transform a set of 3D points in camera frame C to world frame W
    using Drake's RigidTransform X_WC.
    """
    R_WC = X_WC.rotation().matrix()   # 3x3
    p_WC = X_WC.translation()         # 3

    # P_W = R_WC @ P_C.T + p_WC[:, None]  → then transpose back
    P_W = (R_WC @ P_C.T) + p_WC.reshape(3, 1)
    return P_W.T  # (N, 3)

def capture_multi_camera_pointcloud(
    system_handles: Dict[str, Any],
    camera_intrinsics: Dict[str, Dict[str, float]],
    t_capture: float,
) -> Tuple[List[CameraMeasurement], np.ndarray]:
    """
    Advance the simulator to time t_capture, read out RGB, depth, and X_WC
    from all cameras, back-project depth to 3D, and return:

      - per-camera CameraMeasurement objects (with P_W filled), and
      - a single merged world-frame point cloud P_W_all (N_total, 3).

    camera_intrinsics:
      {
        "cam0": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
        "cam1": {...},
        ...
      }
    """

    diagram = system_handles["diagram"]
    simulator = system_handles["simulator"]
    root_context = system_handles["root_context"]
    station = system_handles["station"]
    camera_ports: List[CameraPorts] = system_handles["camera_ports"]
    pc_ports = system_handles.get("point_cloud_ports", {})

    # 1. Advance simulation to desired time (you can decide when)
    print("[perception] Advancing simulator...")
    simulator.AdvanceTo(t_capture)
    print("[perception] Simulator advanced.")
    station_ctx = station.GetMyContextFromRoot(root_context)

    #ctx = diagram.GetMyContextFromRoot(root_context)
    ctx = station_ctx
    measurements: List[CameraMeasurement] = []
    all_points_W: List[np.ndarray] = []

    for cam_ports in camera_ports:
        name = cam_ports.name

        # Prefer point cloud port if available (already in world frame)
        pc_system = pc_ports.get(name, None)
        if pc_system is not None:
            pc_port = pc_system.get_output_port()
            pc_ctx = pc_system.GetMyContextFromRoot(root_context)
            pc = pc_port.Eval(pc_ctx)
            P_W = pc.xyzs().T  # (N,3)
            all_points_W.append(P_W)
            measurements.append(
                CameraMeasurement(
                    name=name,
                    rgb=None,
                    depth=None,
                    X_WC=cam_ports.X_WC_static,
                    P_W=P_W,
                )
            )
            print(f"[perception] {name} point cloud via AddPointClouds: {P_W.shape[0]} points.")
            continue

        # 2. Evaluate camera outputs from the station
        rgb = cam_ports.rgb_port.Eval(ctx)
        depth = cam_ports.depth_port.Eval(ctx)
        if depth.width() == 0 or depth.height() == 0:
            print(f"[perception] {name} depth image has zero size ({depth.width()}x{depth.height()})")
        else:
            print(f"[perception] {name} depth size {depth.width()}x{depth.height()}, buffer bytes {len(depth.data)}")
        if cam_ports.X_WC_port is not None:
            X_WC = cam_ports.X_WC_port.Eval(ctx)
        else:
            X_WC = cam_ports.X_WC_static
        print(f"[perception] Camera {name} outputs acquired.")

        # 3. Get intrinsics for this camera
        intr = camera_intrinsics.get(name, None)
        if intr is None:
            raise ValueError(f"No intrinsics provided for camera '{name}'")

        fx = intr["fx"]
        fy = intr["fy"]
        cx = intr["cx"]
        cy = intr["cy"]

        # 4. Back-project depth to 3D in camera frame
        depth_array = np.frombuffer(depth.data, dtype=np.float32)
        if depth_array.size > 0:
            print(f"[perception] {name} depth stats: min {np.nanmin(depth_array):.4f}, max {np.nanmax(depth_array):.4f}, finite {np.isfinite(depth_array).sum()}/{depth_array.size}")
        P_C = depth_to_points_C(depth, fx=fx, fy=fy, cx=cx, cy=cy)

        # 5. Transform to world frame
        P_W = points_C_to_W(P_C, X_WC)

        all_points_W.append(P_W)

        measurements.append(
            CameraMeasurement(
                name=name,
                rgb=rgb,
                depth=depth,
                X_WC=X_WC,
                P_W=P_W,
            )
        )
        print(f"[perception] Camera {name} processed ({P_W.shape[0]} points).")

    # 6. Merge all camera points into a single big cloud
    if len(all_points_W) > 0:
        P_W_all = np.vstack(all_points_W)
    else:
        P_W_all = np.zeros((0, 3))

    # Downsample to keep downstream processing fast.
    max_points = 200000
    if P_W_all.shape[0] > max_points:
        idx = np.random.choice(P_W_all.shape[0], max_points, replace=False)
        P_W_all = P_W_all[idx]

    return measurements, P_W_all

def ransac_plane_fit(
    P_W: np.ndarray,
    num_iters: int = 200,
    distance_thresh: float = 0.01,
    min_inlier_ratio: float = 0.3,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Very simple RANSAC plane fit:

      Plane: n_hat^T x + d = 0, where ||n_hat|| = 1.

    Inputs:
      P_W            : (N, 3) world points
      num_iters      : number of random samples
      distance_thresh: inlier threshold [m]
      min_inlier_ratio: if best inlier ratio < this, we warn/raise

    Returns:
      n_hat_W        : (3,) unit normal vector in world
      d              : scalar plane offset (n_hat^T x + d = 0)
      inlier_mask    : (N,) boolean array: True for inliers
    """
    N = P_W.shape[0]
    if N < 3:
        raise ValueError("Not enough points for plane fit")

    best_inliers = None
    best_count = 0
    best_n = None
    best_d = None

    for _ in range(num_iters):
        # Randomly sample 3 distinct points
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = P_W[idx]

        # Compute normal from cross product
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8:
            continue  # degenerate sample (nearly collinear)

        n_hat = n / norm_n
        d = -np.dot(n_hat, p1)

        # Distances of all points to this plane
        distances = np.abs(P_W @ n_hat + d)
        inliers = distances < distance_thresh
        count = np.count_nonzero(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_n = n_hat
            best_d = d

    if best_inliers is None:
        raise RuntimeError("RANSAC failed to find a plane")

    inlier_ratio = best_count / N
    if inlier_ratio < min_inlier_ratio:
        print(f"[WARN] RANSAC plane inlier ratio is low: {inlier_ratio:.3f}")

    return best_n, best_d, best_inliers

def make_board_frame_from_plane(
    P_W: np.ndarray,
    inlier_mask: np.ndarray,
    n_hat_W: np.ndarray,
) -> RigidTransform:
    """
    Construct a world->board transform X_WB from:

      - inlier points P_W (N, 3) and inlier_mask
      - plane normal n_hat_W

    z_B is aligned with the plane normal (flipped to mostly +z if needed).
    x_B is chosen as the projection of world x-axis onto the plane.
    The origin p_WB is the centroid of the inlier points.
    """
    # 1. Flip normal if it points mostly down
    if n_hat_W[2] < 0.0:
        n_hat_W = -n_hat_W

    z_B = n_hat_W  # unit-length, world-frame

    # 2. Choose x_B as projection of world x-axis onto plane
    x_world = np.array([1.0, 0.0, 0.0])
    x_proj = x_world - np.dot(x_world, z_B) * z_B
    norm_x = np.linalg.norm(x_proj)
    if norm_x < 1e-8:
        # If that fails (z_B almost parallel to x), use world y instead
        y_world = np.array([0.0, 1.0, 0.0])
        x_proj = y_world - np.dot(y_world, z_B) * z_B
        norm_x = np.linalg.norm(x_proj)
    x_B = x_proj / norm_x

    # 3. y_B = z_B x x_B to complete right-handed frame
    y_B = np.cross(z_B, x_B)

    # 4. Centroid of inlier points as board origin
    P_inliers = P_W[inlier_mask]
    if P_inliers.shape[0] == 0:
        raise RuntimeError("No inliers passed to make_board_frame_from_plane")
    p_WB = np.mean(P_inliers, axis=0)

    # 5. Build rotation + transform
    R_WB = np.column_stack([x_B, y_B, z_B])  # columns = basis vectors
    R = RotationMatrix(R_WB)
    X_WB = RigidTransform(R, p_WB)

    return X_WB

def estimate_board_frame_from_cloud(
    P_W_all: np.ndarray,
    ransac_distance_thresh: float = 0.01,
) -> Tuple[RigidTransform, np.ndarray, float, np.ndarray]:
    """
    High-level wrapper:

      1. Run RANSAC plane fit on P_W_all.
      2. Build X_WB (world->board) from the plane and its inliers.

    Returns:
      X_WB         : RigidTransform
      n_hat_W      : (3,) unit plane normal in world
      d            : plane offset
      inlier_mask  : (N,) boolean mask
    """
    n_hat_W, d, inliers = ransac_plane_fit(
        P_W_all,
        distance_thresh=ransac_distance_thresh,
    )

    X_WB = make_board_frame_from_plane(P_W_all, inliers, n_hat_W.copy())

    return X_WB, n_hat_W, d, inliers

def transform_points_to_board_frame(P_W: np.ndarray, X_WB: RigidTransform) -> np.ndarray:
    """
    Transform a (N, 3) point cloud from world frame W to board frame B.

    X_WB : pose of board frame B in world
    """
    X_BW = X_WB.inverse()          # board-from-world
    R_BW = X_BW.rotation().matrix()
    p_BW = X_BW.translation()

    # P_B = R_BW @ P_W^T + p_BW, then transpose
    P_B = (R_BW @ P_W.T) + p_BW.reshape(3, 1)
    return P_B.T   # (N, 3)

def filter_points_above_board(
    P_B: np.ndarray,
    z_min: float = 0.01,
    z_max: float = 0.20,
) -> np.ndarray:
    """
    Keep only points whose height in board frame is between z_min and z_max.

    z_min: lower bound (ignore points basically on the board or below)
    z_max: upper bound (ignore unlikely outliers way above the board)
    """
    z = P_B[:, 2]
    mask = (z > z_min) & (z < z_max)
    return P_B[mask], mask

def assign_points_to_moles(
    P_B_above: np.ndarray,
    mole_centers_B: np.ndarray,
    max_xy_dist: float = 0.08,
) -> List[np.ndarray]:
    """
    Assign each point in P_B_above to the nearest mole center in (x,y),
    if it's within max_xy_dist.

    Returns:
      mole_points: list of length num_moles, where mole_points[i] is an
                   array of points assigned to mole i (shape (Ni, 3)).
    """
    num_moles = mole_centers_B.shape[0]
    mole_points = [ [] for _ in range(num_moles) ]

    # Extract x,y from points
    x = P_B_above[:, 0]
    y = P_B_above[:, 1]
    points_xy = np.stack([x, y], axis=1)    # (N, 2)

    mole_xy = mole_centers_B[:, :2]         # (M, 2)

    for p_idx in range(points_xy.shape[0]):
        p_xy = points_xy[p_idx]

        # Distances to each mole center in x–y
        diffs = mole_xy - p_xy
        dists = np.linalg.norm(diffs, axis=1)
        mole_idx = np.argmin(dists)
        if dists[mole_idx] < max_xy_dist:
            mole_points[mole_idx].append(P_B_above[p_idx])

    # Convert lists → numpy arrays
    mole_points = [np.array(pts) if len(pts) > 0 else np.zeros((0, 3)) 
                   for pts in mole_points]

    return mole_points

@dataclass
class MoleDetection:
    mole_index: int
    mean_height_B: float
    num_points: int
    center_B: np.ndarray   # (3,) center on board surface
    X_W_mole: RigidTransform  # pose of mole top in world

def make_mole_pose_in_world(
    X_WB: RigidTransform,
    center_B: np.ndarray,
    height_B: float,
) -> RigidTransform:
    """
    Construct a world pose for the mole 'top' frame:

      - orientation: same as the board (z_B up)
      - position   : board center + [x_B, y_B, height_B]

    center_B: (3,) expected mole center on board surface (z_B ~ 0).
    height_B: mean height above board.
    """
    R_WB = X_WB.rotation()
    p_WB = X_WB.translation()

    p_B_mole = center_B.copy()
    p_B_mole[2] = height_B

    p_W_mole = p_WB + R_WB.multiply(p_B_mole)
    return RigidTransform(R_WB, p_W_mole)

def detect_up_moles_from_cloud(
    P_W_all: np.ndarray,
    X_WB: RigidTransform,
    mole_centers_B: np.ndarray,
    z_min: float = 0.01,
    z_max: float = 0.20,
    max_xy_dist: float = 0.08,
    min_points: int = 30,
    height_threshold: float = 0.03,
) -> List[MoleDetection]:
    """
    Full pipeline in board frame:

      1. Transform world cloud → board frame.
      2. Filter points with z_B in [z_min, z_max].
      3. Assign those points to nearest mole centers (within max_xy_dist).
      4. For each mole, compute mean z_B and num_points.
      5. If num_points >= min_points AND mean z_B >= height_threshold,
         mark as 'up' and build X_W_mole.

    Returns:
      List of MoleDetection objects, one per detected 'up' mole.
    """
    # 1. W -> B
    P_B_all = transform_points_to_board_frame(P_W_all, X_WB)

    # 2. Filter on height
    P_B_above, _ = filter_points_above_board(P_B_all, z_min=z_min, z_max=z_max)

    # 3. Assign to moles
    mole_points_B = assign_points_to_moles(P_B_above, mole_centers_B, max_xy_dist=max_xy_dist)

    detections: List[MoleDetection] = []

    for idx, pts_B in enumerate(mole_points_B):
        if pts_B.shape[0] < min_points:
            continue  # not enough evidence

        mean_z = float(np.mean(pts_B[:, 2]))
        if mean_z < height_threshold:
            continue  # not tall enough to be considered "up"

        center_B = mole_centers_B[idx]
        X_W_mole = make_mole_pose_in_world(X_WB, center_B, mean_z)

        detections.append(
            MoleDetection(
                mole_index=idx,
                mean_height_B=mean_z,
                num_points=pts_B.shape[0],
                center_B=center_B,
                X_W_mole=X_W_mole,
            )
        )

    return detections


def print_mole_poses_from_plant(
    mole_poses,
    mode: str = "both",
    top_offset: np.ndarray | None = None,
) -> None:
    """
    Pretty-print mole poses from a plant state. Useful for debugging with
    different scenarios.
    """
    if top_offset is None:
        top_offset = np.array([0.0, 0.0, 0.10])

    mode = mode.lower()
    base_lines = []
    top_lines = []
    for (i, j), X_WM in sorted(mole_poses.items()):
        p_base = X_WM.translation()
        rpy = X_WM.rotation().ToRollPitchYaw().vector()
        p_top = p_base + X_WM.rotation().matrix().dot(top_offset)

        base_lines.append(
            f"  mole_{i}_{j} base: p_W = [{p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f}], "
            f"rpy_W = [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
        )
        top_lines.append(
            f"  mole_{i}_{j} top : p_W = [{p_top[0]:.3f}, {p_top[1]:.3f}, {p_top[2]:.3f}], "
            f"rpy_W = [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
        )

    if mode in ("base", "both", "all", ""):
        print("\nBases:")
        for line in base_lines:
            print(line)
    if mode in ("top", "both", "all", ""):
        print("\nTops:")
        for line in top_lines:
            print(line)


def print_pose_differences(
    system_handles,
    detections: List[MoleDetection],
    board_camera_pose: Optional[RigidTransform] = None,
    top_offset: np.ndarray | None = None,
) -> None:
    """
    Compare camera-estimated poses vs plant ground truth for board and moles.
    """
    if top_offset is None:
        top_offset = np.array([0.0, 0.0, 0.10])

    plant_board = None
    try:
        plant_board = get_board_pose_from_plant(system_handles)
    except Exception as e:
        print(f"[diff] plant board unavailable: {e}")

    if plant_board is not None and board_camera_pose is not None:
        dp = board_camera_pose.translation() - plant_board.translation()
        drpy = (
            board_camera_pose.rotation().ToRollPitchYaw().vector()
            - plant_board.rotation().ToRollPitchYaw().vector()
        )
        print(
            "[diff] Board (camera - plant): "
            f"dp = [{dp[0]:.3f}, {dp[1]:.3f}, {dp[2]:.3f}], "
            f"drpy = [{drpy[0]:.3f}, {drpy[1]:.3f}, {drpy[2]:.3f}]"
        )
    elif board_camera_pose is not None:
        print("[diff] Board camera pose available, plant board missing.")
    elif plant_board is not None:
        print("[diff] Plant board available, camera board missing.")
    else:
        print("[diff] No board poses available for comparison.")

    plant_moles = get_mole_world_poses(system_handles)
    if not detections:
        print("[diff] No camera mole detections to compare.")
        return

    print("[diff] Mole pose differences (camera - plant):")
    for det in detections:
        i, j = divmod(det.mole_index, 3)
        plant_pose = plant_moles.get((i, j))
        if plant_pose is None:
            print(f"  mole_{i}_{j}: plant pose missing")
            continue
        cam_pose = det.X_W_mole
        dp = cam_pose.translation() - plant_pose.translation()
        drpy = (
            cam_pose.rotation().ToRollPitchYaw().vector()
            - plant_pose.rotation().ToRollPitchYaw().vector()
        )
        cam_top = cam_pose.translation() + cam_pose.rotation().matrix().dot(top_offset)
        plant_top = plant_pose.translation() + plant_pose.rotation().matrix().dot(top_offset)
        dp_top = cam_top - plant_top
        print(
            f"  mole_{i}_{j}: base dp = [{dp[0]:.3f}, {dp[1]:.3f}, {dp[2]:.3f}], "
            f"drpy = [{drpy[0]:.3f}, {drpy[1]:.3f}, {drpy[2]:.3f}], "
            f"top dp = [{dp_top[0]:.3f}, {dp_top[1]:.3f}, {dp_top[2]:.3f}]"
        )


def dump_camera_snapshots(
    system_handles,
    t_stamp: float,
    snapshot_dir: Path | str = Path("camera_snaps"),
) -> None:
    """
    Save RGB (PPM) and depth (NPY) images for each camera at the current context.
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    root_context = system_handles["root_context"]
    camera_ports = system_handles["camera_ports"]

    tag = f"{t_stamp:.3f}".replace(".", "p")
    for cam in camera_ports:
        sys_ctx = cam.rgb_port.get_system().GetMyContextFromRoot(root_context)
        rgb_img = cam.rgb_port.Eval(sys_ctx)
        rgb = (
            np.frombuffer(rgb_img.data.tobytes(), dtype=np.uint8)
            .reshape(rgb_img.height(), rgb_img.width(), 4)
        )
        depth_img = cam.depth_port.Eval(sys_ctx)
        depth = (
            np.array(depth_img.data, copy=False)
            .reshape(depth_img.height(), depth_img.width())
        )

        rgb_path = snapshot_dir / f"{cam.name}_t{tag}_rgb.ppm"
        with open(rgb_path, "wb") as f:
            f.write(f"P6\n{rgb_img.width()} {rgb_img.height()}\n255\n".encode("ascii"))
            f.write(rgb[:, :, :3].tobytes())

        depth_path = snapshot_dir / f"{cam.name}_t{tag}_depth.npy"
        np.save(depth_path, depth)

        print(f"[snapshot] saved {rgb_path} and {depth_path}")


def run_single_snapshot(
    build_system_fn,
    t_capture: float = 1.0,
    pose_source: str = "camera",
    pose_report_mode: str = "both",
    top_offset: np.ndarray | None = None,
    save_snapshots: bool = False,
    snapshot_dir: Path | str = Path("camera_snaps"),
    start_meshcat: bool = True,
    report_differences: bool = False,
):
    """
    Build a perception system via build_system_fn(meshcat) and run one snapshot.
    """
    if top_offset is None:
        top_offset = np.array([0.0, 0.0, 0.10])

    meshcat = None
    if start_meshcat:
        try:
            meshcat = StartMeshcat()
            print("Meshcat started; check the link printed above in your terminal.")
        except RuntimeError:
            print("Meshcat not started (port unavailable); continuing headless.")

    print("Building perception system...")
    system_handles = build_system_fn(meshcat=meshcat)
    print("Perception system built.")

    camera_configs = system_handles["camera_configs"]
    camera_intrinsics = system_handles["camera_intrinsics"]

    print("\n=== Perception sanity check ===")
    print(f"Capture time t = {t_capture:.3f} s")
    print(f"Number of cameras: {len(camera_configs)}")
    for cfg in camera_configs:
        print(f"  - {cfg.name}: p_W = {cfg.position_W}, rpy_W = {cfg.rpy_W}")

    print("Running perception...")
    detections = perceive_up_moles(
        system_handles=system_handles,
        camera_intrinsics=camera_intrinsics,
        t_capture=t_capture,
    )
    print("Perception finished.")

    if not detections:
        print("\nNo 'up' moles detected.")
    else:
        print(f"\nDetected {len(detections)} 'up' moles:")
        for det in detections:
            p_W = det.X_W_mole.translation()
            print(
                "  - mole_index = {idx:2d} | "
                "mean_height_B = {h:.3f} m | "
                "num_points = {n:4d} | "
                "p_Wmole = [{x:.3f}, {y:.3f}, {z:.3f}]".format(
                    idx=det.mole_index,
                    h=det.mean_height_B,
                    n=det.num_points,
                    x=float(p_W[0]),
                    y=float(p_W[1]),
                    z=float(p_W[2]),
                )
            )

    if pose_source.lower() == "plant":
        mole_poses = get_mole_world_poses(system_handles)
        print("\nWorld poses of moles (plant state):")
        print_mole_poses_from_plant(
            mole_poses, mode=pose_report_mode, top_offset=top_offset
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

    if save_snapshots:
        dump_camera_snapshots(system_handles, t_stamp=t_capture, snapshot_dir=snapshot_dir)

    if report_differences:
        board_cam = None
        try:
            board_cam = estimate_board_pose_from_cameras(
                system_handles, camera_intrinsics, t_capture
            )
        except Exception:
            board_cam = None
        print_pose_differences(
            system_handles=system_handles,
            detections=detections,
            board_camera_pose=board_cam,
            top_offset=top_offset,
        )

    return detections


def run_multi_snapshots(
    build_system_fn,
    t_end: float = 20.0,
    step: float = 1.0,
    pose_source: str = "camera",
    pose_report_mode: str = "both",
    top_offset: np.ndarray | None = None,
    save_snapshots: bool = False,
    snapshot_dir: Path | str = Path("camera_snaps"),
    start_meshcat: bool = True,
    report_differences: bool = False,
) -> None:
    """
    Advance the simulator in steps, run perception at each step, and optionally
    save snapshots.
    """
    if top_offset is None:
        top_offset = np.array([0.0, 0.0, 0.10])

    meshcat = None
    if start_meshcat:
        try:
            meshcat = StartMeshcat()
            print("Meshcat started; check the link printed above in your terminal.")
        except RuntimeError:
            print("Meshcat not started (port unavailable); continuing headless.")

    print("Building perception system...")
    system_handles = build_system_fn(meshcat=meshcat)
    print("Perception system built.")

    camera_configs = system_handles["camera_configs"]
    camera_intrinsics = system_handles["camera_intrinsics"]

    print("\n=== Perception sanity check (multi-snapshot) ===")
    print(f"Capture range t = {step:.3f} ... {t_end:.3f} s (step {step:.3f})")
    print(f"Number of cameras: {len(camera_configs)}")
    for cfg in camera_configs:
        print(f"  - {cfg.name}: p_W = {cfg.position_W}, rpy_W = {cfg.rpy_W}")

    t = step
    while t <= t_end + 1e-9:
        print(f"\n--- t = {t:.3f} s ---")

        measurements, P_W_all = capture_multi_camera_pointcloud(
            system_handles=system_handles,
            camera_intrinsics=camera_intrinsics,
            t_capture=t,
        )

        P_filtered = P_W_all
        if P_W_all.shape[0] >= 1:
            z = P_W_all[:, 2]
            mask_height = (z > 0.0) & (z < 1.5)
            P_filtered = P_W_all[mask_height]
            if P_filtered.shape[0] < 3:
                print("[perception] Point cloud empty after height filtering.")

        X_WB_est = None
        if pose_source.lower() == "plant":
            try:
                X_WB_truth = get_board_pose_from_plant(system_handles)
                p = X_WB_truth.translation()
                rpy = X_WB_truth.rotation().ToRollPitchYaw().vector()
                print(
                    f"Board pose (plant): p_W = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], "
                    f"rpy_W = [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
                )
                X_WB_est = X_WB_truth
            except Exception as e:
                print(f"Board pose (plant) unavailable: {e}")
        else:
            if P_filtered.shape[0] >= 3:
                try:
                    X_WB_est, _, _, _ = estimate_board_frame_from_cloud(P_filtered)
                except RuntimeError as e:
                    print(f"[perception] Board pose estimation failed: {e}")
            if X_WB_est is not None:
                p = X_WB_est.translation()
                rpy = X_WB_est.rotation().ToRollPitchYaw().vector()
                print(
                    f"Board pose (camera): p_W = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], "
                    f"rpy_W = [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
                )
            else:
                print("Board pose (camera) unavailable")

        detections: list[MoleDetection] = []
        board_for_detection = X_WB_est
        if pose_source.lower() == "plant" and X_WB_est is None:
            try:
                board_for_detection = get_board_pose_from_plant(system_handles)
            except Exception:
                board_for_detection = None

        if board_for_detection is not None and P_filtered.shape[0] >= 3:
            detections = detect_up_moles_from_cloud(
                P_W_all=P_filtered,
                X_WB=board_for_detection,
                mole_centers_B=MOLE_CENTERS_B,
            )
        if not detections:
            print("No 'up' moles detected.")
        else:
            print(f"Detected {len(detections)} 'up' moles:")
            for det in detections:
                p_W = det.X_W_mole.translation()
                print(
                    "  - mole_index = {idx:2d} | "
                    "mean_height_B = {h:.3f} m | "
                    "num_points = {n:4d} | "
                    "p_Wmole = [{x:.3f}, {y:.3f}, {z:.3f}]".format(
                        idx=det.mole_index,
                        h=det.mean_height_B,
                        n=det.num_points,
                        x=float(p_W[0]),
                        y=float(p_W[1]),
                        z=float(p_W[2]),
                    )
                )
        if pose_source.lower() == "plant":
            print("World poses of moles (plant state):")
            mole_poses = get_mole_world_poses(system_handles)
            print_mole_poses_from_plant(
                mole_poses, mode=pose_report_mode, top_offset=top_offset
            )
        else:
            print("World poses of moles (from camera detections):")
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
        if save_snapshots:
            dump_camera_snapshots(system_handles, t_stamp=t, snapshot_dir=snapshot_dir)

        if report_differences:
            print_pose_differences(
                system_handles=system_handles,
                detections=detections,
                board_camera_pose=X_WB_est,
                top_offset=top_offset,
            )
        t += step

def perceive_up_moles(system_handles, camera_intrinsics, t_capture) -> list[MoleDetection]:
    measurements, P_W_all = capture_multi_camera_pointcloud(
        system_handles, camera_intrinsics, t_capture
    )
    if P_W_all.shape[0] < 3:
        return []  # no points, nothing to detect

    # Debug stats on cloud height
    z_vals = P_W_all[:, 2]
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
    z_med = float(np.median(z_vals))
    print(f"[perception] Point cloud size {P_W_all.shape[0]}, z-range [{z_min:.3f}, {z_max:.3f}], z-median {z_med:.3f}")

    # Keep only points in a plausible band around the board height to help RANSAC.
    z = P_W_all[:, 2]
    mask_height = (z > 0.0) & (z < 1.5)
    P_W_all = P_W_all[mask_height]
    if P_W_all.shape[0] < 3:
        print("[perception] Point cloud empty after height filtering.")
        return []

    try:
        X_WB, _, _, _ = estimate_board_frame_from_cloud(P_W_all)
    except RuntimeError as e:
        print(f"[perception] Board frame estimation failed: {e}")
        return []
    detections = detect_up_moles_from_cloud(P_W_all, X_WB, MOLE_CENTERS_B)
    return detections


def estimate_board_pose_from_cameras(
    system_handles,
    camera_intrinsics,
    t_capture,
) -> Optional[RigidTransform]:
    """
    Helper to estimate the board pose X_WB from camera point clouds.
    Returns None if estimation fails.
    """
    measurements, P_W_all = capture_multi_camera_pointcloud(
        system_handles, camera_intrinsics, t_capture
    )
    if P_W_all.shape[0] < 3:
        print("[perception] Not enough points to estimate board pose.")
        return None

    z_vals = P_W_all[:, 2]
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
    z_med = float(np.median(z_vals))
    print(f"[perception] Board pose estimation cloud size {P_W_all.shape[0]}, z-range [{z_min:.3f}, {z_max:.3f}], z-median {z_med:.3f}")

    z = P_W_all[:, 2]
    mask_height = (z > 0.0) & (z < 1.5)
    P_W_all = P_W_all[mask_height]
    if P_W_all.shape[0] < 3:
        print("[perception] Board pose estimation: empty after height filter.")
        return None

    try:
        X_WB, _, _, _ = estimate_board_frame_from_cloud(P_W_all)
        return X_WB
    except RuntimeError as e:
        print(f"[perception] Board pose estimation failed: {e}")
        return None


def get_board_pose_from_plant(system_handles) -> RigidTransform:
    """
    Returns X_WB for the grid board using the plant state (truth).
    """
    plant = system_handles["plant"]
    root_context = system_handles["root_context"]
    plant_context = plant.GetMyContextFromRoot(root_context)

    instance = plant.GetModelInstanceByName("grid_board")
    body = plant.GetBodyByName("board", instance)
    return plant.EvalBodyPoseInWorld(plant_context, body)


def get_mole_world_poses(system_handles) -> Dict[Tuple[int, int], RigidTransform]:
    """
    Convenience to extract world poses of each mole body from the plant context.
    """
    plant = system_handles["plant"]
    root_context = system_handles["root_context"]
    plant_context = plant.GetMyContextFromRoot(root_context)

    poses: Dict[Tuple[int, int], RigidTransform] = {}
    for i in range(3):
        for j in range(3):
            inst = plant.GetModelInstanceByName(f"mole_{i}_{j}")
            body = plant.GetBodyByName("mole", inst)
            poses[(i, j)] = plant.EvalBodyPoseInWorld(plant_context, body)
    return poses
