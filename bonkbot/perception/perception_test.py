# bonkbot/perception/perception_test.py

"""
Sanity-check bonkbot.perception.whack_perception against the simple scene
defined in bonkbot.sim.simple_scene_builder_notebook.

Usage (from repo root or from bonkbot/perception):

    # From repo root:
    #   /workspaces/whackamole
    python bonkbot/perception/perception_test.py

    # OR explicitly:
    python -m bonkbot.perception.perception_test  (from repo root)

This will:
  1. Parse the `scenario_string` from simple_scene_builder_notebook.py,
  2. Build a station + cameras via initialize_whack_perception_system,
  3. Advance the simulator to t_capture,
  4. Call perceive_up_moles(...) and print any detected "up" moles.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

import yaml  # make sure `pyyaml` is in your venv: `pip install pyyaml`
from pydrake.common.yaml import yaml_load

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `import bonkbot` works
# regardless of where you run this file from.
# ---------------------------------------------------------------------------

# This file lives at: <repo_root>/bonkbot/perception/perception_test.py
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # perception -> bonkbot -> <repo_root>

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Now we can safely import bonkbot.*
from bonkbot.perception.whack_perception import (
    initialize_whack_perception_system,
    run_single_snapshot as wp_run_single_snapshot,
    run_multi_snapshots as wp_run_multi_snapshots,
    MoleDetection,
)
from bonkbot.sim import simple_scene_builder_notebook as scene_mod

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_base_scenario_dict() -> Dict[str, Any]:
    """
    Load the YAML scenario string from simple_scene_builder_notebook and
    convert it into a dict suitable for manipulation.scenarios.LoadScenario.

    We only use `scenario_string` and ignore the rest of the notebook-side
    simulation boilerplate.
    """
    scenario_string = scene_mod.scenario_string
    # Use Drake's YAML loader so tags like !IiwaDriver / !Rpy are recognized.
    scenario_dict = yaml_load(data=scenario_string)

    if not isinstance(scenario_dict, dict):
        raise RuntimeError(
            "Expected simple_scene_builder_notebook.scenario_string to "
            "parse into a dict, but got a different type."
        )
    return scenario_dict


def build_perception_system(meshcat=None) -> Dict[str, Any]:
    """
    Build the whack-perception station/diagram for the simple scene.

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
    system_handles = initialize_whack_perception_system(
        base_scenario_dict=base_scenario,
        meshcat=meshcat,
        camera_method=CAMERA_METHOD,
    )
    return system_handles


def run_single_snapshot(t_capture: float = 1.0) -> List[MoleDetection]:
    """
    Thin wrapper around whack_perception.run_single_snapshot that reuses the
    same build_perception_system defined in this file.
    """
    return wp_run_single_snapshot(
        build_system_fn=build_perception_system,
        t_capture=t_capture,
        pose_source=POSE_SOURCE,
        pose_report_mode=POSE_REPORT_MODE,
        top_offset=MOLE_TOP_OFFSET_M,
        save_snapshots=SAVE_CAMERA_SNAPSHOTS,
        snapshot_dir=SNAPSHOT_DIR,
        start_meshcat=True,
        report_differences=REPORT_DIFFERENCES,
    )


def run_multi_snapshots(t_end: float = 20.0, step: float = 1.0) -> None:
    """
    Thin wrapper around whack_perception.run_multi_snapshots that reuses the
    same build_perception_system defined in this file.
    """
    wp_run_multi_snapshots(
        build_system_fn=build_perception_system,
        t_end=t_end,
        step=step,
        pose_source=POSE_SOURCE,
        pose_report_mode=POSE_REPORT_MODE,
        top_offset=MOLE_TOP_OFFSET_M,
        save_snapshots=SAVE_CAMERA_SNAPSHOTS,
        snapshot_dir=SNAPSHOT_DIR,
        start_meshcat=True,
        report_differences=REPORT_DIFFERENCES,
    )

def main() -> None:
    """
    Entry point for a quick manual test.
    """
    # Run for 20 seconds, reporting every second.
    run_multi_snapshots(t_end=20.0, step=1.0)


if __name__ == "__main__":
    main()
