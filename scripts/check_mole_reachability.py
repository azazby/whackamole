#!/usr/bin/env python
"""
Check whether the current pre-hit pose (from
bonkbot.sim.simple_scene_builder_prehit_force_notebook.get_mole_prehit_pose)
is reachable via the IK helper for all 9 moles in the simple scene.

Runs headless (no Meshcat). Prints reachability per mole and a summary.
Usage (from repo root):
    PYTHONPATH=. .venv/bin/python scripts/check_mole_reachability.py
"""

from pathlib import Path
from types import SimpleNamespace
from manipulation.station import LoadScenario, MakeHardwareStation


def load_scene_without_meshcat():
    """
    Load the notebook module with a stubbed StartMeshcat to avoid port issues.
    Returns a namespace with scenario_string, get_mole_prehit_pose, solve_ik.
    """
    mod_path = Path(__file__).resolve().parents[1] / "bonkbot" / "sim" / "simple_scene_builder_prehit_force_notebook.py"
    code = mod_path.read_text()
    # Prevent Meshcat from starting by setting to None and removing triad block
    code = code.replace("meshcat = StartMeshcat()", "meshcat = None")
    code = code.replace("print(\"Click the link above to open Meshcat in your browser!\")", "pass")
    triad_start = code.find("# visualize extra prehit frames")
    triad_end = code.find("# Get initial positions of the iiwa joints")
    if triad_start != -1 and triad_end != -1 and triad_end > triad_start:
        code = code[:triad_start] + code[triad_end:]
    sim_start = code.find("# Run simulation")
    if sim_start != -1:
        code = code[:sim_start]
    ns = {
        "__name__": "__reach_check__",
        "__file__": str(mod_path),
    }
    exec(compile(code, str(mod_path), "exec"), ns)
    return SimpleNamespace(**ns)


def main():
    scene = load_scene_without_meshcat()

    # Build station/plant without Meshcat
    scenario = LoadScenario(data=scene.scenario_string)
    # Disable meshcat creation in visualization config to avoid port usage.
    if hasattr(scenario, "config") and hasattr(scenario.config, "enable_meshcat_creation"):
        scenario.config.enable_meshcat_creation = False
    station = MakeHardwareStation(scenario, meshcat=None)
    plant = station.GetSubsystemByName("plant")

    context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Frames
    hammer = plant.GetModelInstanceByName("hammer")
    hammer_face_frame = plant.GetFrameByName("hammer_face", hammer)

    iiwa = plant.GetModelInstanceByName("iiwa")
    l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
    X_HL7 = plant.CalcRelativeTransform(plant_context, hammer_face_frame, l7_frame)

    all_ok = []
    for i in range(3):
        for j in range(3):
            inst = plant.GetModelInstanceByName(f"mole_{i}_{j}")
            body = plant.GetBodyByName("mole", inst)
            X_WM = plant.EvalBodyPoseInWorld(plant_context, body)
            X_WH, _ = scene.get_mole_prehit_pose(X_WM, X_HL7)
            q0 = plant.GetPositions(plant_context, iiwa)
            _, ok = scene.solve_ik(
                plant,
                plant_context,
                X_WH,
                q_guess=q0,
                pos_tol=5e-3,
                theta_bound=5e-3,
            )
            print(f"mole_{i}_{j}: {'reachable' if ok else 'IK FAILED'}")
            all_ok.append(ok)

    print(f"All reachable: {all(all_ok)}")


if __name__ == "__main__":
    main()
