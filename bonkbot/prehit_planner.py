from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.all import (RigidTransform, RotationMatrix,
                         SolutionResult, Solve,
)
from pydrake.trajectories import PiecewisePolynomial

import numpy as np

class IiwaMotionPlanner:
    def __init__(self, plant, iiwa_model_instance, hammer_frame_name="hammer_face", 
                 default_traj_duration=2.0):
        self.plant = plant
        # We create a specific context for planning so we don't mess with simulation state
        self.context = plant.CreateDefaultContext()
        self.iiwa = iiwa_model_instance

        self.traj_duration = default_traj_duration

        # Store frames
        self.world_frame = plant.world_frame()
        self.hammer_model = plant.GetModelInstanceByName("hammer")
        self.hammer_face_frame = plant.GetFrameByName(hammer_frame_name, self.hammer_model)
        self.iiwa_link7_frame = plant.GetFrameByName("iiwa_link_7", self.iiwa)

        # Extract Iiwa Limits for TOPPRA        
        self.v_lower, self.v_upper, self.a_lower, self.a_upper = self.extract_limits()

    def extract_limits(self):
        """Extracts IIWA-only limits from the full plant."""
        # Find start index of IIWA
        iiwa_indices = self.plant.GetJointIndices(self.iiwa)
        start_idx = 0
        for idx in iiwa_indices:
            joint = self.plant.get_joint(idx)
            if joint.num_velocities() > 0:
                start_idx = joint.velocity_start()
                break
        num_v = self.plant.num_velocities(self.iiwa)
        v_lower = self.plant.GetVelocityLowerLimits()[start_idx : start_idx+num_v]
        v_upper = self.plant.GetVelocityUpperLimits()[start_idx : start_idx+num_v]
        a_lower = self.plant.GetAccelerationLowerLimits()[start_idx : start_idx+num_v]
        a_upper = self.plant.GetAccelerationUpperLimits()[start_idx : start_idx+num_v]
        # Safety clamp for infinite accelerations
        a_lower = np.maximum(a_lower, -10.0)
        a_upper = np.minimum(a_upper, 10.0)
        return v_lower, v_upper, a_lower, a_upper

    def get_prehit_pose(self, X_WO: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
        """
        Calculates the pre-hit pose for hammer face X_WH_prehit based on the object pose X_WO.
        Returns:
            X_WH_prehit: Pose of the hammer face in World
            X_WL7_prehit: Pose of Link 7 in World (useful for debugging/seeding)
        """
        # Fixed offset from Object to Hammer Face (Your logic)
        p_OH = np.array([0.0, 0, 0.2])
        R_OH = RotationMatrix.MakeYRotation(np.pi) @ RotationMatrix.MakeZRotation(np.pi/2)
        X_OH = RigidTransform(R_OH, p_OH) 
        
        X_WH_prehit = X_WO @ X_OH
        
        # Calculate X_HL7 (Hammer Face to Link 7)
        # We can get this from the plant's default context since it's a rigid weld
        X_WL7_default = self.plant.CalcRelativeTransform(
            self.context, self.world_frame, self.iiwa_link7_frame
        )
        X_WH_default = self.plant.CalcRelativeTransform(
            self.context, self.world_frame, self.hammer_face_frame
        )
        # X_HL7 = (X_WH)^-1 @ X_WL7
        X_HL7 = X_WH_default.inverse() @ X_WL7_default
        
        X_WL7_prehit = X_WH_prehit @ X_HL7
        
        return X_WH_prehit, X_WL7_prehit

    def plan_prehit(self, q_current, target_pose_world):
        """
        High-level function: Takes current joints and target object pose,
        returns a trajectory to the pre-hit pose.
        """
        # Get desired prehit pose for hammer face X_WH_prehit given target mole pose
        X_WH_prehit, X_WL7_seed = self.get_prehit_pose(target_pose_world)
        
        # Solve IK for the target iiwa prehit config q_goal
        q_goal, success = self.solve_ik(X_WH_prehit, q_guess=q_current)
        
        if not success:
            print(f"IK failed to find a prehit config for target mole {target_pose_world}")
            return None

        # # Generate Trajectory
        # # Simple straight line in joint space
        # straight_traj = self.make_joint_space_position_trajectory([q_current, q_goal], duration=self.traj_duration)

        # # Generate Time Optimal Trajectory, new PiecewisePolynomial that traces same path but with optimized timing
        # traj = self.generate_time_optimal_trajectory(straight_traj) 

        traj = self.make_smart_cubic_trajectory(
            q_start=q_current, 
            q_goal=q_goal, 
            velocity_limits=self.v_upper 
        )
        return traj

    def make_joint_space_position_trajectory(self, path, duration=2.0):
        times = np.linspace(0, duration, len(path))
        Q = np.column_stack(path)
        return PiecewisePolynomial.FirstOrderHold(times, Q)
    
    def make_smart_cubic_trajectory(self, q_start, q_goal, velocity_limits):
        """
        Creates a cubic trajectory where the duration is calculated dynamically 
        based on how far the robot has to move and its max velocity.
        """
        
        # 1. Calculate the distance for every joint
        delta_q = np.abs(q_goal - q_start)
        
        # 2. Calculate time required for each joint to complete the move
        # time = distance / velocity
        # We add a safety factor (e.g., 0.5) to not run at 100% red-line speed.
        safety_factor = 0.5
        eff_limits = velocity_limits * safety_factor
        
        # Avoid division by zero
        time_per_joint = delta_q / np.maximum(eff_limits, 1e-6)
        
        # 3. The move takes as long as the slowest joint needs
        duration = np.max(time_per_joint)
        
        # 4. Enforce a minimum time (e.g. 0.1s) to avoid numerical issues on tiny moves
        duration = max(duration, 0.1)
        
        # 5. Build the spline
        times = np.array([0.0, duration])
        samples = np.column_stack((q_start, q_goal))
        v_zero = np.zeros(len(q_start))
        
        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            times, samples, v_zero, v_zero
        )

    
    def solve_ik(self, X_WH, q_guess=None, pos_tol=1e-3, theta_bound=1e-2):
        ik = InverseKinematics(self.plant, self.context)
        prog = ik.prog()
        # Positions of all objects in plant
        q_current_full = self.plant.GetPositions(self.context)
        q_vars = ik.q()
        # if q_guess is None:
        #     q_guess = self.plant.GetPositions(self.context, self.iiwa)

        R_WH_desired = X_WH.rotation()
        p_WH_desired = X_WH.translation()

        # Position Constraint
        ik.AddPositionConstraint(
            frameA=self.world_frame,
            p_BQ=np.zeros(3),
            frameB=self.hammer_face_frame,
            p_AQ_lower=p_WH_desired - pos_tol,
            p_AQ_upper=p_WH_desired + pos_tol,
        )
        # Strict Constraint (hammer frame must align exactly with target prehit pose)
        # Orientation Constraint 
        # ik.AddOrientationConstraint(
        #     frameAbar=self.world_frame,
        #     R_AbarA=RotationMatrix(), 
        #     frameBbar=self.hammer_face_frame,
        #     R_BbarB=R_WH_desired,
        #     theta_bound=theta_bound,
        # )

        # Loose Contraint: Align Hammer Z-axis with Target Pose Z-axis
        # Extract the Z-axis column from the desired rotation matrix
        z_axis_desired = R_WH_desired.matrix()[:, 2] 
        ik.AddAngleBetweenVectorsConstraint(
            frameA=self.world_frame,
            na_A=z_axis_desired,      # desired Z direction in World Frame
            frameB=self.hammer_face_frame, 
            nb_B=np.array([0, 0, 1]), # hammer's Z-axis (0,0,1) in Hammer Frame
            angle_lower=0.0,
            angle_upper=theta_bound 
        )

        # IMPORTANT: Lock all non-IIWA joints
        # If we don't do this, the solver will move the moles to satisfy constraints.
        
        # Iterate over all models to find which indices in 'q' belong to "Not IIWA"
        indices_to_lock = []
        world_instance = self.plant.GetModelInstanceByName("WorldModelInstance")
        iiwa_instance = self.iiwa

        for i in range(self.plant.num_model_instances()):
            model_instance = ModelInstanceIndex(i)
            if model_instance in (iiwa_instance, world_instance):
                continue  # skip iiwa and world
            # For each non-iiwa instance, get all joint indices
            for joint_idx in self.plant.GetJointIndices(model_instance):
                joint = self.plant.get_joint(joint_idx)
                # Get the start index in the q vector for this joint
                start = joint.position_start()
                num = joint.num_positions()
                indices_to_lock.extend(range(start, start + num))
    
        # Apply the lock constraint
        if indices_to_lock:
            indices_to_lock = np.array(indices_to_lock, dtype=int)
            prog.AddBoundingBoxConstraint(
                q_current_full[indices_to_lock], # Lower bound (current val)
                q_current_full[indices_to_lock], # Upper bound (current val)
                q_vars[indices_to_lock]          # The decision variables
            )

        # Solve 
        prog.SetInitialGuess(q_vars, q_current_full)
        result = Solve(prog)
        
        # Extract only the iiwa solution
        q_sol_full = result.GetSolution(q_vars)
        self.plant.SetPositions(self.context, q_sol_full)
        q_sol_iiwa = self.plant.GetPositions(self.context, self.iiwa)
        
        # Return solution and success
        if result.get_solution_result() != SolutionResult.kSolutionFound:
            return q_sol_iiwa, False
        return q_sol_iiwa, True