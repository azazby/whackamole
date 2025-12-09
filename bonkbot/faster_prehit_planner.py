import numpy as np
from pydrake.all import (
    RigidTransform, RotationMatrix, InverseKinematics, Solve, 
    SolutionResult, PiecewisePolynomial, Trajectory,
    ModelInstanceIndex
)
# New Import for the optimizer
from pydrake.planning import TimeOptimalTrajectoryGeneration

class IiwaMotionPlanner:
    def __init__(self, plant, iiwa_model_instance, hammer_frame_name="hammer_face"):
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.iiwa = iiwa_model_instance
        
        # Cache frames
        self.world_frame = plant.world_frame()
        self.hammer_model = plant.GetModelInstanceByName("hammer")
        self.hammer_face_frame = plant.GetFrameByName(hammer_frame_name, self.hammer_model)
        self.iiwa_link7_frame = plant.GetFrameByName("iiwa_link_7", self.iiwa)

        # --- NEW: Extract Limits for TOPPRA ---
        # We need to extract the limits specifically for the iiwa joints.
        # We iterate through the plant limits and slice out the ones belonging to iiwa.
        
        num_iiwa_velocities = self.plant.num_velocities(self.iiwa)
        
        # Get full plant limits
        v_lower_full = self.plant.GetVelocityLowerLimits()
        v_upper_full = self.plant.GetVelocityUpperLimits()
        a_lower_full = self.plant.GetAccelerationLowerLimits()
        a_upper_full = self.plant.GetAccelerationUpperLimits()
        
        # Create empty arrays for iiwa-specific limits
        self.v_lower = np.zeros(num_iiwa_velocities)
        self.v_upper = np.zeros(num_iiwa_velocities)
        self.a_lower = np.zeros(num_iiwa_velocities)
        self.a_upper = np.zeros(num_iiwa_velocities)
        
        # Map global indices to iiwa indices
        # This assumes your IIWA joints are sequential in the plant (standard in Drake),
        # but to be safe, we iterate through joints.
        current_idx = 0
        for i in range(self.plant.num_joints()):
            joint = self.plant.get_joint(self.plant.get_joint_indices()[i])
            if joint.model_instance() == self.iiwa:
                # Ignore welds (0 degrees of freedom)
                if joint.num_velocities() > 0:
                    # Get the index in the global qdot vector
                    start = joint.velocity_start()
                    end = start + joint.num_velocities()
                    
                    # Copy limits to our local storage
                    self.v_lower[current_idx:current_idx+joint.num_velocities()] = v_lower_full[start:end]
                    self.v_upper[current_idx:current_idx+joint.num_velocities()] = v_upper_full[start:end]
                    self.a_lower[current_idx:current_idx+joint.num_velocities()] = a_lower_full[start:end]
                    self.a_upper[current_idx:current_idx+joint.num_velocities()] = a_upper_full[start:end]
                    
                    current_idx += joint.num_velocities()
        
        # Fallback: If URDF didn't specify acceleration limits, they might be -inf/inf.
        # We clamp them to something reasonable (e.g., 10 rad/s^2) to prevent the optimizer from 
        # producing instant motion.
        default_accel_limit = 10.0 
        self.a_lower = np.maximum(self.a_lower, -default_accel_limit)
        self.a_upper = np.minimum(self.a_upper, default_accel_limit)

    def get_prehit_pose(self, X_WO: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
        """ Standard Pre-hit pose logic (Unchanged) """
        p_OH = np.array([0.0, 0, 0.2])
        R_OH = RotationMatrix.MakeYRotation(np.pi) @ RotationMatrix.MakeZRotation(np.pi/2)
        X_OH = RigidTransform(R_OH, p_OH) 
        X_WH_prehit = X_WO @ X_OH
        
        X_WL7_default = self.plant.CalcRelativeTransform(self.context, self.world_frame, self.iiwa_link7_frame)
        X_WH_default = self.plant.CalcRelativeTransform(self.context, self.world_frame, self.hammer_face_frame)
        X_HL7 = X_WH_default.inverse() @ X_WL7_default
        X_WL7_prehit = X_WH_prehit @ X_HL7
        return X_WH_prehit, X_WL7_prehit

    def plan_prehit(self, q_current, target_pose_world):
        """
        Uses TimeOptimalTrajectoryGeneration (TOPPRA)
        """
        X_WH_prehit, X_WL7_seed = self.get_prehit_pose(target_pose_world)
        
        # 1. Solve Inverse Kinematics for the Goal
        q_goal, success = self.solve_ik(X_WH_prehit, q_guess=q_current)
        
        if not success:
            print(f"IK failed for target {target_pose_world.translation()}")
            return None

        # 2. Define the Geometry of the path (Straight line in joint space)
        # We use dummy times [0, 1] because TOPPRA will overwrite the time scaling anyway.
        path_geometry = PiecewisePolynomial.FirstOrderHold([0.0, 1.0], np.column_stack((q_current, q_goal)))
        
        # 3. Generate Time Optimal Trajectory
        # This returns a new PiecewisePolynomial that traces the same path but with optimized timing
        traj = self.generate_time_optimal_trajectory(path_geometry)
        
        return traj

    def generate_time_optimal_trajectory(self, path: PiecewisePolynomial) -> PiecewisePolynomial:
        """
        Takes a geometric path and re-parameterizes it to be as fast as possible
        within velocity and acceleration limits.
        """
        
        # Optional: Safety factor (e.g., run at 80% capacity)
        velocity_limit_factor = 1.0
        accel_limit_factor = 1.0

        # pydrake.planning.TimeOptimalTrajectoryGeneration
        toppra_traj = TimeOptimalTrajectoryGeneration(
            path=path,
            velocity_lower_limit=self.v_lower * velocity_limit_factor,
            velocity_upper_limit=self.v_upper * velocity_limit_factor,
            acceleration_lower_limit=self.a_lower * accel_limit_factor,
            acceleration_upper_limit=self.a_upper * accel_limit_factor,
            num_control_points=100 # Resolution of the optimization grid
        )
        
        return toppra_traj

    def solve_ik(self, X_WH, q_guess=None, pos_tol=1e-3, theta_bound=1e-2):
        """ (Your existing IK logic, largely unchanged) """
        ik = InverseKinematics(self.plant, self.context)
        prog = ik.prog()
        q_current_full = self.plant.GetPositions(self.context)
        q_vars = ik.q()

        R_WH_desired = X_WH.rotation()
        p_WH_desired = X_WH.translation()

        ik.AddPositionConstraint(
            frameA=self.world_frame, p_BQ=np.zeros(3),
            frameB=self.hammer_face_frame, p_AQ_lower=p_WH_desired - pos_tol, p_AQ_upper=p_WH_desired + pos_tol,
        )

        ik.AddOrientationConstraint(
            frameAbar=self.world_frame, R_AbarA=RotationMatrix(), 
            frameBbar=self.hammer_face_frame, R_BbarB=R_WH_desired, theta_bound=theta_bound,
        )
        
        # Locking non-iiwa joints logic
        indices_to_lock = []
        world_instance = self.plant.GetModelInstanceByName("WorldModelInstance")
        for i in range(self.plant.num_model_instances()):
            model_instance = ModelInstanceIndex(i)
            if model_instance in (self.iiwa, world_instance): continue
            for joint_idx in self.plant.GetJointIndices(model_instance):
                joint = self.plant.get_joint(joint_idx)
                if joint.num_positions() > 0: # Ensure joint has positions
                    start = joint.position_start()
                    indices_to_lock.extend(range(start, start + joint.num_positions()))
    
        if indices_to_lock:
            indices_to_lock = np.array(indices_to_lock, dtype=int)
            prog.AddBoundingBoxConstraint(
                q_current_full[indices_to_lock], q_current_full[indices_to_lock], q_vars[indices_to_lock]
            )

        prog.SetInitialGuess(q_vars, q_current_full)
        result = Solve(prog)
        
        q_sol_full = result.GetSolution(q_vars)
        self.plant.SetPositions(self.context, q_sol_full)
        q_sol_iiwa = self.plant.GetPositions(self.context, self.iiwa)
        
        if result.get_solution_result() != SolutionResult.kSolutionFound:
            return q_sol_iiwa, False
        return q_sol_iiwa, True