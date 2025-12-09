import numpy as np
from pydrake.all import (
    LeafSystem, BasicVector, AbstractValue, RigidTransform, EventStatus,
    JacobianWrtVariable
)
from bonkbot.hit_force_control import HitAdmittanceController
from bonkbot.prehit_planner import IiwaMotionPlanner

class FSMState:
    WAIT = 0
    PLANNING = 1
    APPROACH = 2
    HIT = 3
    RECOVER = 4
    GO_HOME = 5


class BonkBotBrain(LeafSystem):
    def __init__(self, plant, iiwa_model_instance, mole_instances, debug=False):
        LeafSystem.__init__(self)
        self.debug = debug
        self.plant = plant
        self.iiwa = iiwa_model_instance
        
        # Configuration
        self.q_home = np.array([-1.57, 0.1, 0, -1.2, 0, 1.6, 0]) # home iiwa pose
        self.mole_threshold = 0.08
        self.F_des = 15.0 # Desired Force
        self.dt = 0.01 # 100 Hz Control Loop

        # Initialize Helper Classes
        self.motion_planner = IiwaMotionPlanner(plant, iiwa_model_instance)
        self.admittance = HitAdmittanceController(
            M=1.0, D=40.0, K=0.0, dt=self.dt, n_hat=[0, 0, -1.0]
        )
        
        # Inputs 
        self._iiwa_pos_index = self.DeclareVectorInputPort("iiwa_position", BasicVector(7)).get_index()
        self._iiwa_vel_index = self.DeclareVectorInputPort("iiwa_velocity", BasicVector(7)).get_index()
        self._perception_index = self.DeclareAbstractInputPort(   # (Dictionary of mole id: RigidTransform)
                                    "mole_poses", AbstractValue.Make({})).get_index()
        self._force_index = self.DeclareVectorInputPort("F_meas", BasicVector(1)).get_index()
        
        # Output: iiwa position command
        self.DeclareVectorOutputPort("iiwa_position_command", BasicVector(7), self.CalcIiwaCommand)

        # States
        self._fsm_state_index = self.DeclareAbstractState(AbstractValue.Make(FSMState.WAIT))
        self._traj_index = self.DeclareAbstractState(AbstractValue.Make(None))
        # Logic [target_mole_idx, action_start_time, tick_count]
        self._logic_state_index = self.DeclareDiscreteState(3) 
        # Admittance [s, s_dot, q_anchor(7)] = [displacement along n_hat, velocity along n_hat, iiwa joint positions]
        self._admittance_state_index = self.DeclareDiscreteState(2 + 7)

        # Schedule
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self.dt,
            offset_sec=0.0,
            update=self.UpdateFSM
        )
    
    # --- HELPER FUNCTIONS ---
    
    def get_iiwa_state(self, context):
        """Helper to combine pos and vel into a 14-vector for calculations"""
        q = self.EvalVectorInput(context, self._iiwa_pos_index).get_value()
        v = self.EvalVectorInput(context, self._iiwa_vel_index).get_value()
        return q, v

    def get_active_mole(self, mole_poses):
        """
        Helper that returns index of the first mole that is up (above threshold).
        "First" here just means first mole index, not accounting for time for now.
        Returns -1 if none found.
        """
        heights = {i: p.translation()[2] for i, p in mole_poses.items()}
        for idx, pose in mole_poses.items():
            # if z position of mole greater than threshold
            if pose.translation()[2] > self.mole_threshold:
                return idx
        return -1
    
    def check_target_mole_valid(self, logic_state_val, mole_poses):
        """
        Helper to check whether current target mole is still up.
        Returns tuple (is_valid, target_idx, target_pose)
        """
        # Get current target from logic state
        target_idx = int(logic_state_val[0]) # mole index
        target_pose = mole_poses[target_idx]
        # Check mole existence and height
        target_is_valid = (target_pose is not None and 
                            target_pose.translation()[2] > self.mole_threshold)
        return target_is_valid, target_idx, target_pose

    def start_trajectory_action(self, new_fsm, new_traj, new_logic, 
                                next_state, trajectory, current_time):
        """
        Helper to consolidate updating state, trajectory, and start time.
        """
        new_traj.set_value(trajectory)
        new_logic.get_mutable_value()[1] = current_time
        new_fsm.set_value(next_state)

    def try_attack(self, active_mole, new_fsm, new_logic):
        """
        Helper to check for a mole target and transition to attack, esp while in motion. 
            (BonkBot follows ABC: Always Be Checking for up moles to attack.)
        Returns False if no valid targets.
        """
        if active_mole >= 0:
            if self.debug:
                print(f"[Brain] DETECTED MOLE {active_mole}! ATTACK! Switch to PLANNING.")
            new_logic.get_mutable_value()[0] = active_mole 
            new_fsm.set_value(FSMState.PLANNING)
            return True
        return False
        
    # --- MAIN LOGIC ---

    def CalcIiwaCommand(self, context, output):
        """Continuous control loop that evaluates trajectory."""
        current_time = context.get_time()
        fsm_state = context.get_abstract_state(self._fsm_state_index).get_value()
        logic_state = context.get_discrete_state(self._logic_state_index).get_value()
        q_current, _ = self.get_iiwa_state(context)
        
        # Default to holding current position (WAIT, PLANNING)
        q_cmd = q_current 
        
        # If hitting
        if fsm_state == FSMState.HIT:
            adm_state = context.get_discrete_state(self._admittance_state_index).get_value()
            s = adm_state[0]
            q_anchor = adm_state[2:]
            
            # Calculate Jacobian live
            context_plant = self.motion_planner.context
            self.plant.SetPositions(context_plant, self.iiwa, q_current)
            J_spatial = self.plant.CalcJacobianTranslationalVelocity(
                context_plant, JacobianWrtVariable.kQDot,
                self.motion_planner.hammer_face_frame, np.zeros(3),
                self.plant.world_frame(), self.plant.world_frame()
            )
            # Slice for IIWA (first 7 columns)
            J = J_spatial[:, 0:7] 

            q_cmd = self.admittance.compute_q_cmd(q_anchor, s, J)
        
        # If in other moving state, execute trajectory
        elif fsm_state in {FSMState.APPROACH, FSMState.RECOVER, FSMState.GO_HOME}:
            traj = context.get_abstract_state(self._traj_index).get_value()
            if traj:
                start_time = logic_state[1]
                t_rel = current_time - start_time
                q_cmd = traj.value(min(t_rel, traj.end_time())).flatten()
                
        output.SetFromVector(q_cmd)


    def UpdateFSM(self, context, discrete_state):
        """
        Main BonkBot Brain Logic
        """
        current_time = context.get_time()
        
        # Get Mutable States
        new_fsm_state = discrete_state.get_mutable_abstract_state(self._fsm_state_index)
        new_traj = discrete_state.get_mutable_abstract_state(self._traj_index)
        new_logic = discrete_state.get_mutable_discrete_state(self._logic_state_index)
        new_adm = discrete_state.get_mutable_discrete_state(self._admittance_state_index)
        
        # Current Values (read-only)
        current_fsm = context.get_abstract_state(self._fsm_state_index).get_value()
        mole_poses = self.EvalAbstractInput(context, self._perception_index).get_value()
        logic_val = context.get_discrete_state(self._logic_state_index).get_value() # [target_mole_idx, action_start_time]
        tick_count = int(logic_val[2])
        q_current, v_current = self.get_iiwa_state(context)
        F_meas_scalar = self.EvalVectorInput(context, self._force_index).get_value()[0]

        if self.debug and current_time % 1 == 0:
            print(f"[DEBUG t={current_time:.2f}] State: {current_fsm}")

        # --- FSM LOGIC ---

        # FAST LOOP (100Hz): FORCE CONTROL
        if current_fsm == FSMState.HIT:
            s = new_adm.get_value()[0]
            s_dot = new_adm.get_value()[1]
            
            # Calculate next s, s_dot
            s_new, s_dot_new = self.admittance.compute_next_state(
                s, s_dot, self.F_des, F_meas_scalar
            )
            new_adm.SetAtIndex(0, s_new)
            new_adm.SetAtIndex(1, s_dot_new)
            
            # Hit Timeout Check
            start_time = logic_val[1]
            if current_time - start_time > 0.5:
                if self.debug:
                    print("[Brain] Hit timeout/complete. Switch to RECOVER.")
                # Recover trajectory is just reverse out from where we currently are
                q_anchor = new_adm.get_value()[2:]
                recover_traj = self.motion_planner.make_joint_space_position_trajectory(
                    [q_current, q_anchor], duration=0.5
                )
                self.start_trajectory_action(new_fsm_state, new_traj, new_logic,
                                             FSMState.RECOVER, recover_traj, current_time)
                
        # SLOW LOOP (50Hz): OTHER STATES
        else:
            # Enforce slower rates for non-hit states
            new_logic.SetAtIndex(2, tick_count + 1)
            if tick_count % 2 != 0:
                return EventStatus.Succeeded()

            # Check Perception
            active_mole = self.get_active_mole(mole_poses) # most recent active mole
            
            if current_fsm == FSMState.WAIT:
                # MOLE CHECK: yo mole, you up?
                if not self.try_attack(active_mole, new_fsm_state, new_logic):
                    if self.debug and current_time % 1 == 0:
                        print("[Brain] No mole is up.") # ;-;

            elif current_fsm == FSMState.PLANNING:            
                # MOLE CHECK: you still up?
                valid, target_idx, target_pose = self.check_target_mole_valid(logic_val, mole_poses)
                if not valid:
                    # If current mole gone, try switch to new mole.
                    if active_mole >= 0:
                        if self.debug:
                            print(f"[Brain] Mole {target_idx} gone before planning. Retarget Mole {active_mole}.")
                        new_logic.get_mutable_value()[0] = active_mole
                    else:
                        if self.debug:
                            print(f"[Brain] Mole {target_idx} gone before planning. Switch to WAIT.")
                        new_fsm_state.set_value(FSMState.WAIT)
                        return EventStatus.Succeeded()
                
                # Calculate path from current pose to target prehit pose
                if self.debug:
                    print(f"[Brain] Planning pre-hit path to mole {target_idx} at {target_pose.translation()}")
                traj = self.motion_planner.plan_prehit(q_current, target_pose)
                
                if traj is not None:
                    if self.debug:
                        print("[Brain] Prehit path found. Switch to APPROACH.")
                    self.start_trajectory_action(new_fsm_state, new_traj, new_logic,
                                                FSMState.APPROACH, traj, current_time)
                else:
                    if self.debug:
                        print("[Brain] Prehit plan failed (IK). Return to WAIT.")
                    new_fsm_state.set_value(FSMState.WAIT)

            elif current_fsm == FSMState.APPROACH:
                # MOLE CHECK: you still up?
                valid, target_idx, _ = self.check_target_mole_valid(logic_val, mole_poses)
                if not valid:
                    # If current mole gone, try switch to new mole.
                    if active_mole >= 0:
                        if self.debug:
                            print(f"[Brain] Mole {target_idx} gone while approaching. Retarget Mole {active_mole}.")
                        new_logic.get_mutable_value()[0] = active_mole
                        new_fsm_state.set_value(FSMState.PLANNING)
                    else:
                        if self.debug:
                            print(f"[Brain] Mole {target_idx} gone while approaching. Switch to WAIT.")
                        new_fsm_state.set_value(FSMState.WAIT)
                    return EventStatus.Succeeded()
                
                traj = context.get_abstract_state(self._traj_index).get_value()
                duration = current_time - logic_val[1] # action start time
                if self.debug and current_time % 1 == 0:
                    print(f"[Brain] Approaching... ({duration:.2f}/{traj.end_time():.2f}s)")

                # Check if prehit complete.
                if duration >= traj.end_time():
                    if self.debug:
                        print("[Brain] Pre-hit reached. Switching to ADMITTANCE HIT.")
                    # Set Admittance state
                    # Initial Admittance: s=0, v=0.2 (impact velocity)
                    new_adm.SetAtIndex(0, 0.0)
                    new_adm.SetAtIndex(1, 0.2) 
                    # Set Anchor Pose
                    for i in range(7):
                        new_adm.SetAtIndex(2+i, q_current[i])
                    new_fsm_state.set_value(FSMState.HIT)
                    new_logic.SetAtIndex(1, current_time)

            elif current_fsm == FSMState.RECOVER:
                traj = context.get_abstract_state(self._traj_index).get_value()
                duration = current_time - logic_val[1]

                # Check if recover complete.
                if duration >= traj.end_time():
                    # MOLE CHECK: you up? 
                    if self.try_attack(active_mole, new_fsm_state, new_logic):
                        return EventStatus.Succeeded()
                    if self.debug:
                        print("[Brain] Recovered. No targets. Going HOME.")
                    home_traj = self.motion_planner.make_joint_space_position_trajectory(
                        [q_current, self.q_home], duration=2.0
                    )
                    self.start_trajectory_action(new_fsm_state, new_traj, new_logic,
                                                FSMState.GO_HOME, home_traj, current_time)
                    
            elif current_fsm == FSMState.GO_HOME:
                # MOLE CHECK: you up?
                if self.try_attack(active_mole, new_fsm_state, new_logic):
                    return EventStatus.Succeeded()

                traj = context.get_abstract_state(self._traj_index).get_value()
                duration = current_time - logic_val[1]
                
                # Check if go home complete.
                if duration >= traj.end_time():
                    if self.debug:
                        print("[Brain] Arrived Home. Waiting.")
                    new_fsm_state.set_value(FSMState.WAIT)

        return EventStatus.Succeeded()