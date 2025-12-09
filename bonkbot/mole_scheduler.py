import numpy as np
import random
from pydrake.all import (
    LeafSystem, BasicVector, AbstractValue, EventStatus, ContactResults
)

class MoleController(LeafSystem):
    """
    Manages the logic for popping moles up and down.
    
    Hit Detection Criteria:
    - Active mole must be in contact with Hammer.
    - Active mole must be pushed down below 'hit_threshold' (2.5 cm).
    """
    def __init__(self, plant, hammer_body_index, mole_model_indices, 
                    switch_time_range=(1.5, 3.0), random_switch_time=False, multi_mode=False):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.context_plant = plant.CreateDefaultContext()
        self.hammer_idx = hammer_body_index
        self.mole_models = mole_model_indices 
        self.num_moles = len(mole_model_indices)
        self.min_time, self.max_time = switch_time_range
        self.random_switch_time = random_switch_time
        self.multi_mode = multi_mode
        
        # Get mole body indices for contact check
        self.mole_body_indices = []
        for model_idx in self.mole_models:
            body = plant.GetBodyByName("mole", model_idx)
            self.mole_body_indices.append(body.index())
        
        # Thresholds
        self.hit_height_threshold = 0.025 # Must be pushed down to 2.5cm
        self.spawn_grace_period = 0.5 # Ignore low height for 0.5s after spawn

        # Input Ports
        # Contact Results (to check collisions)
        self._contact_input_index = self.DeclareAbstractInputPort(
                                        "contact_results",
                                        AbstractValue.Make(ContactResults())).get_index()
        # Plant State (to check heights)
        self._state_input_index = self.DeclareVectorInputPort(
                                        "plant_state",
                                        BasicVector(plant.num_multibody_states())).get_index()
        # Output Port
        self.mole_ports = []
        for i in range(self.num_moles):
            port = self.DeclareVectorOutputPort(
                        f"mole_{i}_setpoint", 
                        BasicVector(2), 
                        lambda context, output, idx=i: self.CalcMoleCommand(context, output, idx)
                    )
            self.mole_ports.append(port)
        
        # States
        # Timer State: [next_switch_time]
        self._timer_index = self.DeclareDiscreteState(1)
        # Mole States: vector with len of num_moles, 1 = Active/Up, 0 = Inactive/Down
        self._moles_index = self.DeclareDiscreteState(self.num_moles)
        # Activation Times: track when each mole became active for spawn_grace_period
        self._activation_times_index = self.DeclareDiscreteState(self.num_moles)
        
        # Schedule
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.02, 
            offset_sec=0.0, 
            update=self.UpdateLogic
        )

    def CalcMoleCommand(self, context, output, mole_idx):
        mole_states = context.get_discrete_state(self._moles_index).get_value()
        is_active = (mole_states[mole_idx] > 0.5) # check if mole up
        if is_active:
            q_des = 0.12 # Pop up
        else:
            q_des = 0.0  # Stay down
        v_des = 0.0
        output.SetFromVector([q_des, v_des])

    def handle_hits(self, moles_vec, activation_vec, timer_vec, current_time):
        """
        Checks height only (with time buffer) to register hits.
        """
        for i in range(self.num_moles):
            # Only check active moles
            if moles_vec.get_value()[i] > 0.5:
                # Check Grace Period
                # (Mole starts at 0.0 height, so we must wait for it to rise 
                # before we start checking if it's been pushed down)
                spawn_time = activation_vec.get_value()[i]
                if (current_time - spawn_time) < self.spawn_grace_period:
                    continue
                # Check Height: is mole pushed down?
                model_idx = self.mole_models[i]
                q_mole = self.plant.GetPositions(self.context_plant, model_idx)[0]
                is_pushed_down = q_mole < self.hit_height_threshold
                if is_pushed_down:
                    print(f"[BONK] Mole {i} bonked! (Height: {q_mole:.3f}m)")
                    # Deactivate Mole
                    moles_vec.SetAtIndex(i, 0.0) 
                    # Reset Timer (pause briefly to prevent mole from popping up immediately)
                    new_spawn_time = current_time + 1.0
                    timer_vec.SetAtIndex(0, new_spawn_time)

        
    def UpdateLogic(self, context, discrete_state):
        current_time = context.get_time()
        
        # Mutable State
        timer_vec = discrete_state.get_mutable_vector(self._timer_index)
        moles_vec = discrete_state.get_mutable_vector(self._moles_index)
        activation_vec = discrete_state.get_mutable_vector(self._activation_times_index)

        # Get the next switch, aka next time a mole will move
        next_switch = timer_vec.get_value()[0]
        if next_switch == 0.0:
            if self.random_switch_time:
                step = random.uniform(self.min_time, self.max_time)
            else:
                step = self.max_time
            next_switch = current_time + step
            timer_vec.SetAtIndex(0, next_switch)
        
        # Update plant contexts
        plant_state = self.EvalVectorInput(context, self._state_input_index).get_value()
        self.plant.SetPositionsAndVelocities(self.context_plant, plant_state)
        contact_results = self.EvalAbstractInput(context, self._contact_input_index).get_value()
        
        # Hit Detection
        self.handle_hits(moles_vec, activation_vec, timer_vec, current_time)
        
        # Pick next mole action
        if current_time >= next_switch:
            # Pick random mole index (this is the next mole that can move)
            target_idx = random.randint(0, self.num_moles - 1)
            if self.multi_mode:
                # MULTI MODE: Toggle this specific mole
                current_val = moles_vec.get_value()[target_idx]
                new_val = 1.0 - current_val # flip state of the selected mole
                moles_vec.SetAtIndex(target_idx, new_val)
                if new_val > 0.5:
                    activation_vec.SetAtIndex(target_idx, current_time)
                    state_str = "UP"
                else:
                    state_str = "DOWN"
                print(f"[Mole] Mole {target_idx} -> {state_str} (Multi Mode)")
            else:
                # SINGLE MODE: Clear all, set one
                moles_vec.SetFromVector(np.zeros(self.num_moles))
                moles_vec.SetAtIndex(target_idx, 1.0)
                activation_vec.SetAtIndex(target_idx, current_time)
                print(f"[Mole] Mole {target_idx} UP (Single Mode)")
            
            # Reset Timer
            if self.random_switch_time:
                duration = random.uniform(self.min_time, self.max_time)
            else:
                duration = self.max_time
            timer_vec.SetAtIndex(0, current_time + duration)
            
        return EventStatus.Succeeded()