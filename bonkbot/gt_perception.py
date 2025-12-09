from pydrake.all import LeafSystem, BasicVector, AbstractValue, RigidTransform

class DummyPerception(LeafSystem):
    def __init__(self, plant, mole_indices):
        """
        Args:
            plant: The MultibodyPlant (shared reference).
            mole_indices: List of ModelInstanceIndex for the moles.
        """
        LeafSystem.__init__(self)
        self._plant = plant
        self._context = plant.CreateDefaultContext() # Internal context for calculation
        self._mole_indices = mole_indices
        
        # Input: the full state of the plant (positions + velocities of everything)
        self.DeclareVectorInputPort(
            "plant_state", 
            BasicVector(plant.num_multibody_states())
        )

        # Output: AbstractValue (python dictionary {mole_index: RigidTransform of mole world pose})
        self.DeclareAbstractOutputPort(
            "mole_poses",
            lambda: AbstractValue.Make({}),
            self.CalcMolePoses
        )

    def CalcMolePoses(self, context, output):
        # Update internal plant context with current simulation state
        state_input = self.EvalVectorInput(context, 0).get_value()
        self._plant.SetPositionsAndVelocities(self._context, state_input)
        
        mole_data = {}
        # Compute poses for each mole
        for i, mole_idx in enumerate(self._mole_indices):
            mole_head = self._plant.GetBodyByName("mole", mole_idx)
            X_WMole = self._plant.EvalBodyPoseInWorld(self._context, mole_head)
            mole_data[i] = X_WMole
            
        output.set_value(mole_data)