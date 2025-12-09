from pydrake.all import (LeafSystem, AbstractValue, BasicVector)
from pydrake.multibody.plant import ContactResults
import numpy as np

class HammerContactForce(LeafSystem):
    """
    Reads ContactResults from the plant and outputs the scalar contact force
    on the hammer body along n_hat (in world frame).
    """
    def __init__(self, plant, hammer_body_index, n_hat):
        super().__init__()
        self._plant = plant
        self._hammer_body_index = hammer_body_index
        # Normalize n_hat
        self._n_hat = np.array(n_hat) / np.linalg.norm(n_hat)

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
        contact_results = self.EvalAbstractInput(context, 0).get_value()
        F_W = np.zeros(3)

        # Sum up all forces acting on the hammer
        for i in range(contact_results.num_point_pair_contacts()):
            info = contact_results.point_pair_contact_info(i)
            f_Bc_W = info.contact_force()
            bodyA = info.bodyA_index()
            bodyB = info.bodyB_index()

            if bodyA == self._hammer_body_index:
                F_W -= f_Bc_W    # Force on Hammer is negative
            if bodyB == self._hammer_body_index:
                F_W += f_Bc_W    # Force on Hammer is positive

        F_meas = float(np.dot(self._n_hat, F_W))
        output.SetAtIndex(0, F_meas)

class HitAdmittanceController:
    """
    'Virtual Hammer' Controller with 1D Admittance + joint-space posture spring.
        1D admittance along n_hat using ALL joints to realize the motion; 
        also apply posture spring toward q_prehit.
    
    Physics:
        M*a + D*v + K*s = F_des - F_meas
        
    Kinematics:
        q_cmd = q_anchor + pinv(J) * (n_hat * s)
    """
    def __init__(self, M=0.5, D=10.0, K=0.0, dt=0.01, n_hat=[0,0,-1]):
        self.M = M
        self.D = D
        self.K = K
        self.dt = dt
        self.n_hat = np.array(n_hat) / np.linalg.norm(n_hat)

    def compute_next_state(self, s, s_dot, F_des, F_meas):
        """
        Performs one step of Euler Integration for the admittance physics.
        Returns: (s_new, s_dot_new)
        """
        # F_net opposes motion (F_meas is magnitude against n_hat)
        F_err = F_des - F_meas
        
        # Dynamics: a = (F_err - D*v - K*s) / M
        s_ddot = (F_err - self.D * s_dot - self.K * s) / self.M
        
        # Euler Integration
        s_new = s + s_dot * self.dt
        s_dot_new = s_dot + s_ddot * self.dt
        
        return s_new, s_dot_new

    def compute_q_cmd(self, q_anchor, s, J):
        """
        Converts the admittance scalar 's' into a Joint Configuration command.
        
        Args:
            q_anchor: The joint configuration at the start of the hit (7,)
            s: The current admittance displacement (scalar)
            J: The Jacobian matrix at the current configuration (6,7) or (3,7)
        """
        # Task space displacement vector
        dx = self.n_hat * s
        
        # Inverse Kinematics (Differential)
        # dq = J# * dx
        dq = np.linalg.pinv(J) @ dx\
        
        return q_anchor + dq