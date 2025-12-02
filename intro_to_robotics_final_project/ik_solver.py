import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Optional, Union


class OpenManipulatorIK:

    # IK Solver for OpenMANIPULATOR-X 4-DOF robotic arm.
    
    def __init__(self):
        # Calibrated parameters matched to the simulator
        # Base X offset
        self.x_base = 0.011852
        # Base height to shoulder
        self.z_base = 0.076949
        # Upper arm X component
        self.x2 = 0.023998
        # Upper arm Z component
        self.z2 = 0.127956
        # Forearm length
        self.L3 = 0.121527
        # Wrist to end-effector
        self.L4 = 0.128777
        
        self.L2 = np.sqrt(self.x2**2 + self.z2**2)
        self.alpha2 = np.arctan2(self.z2, self.x2)
        
        # Joint 1-4 limits in rad
        self.joint_limits = [(-np.pi, np.pi), (-2.0, 2.0), (-np.pi/2, 1.53),(-1.8, 2.0)]
        
        # Workspace
        self.max_reach = self.x_base + self.L2 + self.L3 + self.L4
    

    def forward_kinematics(self, joints: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute end-effector position from joint angles
        
        Inputs:
            joints: [q1, q2, q3, q4] in radians
            
        Returns:
            [x, y, z] position in meters
        """
        q1, q2, q3, q4 = joints
        # Negate pitch joints for internal calculation
        q2, q3, q4 = -q2, -q3, -q4
        
        # Build chain in 2D (r, z), then rotate by q1
        p = np.array([self.L4, 0.0])
        
        # Wrist rotation
        c4, s4 = np.cos(q4), np.sin(q4)
        p = np.array([c4 * p[0] - s4 * p[1], s4 * p[0] + c4 * p[1]])
        
        # Add forearm
        p[0] += self.L3
        
        # Elbow rotation
        c3, s3 = np.cos(q3), np.sin(q3)
        p = np.array([c3 * p[0] - s3 * p[1], s3 * p[0] + c3 * p[1]])
        
        # Add upper arm
        p[0] += self.x2
        p[1] += self.z2
        
        # Shoulder rotation
        c2, s2 = np.cos(q2), np.sin(q2)
        p = np.array([c2 * p[0] - s2 * p[1], s2 * p[0] + c2 * p[1]])
        
        # Add base
        r = p[0] + self.x_base
        z = p[1] + self.z_base
        
        # Base rotation
        x = r * np.cos(q1)
        y = r * np.sin(q1)
        
        return np.array([x, y, z])
    

    def get_pitch(self, joints: Union[List[float], np.ndarray]) -> float:
        """
        Get end-effector pitch angle (sum of q2+q3+q4)
        
        Returns:
            Pitch angle in radians (horizontal is 0)
        """
        return joints[1] + joints[2] + joints[3]
    

    def _analytical_ik(self, target_pos: List[float], pitch: float = 0.0, elbow_down: bool = True) -> np.ndarray:
        """
        Analytical IK to get a good initial guess

        Inputs:
            target_pos: [x, y, z] target
            pitch: end-effector pitch in joint space (q2+q3+q4)
            elbow_down: elbow configuration

        Returns:
            Joint angles [q1, q2, q3, q4]
        """
        x, y, z = target_pos

        # Convert joint pitch to world-frame pitch
        # world_pitch = -(q2 + q3 + q4) = -joint_pitch
        world_pitch = -pitch

        # Base rotation
        q1 = np.arctan2(y, x)

        # Work in 2D plane (relative to shoulder)
        r = np.sqrt(x**2 + y**2) - self.x_base
        z_rel = z - self.z_base

        # Wrist position (back-project from end-effector)
        r_w = r - self.L4 * np.cos(world_pitch)
        z_w = z_rel - self.L4 * np.sin(world_pitch)

        # Distance squared from shoulder to wrist
        D_sq = r_w**2 + z_w**2

        # Solve for q3n using: D^2 = L3^2 + L2^2 + 2*L3*L2*cos(q3n - alpha2)
        cos_diff = (D_sq - self.L3**2 - self.L2**2) / (2 * self.L3 * self.L2)
        cos_diff = np.clip(cos_diff, -1, 1)
        diff_angle = np.arccos(cos_diff)

        # Two solutions for q3n
        if elbow_down:
            q3n = self.alpha2 + diff_angle
        else:
            q3n = self.alpha2 - diff_angle

        # Vector from shoulder to wrist in shoulder frame
        v_r = self.L3 * np.cos(q3n) + self.x2
        v_z = self.L3 * np.sin(q3n) + self.z2

        # Find q2n: rotation needed to align v with wrist direction
        wrist_angle = np.arctan2(z_w, r_w)
        v_angle = np.arctan2(v_z, v_r)
        q2n = wrist_angle - v_angle

        # q4n from pitch constraint: world_pitch = q2n + q3n + q4n
        q4n = world_pitch - q2n - q3n

        # Convert to robot convention (negate internal angles)
        q2 = -q2n
        q3 = -q3n
        q4 = -q4n

        return np.array([q1, q2, q3, q4])
    
    
    def inverse_kinematics(self, target_pos: List[float], seed: Optional[List[float]] = None) -> Tuple[np.ndarray, float]:
        """
        Solve inverse kinematics
        
        Inputs:
            target_pos: [x, y, z] target position in meters
            seed: reference joint configuration
            
        Returns:
            Tuples: (joint_angles, position_error_in_meters)
        """
        target = np.array(target_pos)

        pitch = None
        if seed is not None:
            pitch = seed[1] + seed[2] + seed[3]
        
        def objective(joints):
            pos = self.forward_kinematics(joints)
            pos_err = np.sum((pos - target)**2)
            
            pitch_err = 0
            if pitch is not None:
                actual_pitch = joints[1] + joints[2] + joints[3]
                pitch_err = 100 * (actual_pitch - pitch)**2
            
            seed_err = 0
            if seed is not None:
                seed_err = 0.001 * np.sum((np.array(joints) - np.array(seed))**2)
            
            return pos_err + pitch_err + seed_err
        
        best_joints = None
        best_error = float('inf')
        
        # Initial guesses
        guesses = []
        
        if seed is not None:
            guesses.append(list(seed))
        
        # Try analytical solutions with different configs
        pitch_values = [pitch] if pitch is not None else [0.0, 0.5, -0.5, 1.0, -1.0]
        
        for p in pitch_values:
            for elbow_down in [True, False]:
                try:
                    guess = self._analytical_ik(target_pos, pitch=p, elbow_down=elbow_down)
                    guesses.append(guess.tolist())
                except:
                    pass
        
        # Optimize from each starting point
        for guess in guesses:
            try:
                result = minimize(
                    objective, guess, method='L-BFGS-B',
                    bounds=self.joint_limits,
                    options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 2000}
                )
                
                pos = self.forward_kinematics(result.x)
                pos_err = np.linalg.norm(pos - target)
                
                if pos_err < best_error:
                    best_error = pos_err
                    best_joints = result.x.copy()
            except:
                pass
        
        return best_joints, best_error
    

    def is_reachable(self, target_pos: List[float]) -> bool:
        """Check if position is within workspace"""
        x, y, z = target_pos
        dist = np.sqrt(x**2 + y**2)
        return 0.05 < dist < self.max_reach
