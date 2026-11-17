"""
=============================================================================
AeroVision AI - Object Tracking Foundations: Module 01
=============================================================================
Kalman Filter State Estimation for Moving Aerial Objects

Learning Objectives:
1. Understand linear dynamical systems and Gaussian state uncertainty.
2. Formulate the state transition matrix (F) and measurement matrix (H).
3. Implement the Predict and Update equations from scratch in NumPy.
4. Smooth noisy drone GPS and vehicle bounding box trajectory measurements.
"""

import numpy as np
from typing import Tuple, List

class KalmanFilter2D:
    """
    Constant-Velocity 2D Kalman Filter for tracking bounding box center (cx, cy).
    
    State Vector x (4x1):
      x = [cx, cy, vx, vy]^T
      where (cx, cy) is position, and (vx, vy) is velocity in pixels/second.
      
    Measurement Vector z (2x1):
      z = [z_cx, z_cy]^T
    """
    def __init__(self, dt: float = 1.0 / 30.0, acc_std: float = 50.0, measurement_std: float = 8.0):
        self.dt = dt
        
        # State Transition Matrix F (4x4):
        # [ 1  0  dt 0  ]
        # [ 0  1  0  dt ]
        # [ 0  0  1  0  ]
        # [ 0  0  0  1  ]
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement Matrix H (2x4): Maps 4D state to 2D measurement
        # [ 1  0  0  0 ]
        # [ 0  1  0  0 ]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process Noise Covariance Q (Discrete White Noise Acceleration Model):
        # Q = G * G^T * acc_std^2, where G = [dt^2/2, dt^2/2, dt, dt]^T
        q_pos = (dt**4 / 4.0) * (acc_std**2)
        q_vel = (dt**2) * (acc_std**2)
        q_pos_vel = (dt**3 / 2.0) * (acc_std**2)
        self.Q = np.array([
            [q_pos,     0.0,       q_pos_vel, 0.0      ],
            [0.0,       q_pos,     0.0,       q_pos_vel],
            [q_pos_vel, 0.0,       q_vel,     0.0      ],
            [0.0,       q_pos_vel, 0.0,       q_vel    ]
        ], dtype=np.float64)

        # Measurement Noise Covariance R (2x2): Sensor / detector jitter
        self.R = np.eye(2, dtype=np.float64) * (measurement_std**2)

        # State estimate x and Estimate Error Covariance P
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0

    def init_state(self, cx: float, cy: float):
        """Initializes the filter with the first observed measurement."""
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0

    def predict(self) -> Tuple[float, float]:
        """
        Prediction Phase (Extrapolates state forward in time):
          x_pred = F * x
          P_pred = F * P * F^T + Q
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_cx: float, z_cy: float) -> Tuple[float, float]:
        """
        Measurement Update Phase (Fuses prediction with new noisy detector observation):
          y = z - H * x_pred                   (Innovation / residual)
          S = H * P_pred * H^T + R             (Residual covariance)
          K = P_pred * H^T * S^-1              (Optimal Kalman Gain)
          x = x_pred + K * y                   (Updated state estimate)
          P = (I - K * H) * P_pred             (Updated error covariance)
        """
        z = np.array([[z_cx], [z_cy]], dtype=np.float64)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(4, dtype=np.float64)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        return float(self.x[0, 0]), float(self.x[1, 0])

def demonstrate_kalman_trajectory_smoothing():
    print("=" * 70)
    print("STEP 1: 2D KALMAN FILTER STATE ESTIMATION & NOISE REJECTION")
    print("=" * 70)

    np.random.seed(42)
    dt = 1.0 / 30.0 # 30 FPS
    num_steps = 60

    # True drone vehicle trajectory: moves diagonally across camera view
    # Ground truth: constant velocity (vx=120 px/s, vy=60 px/s)
    true_x0, true_y0 = 50.0, 100.0
    vx_true, vy_true = 120.0, 60.0

    true_positions = []
    noisy_measurements = []

    for t_step in range(num_steps):
        t = t_step * dt
        tx = true_x0 + vx_true * t
        ty = true_y0 + vy_true * t
        true_positions.append((tx, ty))

        # Add Gaussian detector noise (sigma = 8.0 pixels)
        noise_x = np.random.normal(0, 8.0)
        noise_y = np.random.normal(0, 8.0)
        noisy_measurements.append((tx + noise_x, ty + noise_y))

    # Initialize Kalman Filter
    kf = KalmanFilter2D(dt=dt, acc_std=50.0, measurement_std=8.0)
    kf.init_state(noisy_measurements[0][0], noisy_measurements[0][1])

    estimated_positions = [noisy_measurements[0]]

    for i in range(1, num_steps):
        # 1. Predict state
        kf.predict()
        # 2. Update with noisy measurement
        est_x, est_y = kf.update(noisy_measurements[i][0], noisy_measurements[i][1])
        estimated_positions.append((est_x, est_y))

    # Compare error once initial velocity has settled (after step 5)
    steady_meas_errors = [np.hypot(m[0] - t[0], m[1] - t[1]) for m, t in zip(noisy_measurements[5:], true_positions[5:])]
    steady_kf_errors = [np.hypot(e[0] - t[0], e[1] - t[1]) for e, t in zip(estimated_positions[5:], true_positions[5:])]

    mean_meas_err = float(np.mean(steady_meas_errors))
    mean_kf_err = float(np.mean(steady_kf_errors))
    reduction_pct = ((mean_meas_err - mean_kf_err) / mean_meas_err) * 100.0

    print(f"Simulation Duration    : {num_steps} frames (2.0 seconds @ 30 FPS)")
    print(f"Mean Detector Error    : {mean_meas_err:.2f} px (Raw observation noise)")
    print(f"Mean Kalman Filter Err : {mean_kf_err:.2f} px (Filtered state estimate)")
    print(f"Variance Error Reduction: {reduction_pct:.1f}% error reduction!")

    assert mean_kf_err < mean_meas_err, "Kalman filter failed to reduce measurement noise!"
    print(f"\nFinal Estimated Velocity: vx={kf.x[2,0]:.1f} px/s (True: {vx_true:.1f}), vy={kf.x[3,0]:.1f} px/s (True: {vy_true:.1f})")
    print("[SUCCESS] Kalman filter mathematically smoothed trajectory and estimated velocities.")
    print()

def main():
    demonstrate_kalman_trajectory_smoothing()
    print("=" * 70)
    print("PHASE 6 - MODULE 01: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
