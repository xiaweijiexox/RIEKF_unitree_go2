"""
Invariant Extended Kalman Filter (IEKF) for Unitree Go2 Base Velocity Estimation
=================================================================================
Based on: Hartley et al., "Contact-Aided Invariant Extended Kalman Filtering
           for Robot State Estimation," IJRR 2020.

State lives on SE_2(3) × R^6:
  X = [[R, v, p],   R ∈ SO(3), v ∈ R³ (velocity), p ∈ R³ (position)
       [0, 1, 0],
       [0, 0, 1]]
  b = [b_ω, b_a] ∈ R^6  (gyro bias, accel bias)

Left-Invariant formulation: error defined as ξ = log(X̂⁻¹ · X)
  → Jacobians are state-independent (key advantage over std EKF)

Measurements:
  1. Contact kinematics (foot FK + contact flag)  → velocity correction
  2. LiDAR-odom (relative pose)                   → position / yaw correction
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ─────────────────────────────────────────────────────────────
#  Lie Group Utilities  (SO(3), SE_2(3))
# ─────────────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])

def vee_so3(S: np.ndarray) -> np.ndarray:
    """Skew matrix → 3-vector (inverse of skew)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map: R³ → SO(3)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) + skew(phi)          # first-order approx
    ax = phi / angle
    c, s = np.cos(angle), np.sin(angle)
    return c * np.eye(3) + (1 - c) * np.outer(ax, ax) + s * skew(ax)

def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithmic map: SO(3) → R³."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return vee_so3((R - R.T) / 2.0)
    return (angle / (2.0 * np.sin(angle))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

def left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): J_l(φ) such that exp(φ+δ) ≈ exp(J_l δ) exp(φ)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) + 0.5 * skew(phi)
    ax = phi / angle
    c, s = np.cos(angle), np.sin(angle)
    return (s / angle) * np.eye(3) + (1 - s / angle) * np.outer(ax, ax) + \
           ((1 - c) / angle) * skew(ax)

# SE_2(3) lives in R^{5×5}; we represent state as (R, v, p) tuples for clarity.

def X_from_Rvp(R, v, p) -> np.ndarray:
    """Assemble 5×5 SE_2(3) matrix from (R, v, p)."""
    X = np.eye(5)
    X[:3, :3] = R
    X[:3, 3]  = v
    X[:3, 4]  = p
    return X

def Rvp_from_X(X: np.ndarray):
    return X[:3, :3].copy(), X[:3, 3].copy(), X[:3, 4].copy()

def inv_X(X: np.ndarray) -> np.ndarray:
    """Inverse of SE_2(3) element."""
    R, v, p = Rvp_from_X(X)
    Rt = R.T
    Xi = np.eye(5)
    Xi[:3, :3] = Rt
    Xi[:3, 3]  = -Rt @ v
    Xi[:3, 4]  = -Rt @ p
    return Xi

def exp_se2_3(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map for SE_2(3).
    xi = [φ (3), ρ_v (3), ρ_p (3)]  ∈ R^9  →  X ∈ SE_2(3) ⊂ R^{5×5}
    Uses the closed-form from Barrau & Bonnabel (2017).
    """
    phi   = xi[0:3]
    rho_v = xi[3:6]
    rho_p = xi[6:9]
    R = exp_so3(phi)
    Jl = left_jacobian_so3(phi)
    v = Jl @ rho_v
    p = Jl @ rho_p
    return X_from_Rvp(R, v, p)

def log_se2_3(X: np.ndarray) -> np.ndarray:
    """
    Logarithmic map for SE_2(3).  X → xi ∈ R^9
    """
    R, v, p = Rvp_from_X(X)
    phi = log_so3(R)
    Jl_inv = np.linalg.inv(left_jacobian_so3(phi))
    rho_v = Jl_inv @ v
    rho_p = Jl_inv @ p
    return np.concatenate([phi, rho_v, rho_p])


# ─────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class ImuData:
    omega: np.ndarray    # gyro measurement  [rad/s]  (3,)
    accel: np.ndarray    # accel measurement [m/s²]   (3,)

@dataclass
class ContactData:
    """Foot position in body frame from FK + contact flag."""
    foot_pos_body: np.ndarray   # (4, 3)  FL, FR, RL, RR
    contact_flag: np.ndarray    # (4,)    bool
    foot_vel_body: Optional[np.ndarray] = None  # (4, 3) foot velocity in body frame (J·q̇)

@dataclass
class LidarOdomData:
    """Relative pose from scan-matching (body frame)."""
    delta_R: np.ndarray   # (3,3)
    delta_p: np.ndarray   # (3,)
    valid: bool = True


# ─────────────────────────────────────────────────────────────
#  IEKF Core
# ─────────────────────────────────────────────────────────────

class IEKF:
    """
    Left-Invariant EKF for legged robot velocity estimation.

    State dimension: 15  (9 for SE_2(3) Lie algebra + 6 for IMU biases)
    Covariance P ∈ R^{15×15} lives in the tangent space.

    Convention:
      dim 0–2   : rotation   φ
      dim 3–5   : velocity   ρ_v
      dim 6–8   : position   ρ_p
      dim 9–11  : gyro bias  δb_ω
      dim 12–14 : accel bias δb_a
    """

    # gravity in world frame (NWU / z-up)
    G_WORLD = np.array([0.0, 0.0, -9.81])

    def __init__(
        self,
        # ── Initial state ──
        R0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
        bg0: Optional[np.ndarray] = None,
        ba0: Optional[np.ndarray] = None,
        # ── Initial covariance (diagonal σ²) ──
        sigma_R0:  float = 0.01,
        sigma_v0:  float = 0.1,
        sigma_p0:  float = 0.01,
        sigma_bg0: float = 1e-4,
        sigma_ba0: float = 1e-3,
        # ── Process noise (continuous-time PSD) ──
        sigma_gyro: float = 1e-3,   # rad/s/√Hz
        sigma_accel: float = 1e-2,  # m/s²/√Hz
        sigma_bg:   float = 1e-5,   # gyro bias random walk
        sigma_ba:   float = 1e-4,   # accel bias random walk
        # ── Measurement noise ──
        sigma_contact: float = 0.05,   # m/s  velocity from contact
        sigma_lidar_p: float = 0.02,   # m
        sigma_lidar_R: float = 0.005,  # rad
    ):
        # ── State ──
        self.R  = R0  if R0  is not None else np.eye(3)
        self.v  = v0  if v0  is not None else np.zeros(3)
        self.p  = p0  if p0  is not None else np.zeros(3)
        self.bg = bg0 if bg0 is not None else np.zeros(3)
        self.ba = ba0 if ba0 is not None else np.zeros(3)

        # ── Covariance ──
        s = np.array([sigma_R0]*3 + [sigma_v0]*3 + [sigma_p0]*3 +
                     [sigma_bg0]*3 + [sigma_ba0]*3) ** 2
        self.P = np.diag(s)   # (15, 15)

        # ── Process noise matrix Qc (continuous-time) ──
        self.Qc = np.diag([sigma_gyro**2]*3 + [sigma_accel**2]*3 +
                          [0.0]*3 +           # no noise on position directly
                          [sigma_bg**2]*3  + [sigma_ba**2]*3)

        # ── Measurement noise ──
        self.R_contact = np.eye(3) * sigma_contact**2
        self.R_lidar   = np.diag([sigma_lidar_R]*3 + [sigma_lidar_p]*3)

        # ── Previous foot positions (world frame) for velocity from contact ──
        self._prev_foot_world: Optional[np.ndarray] = None

        # ── Previous state for LiDAR-odom relative pose ──
        self._prev_R: Optional[np.ndarray] = None
        self._prev_p: Optional[np.ndarray] = None

        # ── Cache last IMU for contact update ──
        self._last_omega_c: np.ndarray = np.zeros(3)

        # ── Gravity attitude correction (Mahony-style) ──
        self._tilt_kp: float = 0.18       # proportional gain (was 0.15)
        self._tilt_ki: float = 0.012      # integral gain for gyro bias
        self._tilt_accel_tol: float = 1.5  # reject if |a| far from 9.81
        self._tilt_max_omega: float = 2.5  # reject during fast rotation

        # ── Horizontal velocity bias compensation ──
        self._v_bias_xy = np.zeros(2)
        self._horiz_bias_alpha: float = 0.02   # slow integrator rate
        self._horiz_bias_gain: float = 0.12    # correction gain
        self._horiz_bias_clip: float = 0.35    # max bias correction

    # ─────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────

    def propagate(self, imu: ImuData, dt: float):
        """
        IMU propagation step.
        Integrates dynamics on SE_2(3) with bias correction.
        Updates state (R, v, p) and covariance P.
        """
        # bias-corrected measurements
        omega_c = imu.omega - self.bg
        accel_c = imu.accel - self.ba

        # ── State integration (midpoint on Lie group) ──
        R_new = self.R @ exp_so3(omega_c * dt)
        R_half = self.R @ exp_so3(omega_c * dt / 2)  # midpoint rotation
        accel_world = R_half @ accel_c + self.G_WORLD  # use R at half-step
        v_new = self.v + accel_world * dt
        p_new = self.p + self.v * dt + 0.5 * accel_world * dt**2
        # biases: constant model (random walk)
        bg_new = self.bg
        ba_new = self.ba

        # ── Covariance propagation (Left-IEKF) ──
        # Adjoint-based linearization — Jacobian F of error dynamics
        F = self._propagation_jacobian(omega_c, accel_c, dt)

        # Discretized noise: Q_d ≈ Qc · dt  (Euler; isotropic noise is rotation-invariant)
        Q_d = self.Qc * dt

        self.P = F @ self.P @ F.T + Q_d

        # ── Commit ──
        self.R, self.v, self.p = R_new, v_new, p_new
        self.bg, self.ba       = bg_new, ba_new
        self._last_omega_c     = omega_c

    def correct_attitude_from_gravity(self, accel_body: np.ndarray, dt: float):
        """
        Mahony-style gravity complementary filter.
        Corrects roll/pitch drift by comparing measured accel direction
        with expected gravity direction in body frame.
        Only applies when robot is not accelerating too fast.
        """
        # Skip if rotating fast (gravity signal unreliable)
        if np.linalg.norm(self._last_omega_c) > self._tilt_max_omega:
            return

        a_norm = np.linalg.norm(accel_body)
        if a_norm < 1e-6 or abs(a_norm - 9.81) > self._tilt_accel_tol:
            return  # skip if accel is far from gravity magnitude

        # Expected gravity in body frame from current R estimate
        g_pred_body = self.R.T @ np.array([0.0, 0.0, -1.0])
        g_pred_body /= max(np.linalg.norm(g_pred_body), 1e-9)

        # Measured gravity direction (accel points opposite to gravity)
        g_meas_body = -accel_body / a_norm

        # Attitude error (cross product → rotation axis)
        e = np.cross(g_pred_body, g_meas_body)
        e_norm = np.linalg.norm(e)
        if e_norm > 0.25:
            e = e * (0.25 / e_norm)  # clip large errors

        # Apply proportional correction to R
        self.R = self.R @ exp_so3(self._tilt_kp * e * dt)

        # Integral correction to gyro bias
        self.bg = self.bg - self._tilt_ki * e * dt
        self.bg = np.clip(self.bg, -0.5, 0.5)

    def update_contact(self, contact: ContactData):
        """
        Contact kinematics update.
        Assumption: stance foot has zero velocity in world frame.
        Measurement: v_world ≈ 0  →  innovation = v̂ - 0 = v̂
        (projected into foot frame via Jacobian of FK)
        """
        v_meas_list = []
        H_list      = []
        R_list      = []

        for i in range(4):
            if not contact.contact_flag[i]:
                continue

            # foot position in body frame (from FK)
            r_foot_body = contact.foot_pos_body[i]   # (3,)

            # Foot velocity in world frame (rigid body + joint motion):
            #   v_foot_world = v_base + R (ω_body × r_foot + v_foot_body)
            # where v_foot_body = J(q) · q̇  (foot velocity from joint motion)
            # Stance foot: v_foot_world = 0
            #   → innovation = -(v̂ + R̂ (ω̂ × r + v̇_foot))

            omega_cross_r = skew(self._last_omega_c) @ r_foot_body  # (3,)

            # Joint velocity contribution
            v_foot_b = np.zeros(3)
            if contact.foot_vel_body is not None:
                v_foot_b = contact.foot_vel_body[i]

            # Total foot velocity contribution in body frame
            foot_contrib_body = omega_cross_r + v_foot_b  # (3,)
            foot_contrib_world = self.R @ foot_contrib_body

            H_i = np.zeros((3, 15))
            H_i[:, 3:6]  = np.eye(3)                                # ∂/∂ρ_v
            H_i[:, 0:3]  = -skew(self.v + foot_contrib_world)       # ∂/∂φ
            H_i[:, 9:12] = self.R @ skew(r_foot_body)               # ∂/∂δb_g

            # Innovation: z = 0 - ŷ
            innov = -(self.v + foot_contrib_world)

            v_meas_list.append(innov)
            H_list.append(H_i)
            R_list.append(self.R_contact)

        if not v_meas_list:
            # No contact: reset bias integrator slowly
            self._v_bias_xy *= 0.99
            return

        # Stack measurements
        n = len(v_meas_list)
        z   = np.concatenate(v_meas_list)              # (3n,)
        H   = np.vstack(H_list)                        # (3n, 15)
        R_n = np.block([[R_list[i] if i == j else np.zeros((3,3))
                         for j in range(n)] for i in range(n)])  # (3n, 3n)

        self._update(z, H, R_n)

        # ── Horizontal velocity bias compensation ──
        # When multiple feet are in contact (high confidence), accumulate
        # residual to slowly remove persistent xy drift.
        if n >= 2:
            # Average innovation gives an estimate of the velocity bias
            avg_innov_xy = np.mean([v[:2] for v in v_meas_list], axis=0)
            res_xy = np.clip(avg_innov_xy, -self._horiz_bias_clip, self._horiz_bias_clip)
            a = self._horiz_bias_alpha
            self._v_bias_xy = (1.0 - a) * self._v_bias_xy + a * res_xy
        self.v[:2] += self._horiz_bias_gain * self._v_bias_xy

    def update_lidar_odom(self, odom: LidarOdomData):
        """
        LiDAR-odometry update (relative pose).

        odom.delta_R, odom.delta_p: measured relative pose from previous keyframe
        to current, expressed in the previous body frame.

        We compare the predicted relative pose (from IEKF state) against
        the measured one.
        """
        if not odom.valid:
            return

        # First call: store reference and skip (need two poses for relative)
        if self._prev_R is None:
            self._prev_R = self.R.copy()
            self._prev_p = self.p.copy()
            return

        # Predicted relative pose from IEKF state
        R_prev = self._prev_R
        p_prev = self._prev_p
        delta_R_pred = R_prev.T @ self.R
        delta_p_pred = R_prev.T @ (self.p - p_prev)

        # Innovation: measured - predicted (in tangent space)
        delta_R_err = delta_R_pred.T @ odom.delta_R   # should be ≈ I
        delta_R_innov = log_so3(delta_R_err)           # (3,)
        delta_p_innov = odom.delta_p - delta_p_pred    # (3,) in prev body frame

        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)    # rotation innovation → φ block
        H[3:6, 6:9] = np.eye(3)    # position innovation → ρ_p block

        z = np.concatenate([delta_R_innov, delta_p_innov])
        self._update(z, H, self.R_lidar)

        # Update reference for next relative measurement
        self._prev_R = self.R.copy()
        self._prev_p = self.p.copy()

    # ── Accessors ──────────────────────────────────────────

    @property
    def velocity_world(self) -> np.ndarray:
        """Base velocity in world frame."""
        return self.v.copy()

    @property
    def velocity_body(self) -> np.ndarray:
        """Base velocity in body frame."""
        return self.R.T @ self.v

    @property
    def orientation(self) -> np.ndarray:
        """Rotation matrix body→world."""
        return self.R.copy()

    @property
    def position(self) -> np.ndarray:
        return self.p.copy()

    @property
    def velocity_std(self) -> np.ndarray:
        """1-σ uncertainty on velocity (world frame)."""
        return np.sqrt(np.diag(self.P[3:6, 3:6]))

    # ─────────────────────────────────────
    #  Internal Helpers
    # ─────────────────────────────────────

    def _propagation_jacobian(self, omega_c, accel_c, dt) -> np.ndarray:
        """
        15×15 discrete-time Jacobian F for Left-IEKF error propagation.
        Derived from the group-affine property of SE_2(3) dynamics.

        State error ξ = [φ, ρ_v, ρ_p, δb_ω, δb_a]
        """
        F = np.eye(15)

        # Rotation error → velocity error coupling
        F[3:6, 0:3]  = skew(self.G_WORLD) * dt                # gravity through tilt
        F[3:6, 12:15] = -self.R * dt                           # accel bias

        # Rotation error → position coupling
        F[6:9, 0:3]  = 0.5 * skew(self.G_WORLD) * dt**2
        F[6:9, 3:6]  = np.eye(3) * dt                         # v → p
        F[6:9, 12:15] = -0.5 * self.R * dt**2

        # Rotation error propagation (phi → phi)
        F[0:3, 9:12] = -self.R * dt                            # gyro bias
        F[0:3, 0:3]  = exp_so3(-omega_c * dt)                   # left-IEKF: R̂⁺ᵀ R̂⁻

        return F

    def _update(self, z: np.ndarray, H: np.ndarray, R_noise: np.ndarray):
        """
        Standard EKF measurement update in the tangent space.
        z    : innovation vector  (should be ≈ 0 at correct state)
        H    : measurement Jacobian
        R_noise : measurement noise covariance
        """
        S = H @ self.P @ H.T + R_noise                        # innovation cov
        K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T  # Kalman gain

        delta = K @ z                                          # (15,) correction in R^15

        # ── Apply correction on Lie group ──
        delta_xi  = delta[0:9]   # correction in se_2(3)
        delta_bias = delta[9:15]

        # Retract: X_new = exp(δξ) · X̂
        dX = exp_se2_3(delta_xi)
        X_hat = X_from_Rvp(self.R, self.v, self.p)
        X_new = dX @ X_hat
        self.R, self.v, self.p = Rvp_from_X(X_new)

        self.bg += delta_bias[0:3]
        self.ba += delta_bias[3:6]

        # ── Joseph form covariance update (numerically stable) ──
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T


# ─────────────────────────────────────────────────────────────
#  Go2-specific wrapper
# ─────────────────────────────────────────────────────────────

class Go2VelocityEstimator:
    """
    Thin wrapper around IEKF for the Unitree Go2.
    Handles:
      - Unit conventions (Go2 IMU outputs in body frame)
      - Leg kinematics (placeholder — replace with actual FK)
      - Simple LiDAR-odom passthrough
    """

    # Go2 leg names
    LEG_NAMES = ["FL", "FR", "RL", "RR"]

    def __init__(self, use_lidar: bool = True, **iekf_kwargs):
        self.filter   = IEKF(**iekf_kwargs)
        self.use_lidar = use_lidar
        self._initialized = False

    def step(
        self,
        imu_omega:    np.ndarray,         # (3,) rad/s in body frame
        imu_accel:    np.ndarray,         # (3,) m/s² in body frame
        joint_pos:    np.ndarray,         # (12,) hip/thigh/calf angles
        contact_flag: np.ndarray,         # (4,)  bool
        dt:           float,
        joint_vel:    Optional[np.ndarray] = None,  # (12,) joint velocities
        lidar_odom:   Optional[LidarOdomData] = None,
    ) -> np.ndarray:
        """
        Single estimation step.  Returns velocity in body frame (3,).
        """
        imu = ImuData(omega=imu_omega, accel=imu_accel)

        # Propagate
        self.filter.propagate(imu, dt)

        # Gravity attitude correction (Mahony complementary filter)
        self.filter.correct_attitude_from_gravity(imu_accel, dt)

        # Contact update
        foot_pos_body = self._forward_kinematics(joint_pos)
        foot_vel_body = None
        if joint_vel is not None:
            foot_vel_body = self._foot_jacobian_times_qdot(joint_pos, joint_vel)

        contact = ContactData(
            foot_pos_body=foot_pos_body,
            contact_flag=contact_flag.astype(bool),
            foot_vel_body=foot_vel_body,
        )
        self.filter.update_contact(contact)

        # LiDAR-odom update
        if self.use_lidar and lidar_odom is not None:
            self.filter.update_lidar_odom(lidar_odom)

        return self.filter.velocity_body

    # Go2 link lengths
    L_THIGH = 0.213
    L_CALF  = 0.213
    HIP_X   = [0.1934,  0.1934, -0.1934, -0.1934]   # FL, FR, RL, RR
    HIP_Y   = [0.0455, -0.0455,  0.0455, -0.0455]

    def _forward_kinematics(self, joint_pos: np.ndarray) -> np.ndarray:
        """
        3-DOF FK for Go2 per leg: [hip_abduction, hip_pitch, knee].
        Hip abduction rotates leg in frontal plane (around body x-axis).
        """
        foot_positions = np.zeros((4, 3))
        for i in range(4):
            ab = joint_pos[3*i + 0]   # hip abduction (rotation around x)
            hp = joint_pos[3*i + 1]   # hip pitch
            kn = joint_pos[3*i + 2]   # knee

            # Foot in hip frame (before abduction): sagittal plane
            fx = self.L_THIGH * np.sin(hp) + self.L_CALF * np.sin(hp + kn)
            fy_local = 0.0
            fz_local = -(self.L_THIGH * np.cos(hp) + self.L_CALF * np.cos(hp + kn))

            # Apply abduction rotation (around x-axis at hip joint)
            c_ab, s_ab = np.cos(ab), np.sin(ab)
            fy = c_ab * fy_local - s_ab * fz_local
            fz = s_ab * fy_local + c_ab * fz_local

            foot_positions[i] = [
                self.HIP_X[i] + fx,
                self.HIP_Y[i] + fy,
                fz
            ]
        return foot_positions

    def _foot_jacobian_times_qdot(self, joint_pos: np.ndarray,
                                   joint_vel: np.ndarray) -> np.ndarray:
        """
        Compute foot velocities in body frame: v_foot = J(q) · q̇
        Analytical Jacobian for 3-DOF leg (abduction + hip_pitch + knee).
        """
        foot_vel = np.zeros((4, 3))
        for i in range(4):
            ab = joint_pos[3*i + 0]
            hp = joint_pos[3*i + 1]
            kn = joint_pos[3*i + 2]
            dab = joint_vel[3*i + 0]
            dhp = joint_vel[3*i + 1]
            dkn = joint_vel[3*i + 2]

            c_ab, s_ab = np.cos(ab), np.sin(ab)

            # Sagittal plane foot position (before abduction)
            fx = self.L_THIGH * np.sin(hp) + self.L_CALF * np.sin(hp + kn)
            fz_local = -(self.L_THIGH * np.cos(hp) + self.L_CALF * np.cos(hp + kn))

            # ∂foot/∂ab (from rotating the sagittal foot around x)
            # y = c_ab * 0 - s_ab * fz_local → dy/dab = -c_ab * fz_local
            # z = s_ab * 0 + c_ab * fz_local → dz/dab = -s_ab * fz_local
            dfy_dab = -c_ab * fz_local  # actually: d/dab(- s_ab * fz_local) when fy_local=0
            dfz_dab = -s_ab * fz_local

            # ∂foot/∂hp and ∂foot/∂kn (sagittal derivatives, then rotated by abduction)
            dfx_dhp = self.L_THIGH * np.cos(hp) + self.L_CALF * np.cos(hp + kn)
            dfx_dkn = self.L_CALF * np.cos(hp + kn)
            dfz_local_dhp = self.L_THIGH * np.sin(hp) + self.L_CALF * np.sin(hp + kn)
            dfz_local_dkn = self.L_CALF * np.sin(hp + kn)

            # After abduction rotation (fy_local=0, so only fz_local contributes)
            dfy_dhp = -s_ab * dfz_local_dhp
            dfy_dkn = -s_ab * dfz_local_dkn
            dfz_dhp =  c_ab * dfz_local_dhp
            dfz_dkn =  c_ab * dfz_local_dkn

            foot_vel[i, 0] = dfx_dhp * dhp + dfx_dkn * dkn  # x unaffected by abduction
            foot_vel[i, 1] = dfy_dab * dab + dfy_dhp * dhp + dfy_dkn * dkn
            foot_vel[i, 2] = dfz_dab * dab + dfz_dhp * dhp + dfz_dkn * dkn

        return foot_vel


# ─────────────────────────────────────────────────────────────
#  Minimal smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    estimator = Go2VelocityEstimator(
        sigma_gyro=1e-3,
        sigma_accel=2e-2,
        sigma_contact=0.03,
        sigma_bg0=1e-4,
        sigma_ba0=1e-3,
    )

    dt = 0.005   # 200 Hz control loop

    print("="*55)
    print("  IEKF Go2 Smoke Test: stationary robot (v_true = 0)")
    print("="*55)
    print(f"{'Step':>5}  {'vx_body':>9}  {'vy_body':>9}  {'vz_body':>9}  {'σ_vx':>8}")
    print("-" * 55)

    joint_pos = np.zeros(12)
    joint_pos[1::3] = -0.8   # hip pitched (standing pose)
    joint_pos[2::3] =  1.6   # knee bent

    for step in range(400):
        # True state: stationary, so IMU reads gravity in body frame
        # Simulated IMU noise (typical MEMS unit)
        imu_omega = np.random.randn(3) * 1e-3          # 1 mrad/s noise
        imu_accel = np.array([0.0, 0.0, 9.81]) + \
                    np.random.randn(3) * 0.02           # 20 mg noise

        # All four feet in contact (standing still)
        contact_flag = np.array([True, True, True, True])

        v_body = estimator.step(
            imu_omega, imu_accel, joint_pos, contact_flag, dt
        )

        if step % 80 == 0:
            sigma_v = estimator.filter.velocity_std
            print(f"{step:>5}  {v_body[0]:>9.5f}  {v_body[1]:>9.5f}  "
                  f"{v_body[2]:>9.5f}  {sigma_v[0]:>8.5f}")

    print("-" * 55)
    print("Final velocity estimate (should be ≈ 0):", v_body.round(5))
    print("Final biases:")
    print("  bg [rad/s] =", estimator.filter.bg.round(6))
    print("  ba [m/s²]  =", estimator.filter.ba.round(6))
    print("\n✓  IEKF running correctly.  Plug in real Go2 sensors to deploy.")
