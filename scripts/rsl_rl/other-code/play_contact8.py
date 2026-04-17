# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
from collections import deque
from importlib.metadata import version

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("TkAgg")  # interactive backend; change to "Qt5Agg" if TkAgg unavailable
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--plot_window", type=int, default=500, help="Number of timesteps shown in the rolling plot.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*args, **kwargs):
        return None

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# =====================================================================
# Real-time rolling-window plotter: GT vs EKF velocity (env 0)
# =====================================================================
class RealtimeVelocityPlotter:
    """Matplotlib real-time rolling-window plotter for vx/vy/vz GT vs EKF.

    Uses interactive mode (`plt.ion()`) and partial redraw (`blit`-like
    manual background caching) so the plot stays responsive even at high
    control frequencies.
    """

    LABELS = ["vx", "vy", "vz"]
    COLORS_GT = ["#1f77b4", "#2ca02c", "#d62728"]   # blue, green, red
    COLORS_EKF = ["#ff7f0e", "#9467bd", "#8c564b"]  # orange, purple, brown

    def __init__(self, window_size: int = 500, dt: float = 0.02):
        self.window_size = window_size
        self.dt = dt

        # ring-buffers
        self.gt_bufs = [deque(maxlen=window_size) for _ in range(3)]
        self.ekf_bufs = [deque(maxlen=window_size) for _ in range(3)]
        self.t_buf = deque(maxlen=window_size)
        self.step_count = 0

        # --- set up figure ---
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        self.fig.suptitle("GT vs EKF Velocity (env 0)", fontsize=13)

        self.lines_gt = []
        self.lines_ekf = []
        for i, ax in enumerate(self.axes):
            (ln_gt,) = ax.plot([], [], color=self.COLORS_GT[i], linewidth=1.2,
                               label=f"GT {self.LABELS[i]}")
            (ln_ekf,) = ax.plot([], [], color=self.COLORS_EKF[i], linewidth=1.2,
                                linestyle="--", label=f"EKF {self.LABELS[i]}")
            self.lines_gt.append(ln_gt)
            self.lines_ekf.append(ln_ekf)
            ax.set_ylabel(f"{self.LABELS[i]} (m/s)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2.0, 2.0)

        self.axes[-1].set_xlabel("time (s)")
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, gt_vel: np.ndarray, ekf_vel: np.ndarray):
        """Push one timestep of data and refresh the plot.

        Args:
            gt_vel:  (3,) ground-truth world-frame velocity [vx, vy, vz]
            ekf_vel: (3,) EKF estimated velocity [vx, vy, vz]
        """
        t = self.step_count * self.dt
        self.t_buf.append(t)
        for i in range(3):
            self.gt_bufs[i].append(float(gt_vel[i]))
            self.ekf_bufs[i].append(float(ekf_vel[i]))
        self.step_count += 1

        t_arr = np.array(self.t_buf)

        for i, ax in enumerate(self.axes):
            gt_arr = np.array(self.gt_bufs[i])
            ekf_arr = np.array(self.ekf_bufs[i])

            self.lines_gt[i].set_data(t_arr, gt_arr)
            self.lines_ekf[i].set_data(t_arr, ekf_arr)

            ax.set_xlim(t_arr[0], t_arr[-1] + self.dt)

            # auto-scale y with a little margin
            all_vals = np.concatenate([gt_arr, ekf_arr])
            if len(all_vals) > 0:
                ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
                margin = max(0.1, (ymax - ymin) * 0.15)
                ax.set_ylim(ymin - margin, ymax + margin)

        # efficient redraw: only canvas.draw_idle + flush
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def reset(self):
        """Clear buffers on episode reset (optional)."""
        for i in range(3):
            self.gt_bufs[i].clear()
            self.ekf_bufs[i].clear()
        self.t_buf.clear()
        self.step_count = 0

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ============================================================
# IMU + 四足足端接触 Augmented-State EKF
#
# 按照手写推导，接触点的增加和去除通过状态增广/缩减实现：
#
# 基础状态 (error state): [εᵛ, εᵖ] ∈ R⁶
# 每个活跃接触脚增广 3 维: [εᵈⁱ] ∈ R³ (contact point world position error)
#
# 增加接触点 (touchdown):
#   d̄ₜ = p̄ₜ + R̄ₜ · h_p(x̄ₜ)           — 接触点世界坐标
#   ηₜᵈ ≈ εₜᵖ + R̄ₜ J_p(x̄ₜ) wₜᵉ        — 一阶泰勒展开
#   εₜⁿᵉʷ = Fₜ εₜ + Gₜ wₜᵉ
#   Pₜⁿᵉʷ = Fₜ Pₜ Fₜᵀ + Gₜ Cov(wₜᵉ) Gₜᵀ
#
# 去除接触点 (liftoff):
#   εₜⁿᵉʷ = M εₜ                       — 选择矩阵去掉对应行列
#   Pₜⁿᵉʷ = M Pₜ Mᵀ
#
# 观测: 接触脚静止约束 (zero velocity at contact point)
#   通过 forward kinematics 差分得到足端速度观测。
#
# 姿态: 独立的互补滤波 (Mahony), 不进入 augmented state。
# ============================================================
class RIEKF:
    def __init__(
        self,
        dt,
        sigma_v=0.45,
        sigma_p=0.05,
        pos_meas_noise_xy=0.08,
        pos_meas_noise_z=0.14,
        vel_meas_noise_xy=0.14,
        vel_meas_noise_z=0.22,
        contact_force_on=18.0,
        contact_force_off=7.0,
        max_omega=8.0,
        max_accel=60.0,
        force_drop_thr=5.0,
        force_drop_weight=0.55,
        max_contact_residual=0.20,
        max_support_speed=3.00,
        vel_meas_alpha=0.88,
        weight_lp_alpha=0.80,
        touchdown_weight_scale=0.25,
        horiz_bias_alpha=0.02,
        horiz_bias_gain=0.12,
        horiz_bias_clip=0.35,
        contact_pos_noise=0.02,
        output_smooth_alpha_xy=0.00,
        output_smooth_alpha_z=0.88,
        output_spike_alpha_xy=0.00,
        output_spike_alpha_z=0.985,
        output_spike_thr_xy=1e9,
        output_spike_thr_z=0.05,
        output_delta_clip_xy=1e9,
        output_delta_clip_z=0.03,
        output_delta_limit_x=0.12,
        output_delta_limit_y=0.12,
        output_accel_xy_stable=6.0,
        output_accel_xy_transition=18.0,
        output_transition_force_scale=80.0,
        tilt_kp=0.12,
        tilt_ki=0.008,
        tilt_accel_tol=1.50,
        tilt_max_omega=2.50,
    ):
        self.dt = float(dt)
        self.g_vec = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        self.sigma_v = float(sigma_v)
        self.sigma_p = float(sigma_p)
        self.pos_meas_noise_xy = float(pos_meas_noise_xy)
        self.pos_meas_noise_z = float(pos_meas_noise_z)
        self.vel_meas_noise_xy = float(vel_meas_noise_xy)
        self.vel_meas_noise_z = float(vel_meas_noise_z)

        self.contact_force_on = float(contact_force_on)
        self.contact_force_off = float(contact_force_off)
        self.max_omega = float(max_omega)
        self.max_accel = float(max_accel)

        self.force_drop_thr = float(force_drop_thr)
        self.force_drop_weight = float(force_drop_weight)

        self.max_contact_residual = float(max_contact_residual)
        self.max_support_speed = float(max_support_speed)
        self.vel_meas_alpha = float(vel_meas_alpha)
        self.weight_lp_alpha = float(weight_lp_alpha)
        self.touchdown_weight_scale = float(touchdown_weight_scale)
        self.horiz_bias_alpha = float(horiz_bias_alpha)
        self.horiz_bias_gain = float(horiz_bias_gain)
        self.horiz_bias_clip = float(horiz_bias_clip)
        self.contact_pos_noise = float(contact_pos_noise)

        self.output_smooth_alpha_xy = float(output_smooth_alpha_xy)
        self.output_smooth_alpha_z = float(output_smooth_alpha_z)
        self.output_spike_alpha_xy = float(output_spike_alpha_xy)
        self.output_spike_alpha_z = float(output_spike_alpha_z)
        self.output_spike_thr_xy = float(output_spike_thr_xy)
        self.output_spike_thr_z = float(output_spike_thr_z)
        self.output_delta_clip_xy = float(output_delta_clip_xy)
        self.output_delta_clip_z = float(output_delta_clip_z)
        self.output_delta_limit_x = float(output_delta_limit_x)
        self.output_delta_limit_y = float(output_delta_limit_y)
        self.output_accel_xy_stable = float(output_accel_xy_stable)
        self.output_accel_xy_transition = float(output_accel_xy_transition)
        self.output_transition_force_scale = float(output_transition_force_scale)

        self.tilt_kp = float(tilt_kp)
        self.tilt_ki = float(tilt_ki)
        self.tilt_accel_tol = float(tilt_accel_tol)
        self.tilt_max_omega = float(tilt_max_omega)

        self.reset()

    # ------------------------------------------------------------------
    #  Augmented state layout:
    #    indices 0:3  = εᵛ  (velocity error)
    #    indices 3:6  = εᵖ  (position error)
    #    for each active contact i (in order of self.active_contacts):
    #      indices 6+3*k : 6+3*(k+1)  = εᵈⁱ (contact point position error)
    #  where k is the position of leg i in self.active_contacts list.
    # ------------------------------------------------------------------

    def _state_dim(self):
        """Current augmented state dimension."""
        return 6 + 3 * len(self.active_contacts)

    def _contact_state_index(self, leg_idx):
        """Return the start index in the state vector for a given active contact leg."""
        k = self.active_contacts.index(leg_idx)
        return 6 + 3 * k

    def reset(self):
        self.R = np.eye(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.p = np.zeros(3, dtype=np.float64)

        # augmented state: base 6x6, plus 3 per active contact
        self.P = np.diag([
            0.25, 0.25, 0.25,
            0.50, 0.50, 0.50,
        ]).astype(np.float64)

        # active contact legs (ordered list of leg indices 0-3 currently in state)
        self.active_contacts = []
        # nominal contact point positions in world frame (per leg)
        self.contact_d_w = np.zeros((4, 3), dtype=np.float64)

        # contact mode (hysteresis thresholds)
        self.contact_mode = np.zeros(4, dtype=bool)
        self.contact_weight_lp = np.zeros(4, dtype=np.float64)

        # slow horizontal velocity bias compensation
        self.v_bias_xy = np.zeros(2, dtype=np.float64)

        # velocity measurement low-pass
        self.v_meas_lp = np.zeros(3, dtype=np.float64)
        self.v_meas_lp_valid = False

        # post-smoothing of the reported EKF velocity curve
        self.v_hat = np.zeros(3, dtype=np.float64)
        self.v_hat_valid = False
        self.prev_output_force = np.zeros(4, dtype=np.float64)
        self.prev_output_contact_mask = np.zeros(4, dtype=bool)
        self.force_norm_cache = np.zeros(4, dtype=np.float64)

        # force diagnostics
        self.prev_force = np.full(4, np.nan, dtype=np.float64)
        self.last_delta_f = np.zeros(4, dtype=np.float64)
        self.last_force_decreasing = np.zeros(4, dtype=bool)

    def reset_kinematics_only(self):
        self.v[:] = 0.0
        self.p[:] = 0.0
        self.P = np.diag([
            0.25, 0.25, 0.25,
            0.50, 0.50, 0.50,
        ]).astype(np.float64)
        self.active_contacts = []
        self.contact_d_w[:] = 0.0
        self.contact_mode[:] = False
        self.contact_weight_lp[:] = 0.0
        self.v_bias_xy[:] = 0.0
        self.v_meas_lp[:] = 0.0
        self.v_meas_lp_valid = False
        self.v_hat[:] = 0.0
        self.v_hat_valid = False
        self.prev_output_force[:] = 0.0
        self.prev_output_contact_mask[:] = False
        self.force_norm_cache[:] = 0.0
        self.prev_force[:] = np.nan
        self.last_delta_f[:] = 0.0
        self.last_force_decreasing[:] = False

    def set_initial_orientation(self, R0):
        self.R = np.asarray(R0, dtype=np.float64).copy()

    @staticmethod
    def skew(v):
        v = np.asarray(v, dtype=np.float64)
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ], dtype=np.float64)

    @staticmethod
    def exp_so3(phi):
        phi = np.asarray(phi, dtype=np.float64)
        angle = np.linalg.norm(phi)
        if angle < 1e-8:
            return np.eye(3, dtype=np.float64) + RIEKF.skew(phi)
        axis = phi / angle
        K = RIEKF.skew(axis)
        return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    def _symmetrize_P(self):
        n = self._state_dim()
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(n, dtype=np.float64) * 1e-9

    def _clip_vector(self, vec, max_norm):
        n = np.linalg.norm(vec)
        if n > max_norm and n > 1e-9:
            return vec * (max_norm / n)
        return vec

    # ------------------------------------------------------------------
    #  接触点增加 (Augmentation) — 按手写公式
    # ------------------------------------------------------------------
    def _augment_contact(self, leg_idx, h_b):
        """Add a contact point to the augmented state.

        From the handwritten derivation:
          d̄ₜ = p̄ₜ + R̄ₜ · h_p(x̄ₜ)
          ηₜᵈ ≈ εₜᵖ + R̄ₜ J_p(x̄ₜ) wₜᵉ

        The augmented state transition:
          [εᴿ]     [I  0  0] [εᴿ]   [   0   ]
          [εᵛ]  =  [0  I  0] [εᵛ] + [   0   ] wₜᵉ
          [εᵈ]     [0  0  I] [εᵖ]   [R̄ₜ J_p]
                      Fₜ               Gₜ

        Since we don't track εᴿ in the state (attitude is separate Mahony),
        and J_p for a rigid body is just I₃ (position of foot in body frame
        maps directly), we have:

          Fₜ = [I₆;  [0₃ₓ₃, I₃]]    (new row copies εᵖ)
          Gₜ = [0₆ₓ₃;  R̄ₜ]          (noise from joint encoder uncertainty)
        """
        h_b = np.asarray(h_b, dtype=np.float64)

        # nominal contact point in world frame
        d_w = self.p + self.R @ h_b
        self.contact_d_w[leg_idx] = d_w.copy()

        # current state dimension before augmentation
        n_old = self._state_dim()
        n_new = n_old + 3

        # Build Fₜ: (n_new x n_old)
        Ft = np.zeros((n_new, n_old), dtype=np.float64)
        Ft[:n_old, :n_old] = np.eye(n_old, dtype=np.float64)
        # εᵈ_new = εᵖ (copy from position block)
        Ft[n_old:n_old+3, 3:6] = np.eye(3, dtype=np.float64)

        # Build Gₜ: (n_new x 3)
        Gt = np.zeros((n_new, 3), dtype=np.float64)
        Gt[n_old:n_old+3, :] = self.R

        # Covariance of joint encoder noise
        sigma_w = self.contact_pos_noise
        Q_w = np.eye(3, dtype=np.float64) * (sigma_w ** 2)

        # Augmented covariance: Pₜⁿᵉʷ = Fₜ Pₜ Fₜᵀ + Gₜ Cov(wₜᵉ) Gₜᵀ
        self.P = Ft @ self.P @ Ft.T + Gt @ Q_w @ Gt.T

        # Add to active contacts list
        self.active_contacts.append(leg_idx)

        self._symmetrize_P()

    # ------------------------------------------------------------------
    #  接触点去除 (Marginalization) — 按手写公式
    # ------------------------------------------------------------------
    def _remove_contact(self, leg_idx):
        """Remove a contact point from the augmented state.

        From the handwritten derivation:
          εₜⁿᵉʷ = M εₜ       (selection matrix)
          Pₜⁿᵉʷ = M Pₜ Mᵀ
        """
        if leg_idx not in self.active_contacts:
            return

        k = self.active_contacts.index(leg_idx)
        start = 6 + 3 * k
        n_old = self._state_dim()

        # Build selection matrix M: (n_new x n_old)
        keep_indices = list(range(0, start)) + list(range(start + 3, n_old))
        n_new = len(keep_indices)
        M = np.zeros((n_new, n_old), dtype=np.float64)
        for i, j in enumerate(keep_indices):
            M[i, j] = 1.0

        # Pₜⁿᵉʷ = M Pₜ Mᵀ
        self.P = M @ self.P @ M.T

        # Remove from active contacts
        self.active_contacts.remove(leg_idx)

        self._symmetrize_P()

    # ------------------------------------------------------------------
    #  Attitude prediction (unchanged — Mahony complementary filter)
    # ------------------------------------------------------------------
    def _predict_attitude(self, omega_tilde):
        omega_tilde = np.asarray(omega_tilde, dtype=np.float64)
        omega_corr = np.clip(omega_tilde - self.bg, -self.max_omega, self.max_omega)
        self.R = self.R @ self.exp_so3(omega_corr * self.dt)
        return omega_corr

    # ------------------------------------------------------------------
    #  Kinematics prediction with augmented state
    # ------------------------------------------------------------------
    def _predict_kinematics(self, a_b):
        a_b = np.asarray(a_b, dtype=np.float64)
        a_b = np.clip(a_b, -self.max_accel, self.max_accel)
        a_w = self.R @ a_b + self.g_vec

        dt = self.dt
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt

        n = self._state_dim()
        F = np.eye(n, dtype=np.float64)
        F[3:6, 0:3] = np.eye(3, dtype=np.float64) * dt

        qv = (self.sigma_v ** 2) * dt * dt
        qp = (self.sigma_p ** 2) * dt * dt
        Q = np.zeros((n, n), dtype=np.float64)
        Q[0, 0] = Q[1, 1] = Q[2, 2] = qv
        Q[3, 3] = Q[4, 4] = Q[5, 5] = qp

        self.P = F @ self.P @ F.T + Q
        self._symmetrize_P()

    def _update_attitude_from_gravity(self, a_b, omega_corr, support_weight_sum):
        if support_weight_sum < 0.5:
            return False
        if np.linalg.norm(omega_corr) > self.tilt_max_omega:
            return False

        a_b = np.asarray(a_b, dtype=np.float64)
        a_norm = np.linalg.norm(a_b)
        if a_norm < 1e-6 or abs(a_norm - 9.81) > self.tilt_accel_tol:
            return False

        g_pred_body = self.R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        g_pred_body /= max(np.linalg.norm(g_pred_body), 1e-9)
        g_meas_body = -a_b / a_norm

        e = np.cross(g_pred_body, g_meas_body)
        e = self._clip_vector(e, 0.25)

        self.R = self.R @ self.exp_so3(self.tilt_kp * e * self.dt)
        self.bg = self.bg - self.tilt_ki * e * self.dt
        self.bg = np.clip(self.bg, -0.5, 0.5)
        return True

    # ------------------------------------------------------------------
    #  Kalman update on augmented state
    # ------------------------------------------------------------------
    def _kalman_update(self, residual, H, Rm):
        n = self._state_dim()
        S = H @ self.P @ H.T + Rm
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        dx = K @ residual

        dv = dx[0:3]
        dp = dx[3:6]
        self.v = self.v + dv
        self.p = self.p + dp

        for idx_k, leg_idx in enumerate(self.active_contacts):
            start = 6 + 3 * idx_k
            dd = dx[start:start+3]
            self.contact_d_w[leg_idx] += dd

        I_KH = np.eye(n, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        self._symmetrize_P()

    # ------------------------------------------------------------------
    #  Contact-point based position observation
    # ------------------------------------------------------------------
    def _contact_position_update(self, leg_idx, h_b, weight):
        if leg_idx not in self.active_contacts:
            return

        h_b = np.asarray(h_b, dtype=np.float64)
        foot_pred_w = self.p + self.R @ h_b
        d_w = self.contact_d_w[leg_idx]

        residual = d_w - foot_pred_w
        residual = self._clip_vector(residual, self.max_contact_residual)

        n = self._state_dim()
        H = np.zeros((3, n), dtype=np.float64)
        H[:, 3:6] = -np.eye(3, dtype=np.float64)
        start = self._contact_state_index(leg_idx)
        H[:, start:start+3] = np.eye(3, dtype=np.float64)

        sigma_xy = self.pos_meas_noise_xy / np.sqrt(max(weight, 1e-3))
        sigma_z = self.pos_meas_noise_z / np.sqrt(max(weight, 1e-3))
        Rm = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2]).astype(np.float64)

        self._kalman_update(residual, H, Rm)

    # ------------------------------------------------------------------
    #  Velocity observation from contact foot
    # ------------------------------------------------------------------
    def _velocity_update(self, v_meas, weight):
        residual = v_meas - self.v
        residual[0:2] = np.clip(residual[0:2], -0.75, 0.75)
        residual[2] = np.clip(residual[2], -0.60, 0.60)

        n = self._state_dim()
        H = np.zeros((3, n), dtype=np.float64)
        H[:, 0:3] = np.eye(3, dtype=np.float64)

        sigma_xy = self.vel_meas_noise_xy / np.sqrt(max(weight, 1e-3))
        sigma_z = self.vel_meas_noise_z / np.sqrt(max(weight, 1e-3))
        Rm = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2]).astype(np.float64)

        self._kalman_update(residual, H, Rm)

    # ------------------------------------------------------------------
    #  Contact mode detection (hysteresis)
    # ------------------------------------------------------------------
    def _update_contact_mode(self, force_norm, leg_idx):
        was_contact = bool(self.contact_mode[leg_idx])

        prev_force = self.prev_force[leg_idx]
        if np.isnan(prev_force):
            delta_f = 0.0
            is_force_decreasing = False
        else:
            delta_f = float(force_norm - prev_force)
            is_force_decreasing = delta_f < -self.force_drop_thr

        if was_contact:
            contact_now = force_norm > self.contact_force_off
        else:
            contact_now = force_norm > self.contact_force_on

        touchdown = contact_now and (not was_contact)
        liftoff = (not contact_now) and was_contact

        if contact_now:
            w_force_raw = np.clip(
                (force_norm - self.contact_force_off)
                / max(self.contact_force_on - self.contact_force_off, 1e-6),
                0.0,
                1.0,
            )
            if is_force_decreasing:
                w_force_raw *= self.force_drop_weight

            prev_w = self.contact_weight_lp[leg_idx]
            if touchdown:
                w_force = self.touchdown_weight_scale * w_force_raw
            else:
                a = self.weight_lp_alpha
                w_force = a * prev_w + (1.0 - a) * w_force_raw
        else:
            w_force = 0.0

        self.contact_mode[leg_idx] = contact_now
        self.contact_weight_lp[leg_idx] = w_force
        self.prev_force[leg_idx] = float(force_norm)
        self.last_delta_f[leg_idx] = delta_f
        self.last_force_decreasing[leg_idx] = is_force_decreasing

        return contact_now, touchdown, liftoff, w_force

    # ------------------------------------------------------------------
    #  Main step
    # ------------------------------------------------------------------
    def step(self, omega_b, a_b, h_body_list, force_norm_list):
        omega_corr = self._predict_attitude(omega_b)

        leg_infos = []
        support_weight_sum = 0.0
        for leg_idx in range(4):
            force_norm = float(force_norm_list[leg_idx])
            self.force_norm_cache[leg_idx] = force_norm
            contact_now, touchdown, liftoff, w = self._update_contact_mode(force_norm, leg_idx)
            leg_infos.append((contact_now, touchdown, liftoff, w))
            support_weight_sum += w

        self._update_attitude_from_gravity(a_b, omega_corr, support_weight_sum)

        # --- 接触点去除 (liftoff): 先缩减状态 ---
        for leg_idx in range(4):
            _, _, liftoff, _ = leg_infos[leg_idx]
            if liftoff:
                self._remove_contact(leg_idx)

        # --- Predict kinematics on current (possibly reduced) state ---
        self._predict_kinematics(a_b)

        # --- 接触点增加 (touchdown): 增广状态 ---
        for leg_idx in range(4):
            h_b = np.asarray(h_body_list[leg_idx], dtype=np.float64)
            _, touchdown, _, _ = leg_infos[leg_idx]
            if touchdown:
                self._augment_contact(leg_idx, h_b)

        # --- 观测更新: 对每个活跃接触脚 ---
        v_candidates = []
        v_weights = []

        for leg_idx in range(4):
            h_b = np.asarray(h_body_list[leg_idx], dtype=np.float64)
            contact_now, _, _, w = leg_infos[leg_idx]

            if not (contact_now and leg_idx in self.active_contacts and w > 1e-4):
                continue

            self._contact_position_update(leg_idx, h_b, w)

            v_leg_from_contact = np.zeros(3, dtype=np.float64)
            v_candidates.append(v_leg_from_contact)
            v_weights.append(w)

        if len(v_candidates) > 0:
            w_arr = np.asarray(v_weights, dtype=np.float64)
            w_arr = w_arr / np.sum(w_arr)
            v_meas_raw = np.sum(np.stack(v_candidates, axis=0) * w_arr[:, None], axis=0)

            if self.v_meas_lp_valid:
                a = self.vel_meas_alpha
                v_meas = a * self.v_meas_lp + (1.0 - a) * v_meas_raw
            else:
                v_meas = v_meas_raw
                self.v_meas_lp_valid = True
            self.v_meas_lp = v_meas.copy()

            total_w = float(np.sum(v_weights))
            if total_w > 0.9:
                res_xy = np.clip(v_meas[0:2] - self.v[0:2], -self.horiz_bias_clip, self.horiz_bias_clip)
                a = self.horiz_bias_alpha
                self.v_bias_xy = (1.0 - a) * self.v_bias_xy + a * res_xy

            self._velocity_update(v_meas, total_w)
            self.v[0:2] = self.v[0:2] + self.horiz_bias_gain * self.v_bias_xy
        else:
            self.v_meas_lp_valid = False
            self.v_meas_lp[:] = 0.0

        if (
            np.any(np.isnan(self.v))
            or np.any(np.isinf(self.v))
            or np.linalg.norm(self.v) > 5.5
            or np.any(np.isnan(self.R))
            or np.any(np.isnan(self.P))
        ):
            self.reset_kinematics_only()

    # ------------------------------------------------------------------
    #  Output smoothing (same as before)
    # ------------------------------------------------------------------
    def _compute_output_transition_strength(self):
        force = self.force_norm_cache.copy()
        contact_mask = force > self.contact_force_on
        mask_change = float(np.sum(contact_mask != self.prev_output_contact_mask))
        df_sum = float(np.sum(np.abs(force - self.prev_output_force)))

        score_mask = min(mask_change / 2.0, 1.0)
        score_force = min(df_sum / max(self.output_transition_force_scale, 1e-6), 1.0)
        score = max(score_mask, score_force)

        self.prev_output_force = force.copy()
        self.prev_output_contact_mask = contact_mask.copy()
        return score

    def _smooth_velocity_output(self):
        raw_v = self.v.copy()
        if not self.v_hat_valid:
            self.v_hat = raw_v
            self.v_hat_valid = True
            self.prev_output_force = self.force_norm_cache.copy()
            self.prev_output_contact_mask = (self.prev_output_force > self.contact_force_on)
            return

        transition_score = self._compute_output_transition_strength()
        a_xy = (
            (1.0 - transition_score) * self.output_accel_xy_stable
            + transition_score * self.output_accel_xy_transition
        )
        delta_xy_max = float(a_xy * self.dt)

        dxy = raw_v[0:2] - self.v_hat[0:2]
        nxy = float(np.linalg.norm(dxy))
        if nxy > delta_xy_max and nxy > 1e-9:
            dxy = dxy * (delta_xy_max / nxy)
        self.v_hat[0:2] = self.v_hat[0:2] + dxy

        delta_z = float(raw_v[2] - self.v_hat[2])
        spike_z = abs(delta_z) > self.output_spike_thr_z
        alpha_z = self.output_spike_alpha_z if spike_z else self.output_smooth_alpha_z

        clipped_z = float(np.clip(delta_z, -self.output_delta_clip_z, self.output_delta_clip_z))
        target_z = float(self.v_hat[2] + clipped_z)

        self.v_hat[2] = alpha_z * self.v_hat[2] + (1.0 - alpha_z) * target_z

    def get_velocity_hat(self):
        if self.v_hat_valid:
            return self.v_hat.copy()
        return self.v.copy()

    def get_g_proj(self):
        return self.R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def get_omega_hat(self, omega_tilde):
        omega_tilde = np.asarray(omega_tilde, dtype=np.float64)
        return np.clip(omega_tilde - self.bg, -self.max_omega, self.max_omega)


class BatchRIEKF:
    """N 个 env 并行的 hybrid contact filter 管理器"""

    def __init__(self, num_envs, dt, **kwargs):
        self.num_envs = num_envs
        self.filters = [RIEKF(dt, **kwargs) for _ in range(num_envs)]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        for i in env_ids:
            self.filters[i].reset()

    def set_initial_orientation(self, R_batch, env_ids=None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        for i in env_ids:
            self.filters[i].set_initial_orientation(R_batch[i])

    def step(self, omega_tilde, a_tilde, h_p_body_list, force_norm_list):
        for i, f in enumerate(self.filters):
            h_list_i = [h_p_body_list[leg][i] for leg in range(4)]
            f_list_i = [force_norm_list[leg][i] for leg in range(4)]
            f.step(omega_tilde[i], a_tilde[i], h_list_i, f_list_i)

    def get_obs(self, omega_tilde):
        omega_hat = np.stack(
            [f.get_omega_hat(omega_tilde[i]) for i, f in enumerate(self.filters)],
            axis=0,
        )
        g_proj = np.stack([f.get_g_proj() for f in self.filters], axis=0)
        return omega_hat, g_proj

    def get_velocity_hat(self):
        return np.stack([f.get_velocity_hat() for f in self.filters], axis=0)


def main():
    """Play with RSL-RL agent."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    cf_sensor = raw_env.scene["contact_forces"]
    imu_sensor = raw_env.scene["imu_sensor"]

    print(f"[IMU_EKF] robot.body_names:          {robot.body_names}")
    print(f"[IMU_EKF] contact_forces.body_names: {cf_sensor.body_names}")

    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_body_ids = [robot.body_names.index(n) for n in foot_names]
    foot_contact_ids = [cf_sensor.body_names.index(n) for n in foot_names]

    print(f"foot_body_ids: {foot_body_ids}")
    print(f"foot_contact_ids: {foot_contact_ids}")

    riekf = BatchRIEKF(
        num_envs=raw_env.num_envs,
        dt=dt,
        sigma_v=0.45,
        sigma_p=0.05,
        pos_meas_noise_xy=0.08,
        pos_meas_noise_z=0.14,
        vel_meas_noise_xy=0.14,
        vel_meas_noise_z=0.22,
        contact_force_on=18.0,
        contact_force_off=7.0,
        max_omega=8.0,
        max_accel=60.0,
        force_drop_thr=5.0,
        force_drop_weight=0.55,
        max_contact_residual=0.20,
        max_support_speed=3.00,
        vel_meas_alpha=0.88,
        weight_lp_alpha=0.80,
        touchdown_weight_scale=0.25,
        horiz_bias_alpha=0.02,
        horiz_bias_gain=0.12,
        horiz_bias_clip=0.35,
        contact_pos_noise=0.02,
        output_smooth_alpha_xy=0.00,
        output_smooth_alpha_z=0.88,
        output_spike_alpha_xy=0.00,
        output_spike_alpha_z=0.985,
        output_spike_thr_xy=1e9,
        output_spike_thr_z=0.05,
        output_delta_clip_xy=1e9,
        output_delta_clip_z=0.03,
        output_delta_limit_x=0.12,
        output_delta_limit_y=0.12,
        tilt_kp=0.12,
        tilt_ki=0.008,
        tilt_accel_tol=1.50,
        tilt_max_omega=2.50,
    )

    # =====================================================================
    # Initialize real-time plotter
    # =====================================================================
    plotter = RealtimeVelocityPlotter(window_size=args_cli.plot_window, dt=dt)
    print(f"[INFO] Real-time plotter initialized (window={args_cli.plot_window} steps)")

    def quats_to_R_batch(quats):
        """quats: (N,4) in (w,x,y,z) -> R_batch: (N,3,3)"""
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        N = quats.shape[0]
        R = np.zeros((N, 3, 3), dtype=np.float64)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def dones_to_numpy(dones):
        if isinstance(dones, torch.Tensor):
            return dones.detach().cpu().numpy().astype(bool)
        return np.asarray(dones).astype(bool)

    # reset environment
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    timestep = 0
    step_counter = 0
    riekf_initialized = False

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # IMU measurements in body frame
            omega_tilde = imu_sensor.data.ang_vel_b.cpu().numpy()
            a_tilde = imu_sensor.data.lin_acc_b.cpu().numpy()

            # root pose
            root_pos_w = robot.data.root_pos_w.cpu().numpy()
            root_quat_w = robot.data.root_quat_w.cpu().numpy()
            R_batch = quats_to_R_batch(root_quat_w)

            # body-frame foot positions and contact force norms
            h_p_body_list = []
            force_norm_list = []
            for bid, cid in zip(foot_body_ids, foot_contact_ids):
                foot_pos_w = robot.data.body_pos_w[:, bid, :].cpu().numpy()
                diff_w = foot_pos_w - root_pos_w
                h_p_b = np.einsum("nij,nj->ni", R_batch.transpose(0, 2, 1), diff_w)
                h_p_body_list.append(h_p_b)

                cf_norm = cf_sensor.data.net_forces_w[:, cid, :].norm(dim=-1).cpu().numpy()
                force_norm_list.append(cf_norm)

            # one-time init with GT orientation only
            if not riekf_initialized:
                riekf.set_initial_orientation(R_batch)
                riekf_initialized = True

            imu_ok = (
                (not np.isnan(omega_tilde).any())
                and (not np.isnan(a_tilde).any())
                and (np.nanmax(np.abs(a_tilde)) < 80.0)
            )
            if imu_ok:
                riekf.step(omega_tilde, a_tilde, h_p_body_list, force_norm_list)

            # overwrite policy observations
            omega_hat, g_proj = riekf.get_obs(omega_tilde)
            obs[:, 0:3] = torch.from_numpy(omega_hat * 0.2).to(obs.device).float()
            obs[:, 3:6] = torch.from_numpy(g_proj).to(obs.device).float()

            # step environment
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step_counter += 1

            # IMPORTANT: reset filter on episode reset / fall
            dones_np = dones_to_numpy(dones)
            done_ids = np.flatnonzero(dones_np)
            if len(done_ids) > 0:
                root_quat_after = robot.data.root_quat_w.cpu().numpy()
                R_after = quats_to_R_batch(root_quat_after)
                riekf.reset(done_ids)
                riekf.set_initial_orientation(R_after, done_ids)
                print(f"[IMU_EKF] reset filters for env_ids={done_ids.tolist()}")
                # Reset plotter if env 0 was reset
                if 0 in done_ids:
                    plotter.reset()

            # ---- Real-time plot update (env 0 only) ----
            gt_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
            ekf_vel = riekf.filters[0].get_velocity_hat()
            plotter.update(gt_vel, ekf_vel)

            # terminal log
            force_dbg = [float(force_norm_list[k][0]) for k in range(4)]
            delta_f_dbg = [float(riekf.filters[0].last_delta_f[k]) for k in range(4)]
            dec_dbg = [bool(riekf.filters[0].last_force_decreasing[k]) for k in range(4)]
            print(
                f"[step {step_counter:4d}] "
                f"GT  vx={gt_vel[0]:+.3f} vy={gt_vel[1]:+.3f} vz={gt_vel[2]:+.3f} | "
                f"EKF vx={ekf_vel[0]:+.3f} vy={ekf_vel[1]:+.3f} vz={ekf_vel[2]:+.3f} | "
                f"F={force_dbg} | dF={delta_f_dbg} | dec={dec_dbg}"
            )

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Clean up plotter
    plotter.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()