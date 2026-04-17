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
# Right-Invariant EKF with Gyro Bias Estimation (no accel bias)
#
# 状态 (名义):
#   R  : SO(3) body->world
#   v  : R³ world-frame velocity
#   p  : R³ world-frame position
#   d  : R³ contact point world position (anchor, 状态外管理)
#   bg : R³ gyro bias
#
# 误差状态 (右不变 + gyro bias):
#   ε = [εᵛ(3), εᵖ(3), ζᵍ(3)] ∈ R⁹
#
# At (去掉 εᴿ 和 ζᵃ 后):
#        | 0   0  -(v̄)×R̄ |
#   At = | I   0  -(p̄)×R̄ |
#        | 0   0    0     |
#
# accel bias 不估计 (可观测性差, 反而引入偏移)
# ============================================================
class RIEKF:
    # [εᵛ(3), εᵖ(3), ζᵍ(3)] = 9 维
    # 索引:  0:3    3:6    6:9
    N_STATE = 9

    def __init__(
        self,
        dt,
        # --- IMU process noise ---
        sigma_gyro=0.01,       # gyro white noise (rad/s/√Hz)
        sigma_accel=0.50,      # accel white noise (m/s²/√Hz)
        sigma_bg=0.0001,       # gyro bias random walk (rad/s²/√Hz)
        # --- position process noise (catch-all) ---
        sigma_p=0.05,
        # --- observation noise ---
        pos_meas_noise_xy=0.08,
        pos_meas_noise_z=0.14,
        vel_meas_noise_xy=0.14,
        vel_meas_noise_z=0.22,
        # --- contact detection ---
        contact_force_on=18.0,
        contact_force_off=7.0,
        force_drop_thr=5.0,
        force_drop_weight=0.55,
        # --- safety clamps ---
        max_omega=8.0,
        max_accel=60.0,
        max_contact_residual=0.20,
        max_support_speed=3.00,
        max_bg=0.5,            # gyro bias clamp (rad/s)
        # --- contact weight smoothing ---
        vel_meas_alpha=0.88,
        weight_lp_alpha=0.80,
        touchdown_weight_scale=0.25,
        # --- horizontal bias (output-side, kept for compatibility) ---
        horiz_bias_alpha=0.02,
        horiz_bias_gain=0.12,
        horiz_bias_clip=0.35,
        # --- output smoothing ---
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
        # --- attitude complementary filter ---
        tilt_kp=0.12,
        tilt_ki=0.0,           # 现在 bg 通过 EKF 估计, Mahony 的 ki 关掉
        tilt_accel_tol=1.50,
        tilt_max_omega=2.50,
    ):
        self.dt = float(dt)
        self.g_vec = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.g_norm = 9.81

        # IMU noise parameters
        self.sigma_gyro = float(sigma_gyro)
        self.sigma_accel = float(sigma_accel)
        self.sigma_bg = float(sigma_bg)
        self.sigma_p = float(sigma_p)

        # observation noise
        self.pos_meas_noise_xy = float(pos_meas_noise_xy)
        self.pos_meas_noise_z = float(pos_meas_noise_z)
        self.vel_meas_noise_xy = float(vel_meas_noise_xy)
        self.vel_meas_noise_z = float(vel_meas_noise_z)

        # contact
        self.contact_force_on = float(contact_force_on)
        self.contact_force_off = float(contact_force_off)
        self.force_drop_thr = float(force_drop_thr)
        self.force_drop_weight = float(force_drop_weight)

        # clamps
        self.max_omega = float(max_omega)
        self.max_accel = float(max_accel)
        self.max_contact_residual = float(max_contact_residual)
        self.max_support_speed = float(max_support_speed)
        self.max_bg = float(max_bg)

        # contact weight
        self.vel_meas_alpha = float(vel_meas_alpha)
        self.weight_lp_alpha = float(weight_lp_alpha)
        self.touchdown_weight_scale = float(touchdown_weight_scale)

        # horizontal bias (output side)
        self.horiz_bias_alpha = float(horiz_bias_alpha)
        self.horiz_bias_gain = float(horiz_bias_gain)
        self.horiz_bias_clip = float(horiz_bias_clip)

        # output smoothing
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

        # attitude
        self.tilt_kp = float(tilt_kp)
        self.tilt_ki = float(tilt_ki)
        self.tilt_accel_tol = float(tilt_accel_tol)
        self.tilt_max_omega = float(tilt_max_omega)

        self.reset()

    def reset(self):
        self.R = np.eye(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.p = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)

        # P: 9x9 for [εᵛ, εᵖ, ζᵍ]
        self.P = np.diag([
            0.25, 0.25, 0.25,   # εᵛ
            0.50, 0.50, 0.50,   # εᵖ
            0.01, 0.01, 0.01,   # ζᵍ  (保守: 假设初始 bias 小)
        ]).astype(np.float64)

        # anchor-based contact (保留原方案)
        self.contact_mode = np.zeros(4, dtype=bool)
        self.anchor_valid = np.zeros(4, dtype=bool)
        self.foot_anchor_w = np.zeros((4, 3), dtype=np.float64)

        self.prev_p_leg_meas = np.full((4, 3), np.nan, dtype=np.float64)
        self.contact_weight_lp = np.zeros(4, dtype=np.float64)
        self.v_meas_lp = np.zeros(3, dtype=np.float64)
        self.v_meas_lp_valid = False

        self.v_bias_xy = np.zeros(2, dtype=np.float64)

        self.v_hat = np.zeros(3, dtype=np.float64)
        self.v_hat_valid = False
        self.prev_output_force = np.zeros(4, dtype=np.float64)
        self.prev_output_contact_mask = np.zeros(4, dtype=bool)
        self.force_norm_cache = np.zeros(4, dtype=np.float64)

        self.prev_force = np.full(4, np.nan, dtype=np.float64)
        self.last_delta_f = np.zeros(4, dtype=np.float64)
        self.last_force_decreasing = np.zeros(4, dtype=bool)

    def reset_kinematics_only(self):
        self.v[:] = 0.0
        self.p[:] = 0.0
        # 保守: reset 时不清零 bias, 因为 bias 是持续量
        # 但重置 bias 协方差到初始值
        self.P = np.diag([
            0.25, 0.25, 0.25,
            0.50, 0.50, 0.50,
            0.01, 0.01, 0.01,
        ]).astype(np.float64)
        self.contact_mode[:] = False
        self.anchor_valid[:] = False
        self.foot_anchor_w[:] = 0.0
        self.prev_p_leg_meas[:] = np.nan
        self.contact_weight_lp[:] = 0.0
        self.v_meas_lp[:] = 0.0
        self.v_meas_lp_valid = False
        self.v_bias_xy[:] = 0.0
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
        N = self.N_STATE
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(N, dtype=np.float64) * 1e-9

    def _clip_vector(self, vec, max_norm):
        n = np.linalg.norm(vec)
        if n > max_norm and n > 1e-9:
            return vec * (max_norm / n)
        return vec

    # ------------------------------------------------------------------
    #  Predict: 名义状态传播 + 协方差传播
    # ------------------------------------------------------------------
    def _predict(self, omega_tilde, a_b):
        omega_tilde = np.asarray(omega_tilde, dtype=np.float64)
        a_b = np.asarray(a_b, dtype=np.float64)

        # bias-corrected IMU (only gyro bias)
        omega = np.clip(omega_tilde - self.bg, -self.max_omega, self.max_omega)
        a_corr = np.clip(a_b, -self.max_accel, self.max_accel)

        dt = self.dt
        R_bar = self.R  # 当前名义 R

        # --- 名义状态传播 ---
        self.R = R_bar @ self.exp_so3(omega * dt)
        a_w = R_bar @ a_corr + self.g_vec
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt

        # --- 离散化 At -> Fd (一阶: Fd ≈ I + At*dt) ---
        # At (9x9), 状态 [εᵛ(3), εᵖ(3), ζᵍ(3)]
        #
        #   d/dt εᵛ = -(v̄)× R̄ ζᵍ     → At[0:3, 6:9] = -(v̄)× R̄
        #   d/dt εᵖ = εᵛ              → At[3:6, 0:3] = I
        #            -(p̄)× R̄ ζᵍ       → At[3:6, 6:9] = -(p̄)× R̄
        #   d/dt ζᵍ = 0
        #
        N = self.N_STATE
        At = np.zeros((N, N), dtype=np.float64)
        At[3:6, 0:3] = np.eye(3, dtype=np.float64)       # εᵖ ← εᵛ
        At[0:3, 6:9] = -self.skew(self.v) @ R_bar        # εᵛ ← ζᵍ
        At[3:6, 6:9] = -self.skew(self.p) @ R_bar        # εᵖ ← ζᵍ

        Fd = np.eye(N, dtype=np.float64) + At * dt

        # --- 过程噪声协方差 Q ---
        sg2 = (self.sigma_gyro * dt) ** 2
        sa2 = (self.sigma_accel * dt) ** 2
        sp2 = (self.sigma_p * dt) ** 2
        sbg2 = (self.sigma_bg * dt) ** 2

        v_norm_sq = np.dot(self.v, self.v)
        qv = sa2 + sg2 * v_norm_sq
        p_norm_sq = np.dot(self.p, self.p)
        qp = sp2 + sg2 * p_norm_sq

        Q = np.diag([
            qv, qv, qv,        # εᵛ
            qp, qp, qp,        # εᵖ
            sbg2, sbg2, sbg2,   # ζᵍ
        ]).astype(np.float64)

        self.P = Fd @ self.P @ Fd.T + Q
        self._symmetrize_P()

        return omega

    # ------------------------------------------------------------------
    #  Attitude complementary filter (Mahony)
    #  bg 现在由 EKF 主导, Mahony 只做 R 校正, 不积 bias
    # ------------------------------------------------------------------
    def _update_attitude_from_gravity(self, a_b, omega_corr, support_weight_sum):
        if support_weight_sum < 0.5:
            return False
        if np.linalg.norm(omega_corr) > self.tilt_max_omega:
            return False

        a_b = np.asarray(a_b, dtype=np.float64)
        a_norm = np.linalg.norm(a_b)
        if a_norm < 1e-6 or abs(a_norm - self.g_norm) > self.tilt_accel_tol:
            return False

        g_pred_body = self.R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        g_pred_body /= max(np.linalg.norm(g_pred_body), 1e-9)
        g_meas_body = -a_b / a_norm

        e = np.cross(g_pred_body, g_meas_body)
        e = self._clip_vector(e, 0.25)

        self.R = self.R @ self.exp_so3(self.tilt_kp * e * self.dt)
        # Mahony 的 ki 积分现在关掉 (tilt_ki=0), bg 由 EKF 管
        if self.tilt_ki > 0:
            self.bg = self.bg - self.tilt_ki * e * self.dt
            self.bg = np.clip(self.bg, -self.max_bg, self.max_bg)
        return True

    # ------------------------------------------------------------------
    #  Kalman Update (9 维)
    # ------------------------------------------------------------------
    def _kalman_update(self, residual, H, Rm):
        N = self.N_STATE
        S = H @ self.P @ H.T + Rm
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        dx = K @ residual  # (9,)

        # 应用修正
        dv = dx[0:3]
        dp = dx[3:6]
        dbg = dx[6:9]

        self.v = self.v + dv
        self.p = self.p + dp
        self.bg = self.bg + dbg

        # clamp bias
        self.bg = np.clip(self.bg, -self.max_bg, self.max_bg)

        I_KH = np.eye(N, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        self._symmetrize_P()

    def _position_update(self, p_leg_meas, weight):
        residual = p_leg_meas - self.p
        residual = self._clip_vector(residual, self.max_contact_residual)

        N = self.N_STATE
        H = np.zeros((3, N), dtype=np.float64)
        H[:, 3:6] = np.eye(3, dtype=np.float64)  # 观测 εᵖ

        sigma_xy = self.pos_meas_noise_xy / np.sqrt(max(weight, 1e-3))
        sigma_z = self.pos_meas_noise_z / np.sqrt(max(weight, 1e-3))
        Rm = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2]).astype(np.float64)
        self._kalman_update(residual, H, Rm)

    def _velocity_update(self, v_meas, weight):
        residual = v_meas - self.v
        residual[0:2] = np.clip(residual[0:2], -0.75, 0.75)
        residual[2] = np.clip(residual[2], -0.60, 0.60)

        N = self.N_STATE
        H = np.zeros((3, N), dtype=np.float64)
        H[:, 0:3] = np.eye(3, dtype=np.float64)  # 观测 εᵛ

        sigma_xy = self.vel_meas_noise_xy / np.sqrt(max(weight, 1e-3))
        sigma_z = self.vel_meas_noise_z / np.sqrt(max(weight, 1e-3))
        Rm = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2]).astype(np.float64)
        self._kalman_update(residual, H, Rm)

    # ------------------------------------------------------------------
    #  Contact mode detection (与原方案完全一致)
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
    #  Main step (anchor-based contact 保留原方案逻辑)
    # ------------------------------------------------------------------
    def step(self, omega_b, a_b, h_body_list, force_norm_list):
        # 1. 先检测 contact mode
        leg_infos = []
        support_weight_sum = 0.0
        for leg_idx in range(4):
            force_norm = float(force_norm_list[leg_idx])
            self.force_norm_cache[leg_idx] = force_norm
            contact_now, touchdown, liftoff, w = self._update_contact_mode(force_norm, leg_idx)
            leg_infos.append((contact_now, touchdown, liftoff, w))
            support_weight_sum += w

        # 2. Predict (名义传播 + 协方差, 包含 bias)
        omega_corr = self._predict(omega_b, a_b)

        # 3. 姿态互补滤波校正
        self._update_attitude_from_gravity(a_b, omega_corr, support_weight_sum)

        # 4. Anchor-based contact observations (与原方案逻辑完全一致)
        p_candidates = []
        p_weights = []
        v_candidates = []
        v_weights = []

        for leg_idx in range(4):
            h_b = np.asarray(h_body_list[leg_idx], dtype=np.float64)
            contact_now, touchdown, liftoff, w = leg_infos[leg_idx]

            if touchdown:
                self.foot_anchor_w[leg_idx] = self.p + self.R @ h_b
                self.anchor_valid[leg_idx] = True
                self.prev_p_leg_meas[leg_idx] = self.p.copy()

            if liftoff:
                self.anchor_valid[leg_idx] = False
                self.prev_p_leg_meas[leg_idx] = np.nan

            if not (contact_now and self.anchor_valid[leg_idx] and w > 1e-4):
                continue

            p_leg_meas = self.foot_anchor_w[leg_idx] - self.R @ h_b
            p_candidates.append(p_leg_meas)
            p_weights.append(w)

            if not np.isnan(self.prev_p_leg_meas[leg_idx]).any():
                v_leg_meas = (p_leg_meas - self.prev_p_leg_meas[leg_idx]) / self.dt
                v_leg_meas = self._clip_vector(v_leg_meas, self.max_support_speed)
                v_candidates.append(v_leg_meas)
                v_weights.append(w)

            self.prev_p_leg_meas[leg_idx] = p_leg_meas.copy()

        # 多脚融合
        if len(p_candidates) > 0:
            w_arr = np.asarray(p_weights, dtype=np.float64)
            w_arr = w_arr / np.sum(w_arr)
            p_meas = np.sum(np.stack(p_candidates, axis=0) * w_arr[:, None], axis=0)
            self._position_update(p_meas, float(np.sum(p_weights)))

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

        # 安全检查
        if (
            np.any(np.isnan(self.v))
            or np.any(np.isinf(self.v))
            or np.linalg.norm(self.v) > 5.5
            or np.any(np.isnan(self.R))
            or np.any(np.isnan(self.P))
        ):
            self.reset_kinematics_only()

    # ------------------------------------------------------------------
    #  Output smoothing (与原方案一致)
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
        # IMU noise
        sigma_gyro=0.01,
        sigma_accel=0.50,
        sigma_bg=0.0001,
        sigma_p=0.05,
        # observation noise
        pos_meas_noise_xy=0.08,
        pos_meas_noise_z=0.14,
        vel_meas_noise_xy=0.14,
        vel_meas_noise_z=0.22,
        # contact detection
        contact_force_on=18.0,
        contact_force_off=7.0,
        force_drop_thr=5.0,
        force_drop_weight=0.55,
        # safety clamps
        max_omega=8.0,
        max_accel=60.0,
        max_contact_residual=0.20,
        max_support_speed=3.00,
        max_bg=0.5,
        # contact weight smoothing
        vel_meas_alpha=0.88,
        weight_lp_alpha=0.80,
        touchdown_weight_scale=0.25,
        # horizontal bias (output side)
        horiz_bias_alpha=0.02,
        horiz_bias_gain=0.12,
        horiz_bias_clip=0.35,
        # output smoothing
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
        # attitude
        tilt_kp=0.12,
        tilt_ki=0.0,
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