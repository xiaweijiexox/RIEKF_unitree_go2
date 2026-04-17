# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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

import gymnasium as gym
import os
import time
from collections import deque
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # interactive backend; change to "Qt5Agg" if TkAgg unavailable
import matplotlib.pyplot as plt

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
# Real-time rolling-window plotter: EKF velocity (env 0)
# =====================================================================
class RealtimeVelocityPlotter:
    """Matplotlib real-time rolling-window plotter: CMD vs EKF (body frame)."""

    LABELS = ["vx", "vy", "vz"]
    COLORS_EKF = ["#ff7f0e", "#9467bd", "#8c564b"]  # orange, purple, brown
    COLORS_CMD = ["#000000", "#000000", "#000000"]  # black (dotted) for command

    def __init__(self, window_size: int = 500, dt: float = 0.02):
        self.window_size = window_size
        self.dt = dt

        # ring-buffers
        self.ekf_bufs = [deque(maxlen=window_size) for _ in range(3)]
        self.cmd_bufs = [deque(maxlen=window_size) for _ in range(3)]
        self.t_buf = deque(maxlen=window_size)
        self.step_count = 0

        # --- set up figure ---
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        self.fig.suptitle("CMD vs EKF Velocity (env 0, body frame)", fontsize=13)

        self.lines_ekf = []
        self.lines_cmd = []
        for i, ax in enumerate(self.axes):
            (ln_cmd,) = ax.plot([], [], color=self.COLORS_CMD[i], linewidth=1.5,
                                linestyle=":", label=f"CMD {self.LABELS[i]}")
            (ln_ekf,) = ax.plot([], [], color=self.COLORS_EKF[i], linewidth=1.2,
                                linestyle="--", label=f"EKF {self.LABELS[i]}")
            self.lines_cmd.append(ln_cmd)
            self.lines_ekf.append(ln_ekf)
            ax.set_ylabel(f"{self.LABELS[i]} (m/s)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2.0, 2.0)

        self.axes[-1].set_xlabel("time (s)")
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, gt_vel: np.ndarray, ekf_vel: np.ndarray, cmd_vel: np.ndarray = None):
        """Push one timestep of data and refresh the plot.

        Args:
            gt_vel:  (3,) ground-truth velocity [vx, vy, vz] (body frame recommended)
            ekf_vel: (3,) EKF estimated velocity [vx, vy, vz] (body frame recommended)
            cmd_vel: (3,) commanded velocity [vx, vy, vz] (body frame)
        """
        t = self.step_count * self.dt
        self.t_buf.append(t)
        for i in range(3):
            self.ekf_bufs[i].append(float(ekf_vel[i]))
        if cmd_vel is not None:
            for i in range(3):
                self.cmd_bufs[i].append(float(cmd_vel[i]))
        else:
            for i in range(3):
                self.cmd_bufs[i].append(np.nan)
        self.step_count += 1

        t_arr = np.array(self.t_buf)

        for i, ax in enumerate(self.axes):
            ekf_arr = np.array(self.ekf_bufs[i])
            cmd_arr = np.array(self.cmd_bufs[i])

            self.lines_ekf[i].set_data(t_arr, ekf_arr)
            self.lines_cmd[i].set_data(t_arr, cmd_arr)

            ax.set_xlim(t_arr[0], t_arr[-1] + self.dt)

            # auto-scale y with a little margin, include cmd so steps are visible
            all_vals = np.concatenate([ekf_arr, cmd_arr])
            all_vals = all_vals[~np.isnan(all_vals)]
            if len(all_vals) > 0:
                ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
                margin = max(0.1, (ymax - ymin) * 0.15)
                ax.set_ylim(ymin - margin, ymax + margin)

        # efficient redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def reset(self):
        """Clear buffers on episode reset (optional)."""
        for i in range(3):
            self.ekf_bufs[i].clear()
            self.cmd_bufs[i].clear()
        self.t_buf.clear()
        self.step_count = 0

    def close(self):
        plt.ioff()
        plt.close(self.fig)

# =====================================================================
# Metric accumulator: evaluates EKF velocity estimates inside a fixed
# time window [start_delay, start_delay + duration] after motion starts.
# All time-series quantities are averaged to produce scalar metrics.
# =====================================================================
class VelocityMetricEvaluator:
    """Accumulates GT / EKF velocity samples inside an evaluation window
    and computes scalar metrics (RMSE, MAE, max err, err std, relative RMSE).

    Window definition:
        - t = 0 corresponds to the first step after motion starts (or last reset).
        - Samples are recorded only when start_delay <= t < start_delay + duration.
        - When the window closes, metrics are computed once and saved to file.
    """

    def __init__(self, dt: float, start_delay: float = 3.0, duration: float = 10.0,
                 save_path: str = "velocity_metrics_bias.txt"):
        self.dt = float(dt)
        self.start_delay = float(start_delay)
        self.duration = float(duration)
        self.save_path = save_path

        self.gt_list = []    # list of (3,) arrays
        self.ekf_list = []   # list of (3,) arrays
        self.t_list = []     # list of floats (relative time in seconds)

        self.step_count = 0
        self.finalized = False

    def reset(self):
        """Called on env reset — restart the window from scratch."""
        self.gt_list.clear()
        self.ekf_list.clear()
        self.t_list.clear()
        self.step_count = 0
        self.finalized = False
        print("[METRIC] Evaluator reset.")

    def update(self, gt_vel: np.ndarray, ekf_vel: np.ndarray):
        """Push one sample. Only records if inside the evaluation window."""
        if self.finalized:
            return

        t = self.step_count * self.dt
        self.step_count += 1

        # not yet inside window
        if t < self.start_delay:
            return

        # inside window
        if t < self.start_delay + self.duration:
            self.gt_list.append(np.asarray(gt_vel, dtype=np.float64).copy())
            self.ekf_list.append(np.asarray(ekf_vel, dtype=np.float64).copy())
            self.t_list.append(t)
            return

        # just passed the end of the window — finalize once
        self.finalize()

    def finalize(self):
        if self.finalized:
            return
        self.finalized = True

        if len(self.gt_list) == 0:
            print("[METRIC] No samples collected — evaluation window never reached.")
            return

        gt = np.stack(self.gt_list, axis=0)     # (N, 3)
        ekf = np.stack(self.ekf_list, axis=0)   # (N, 3)
        err = ekf - gt                           # (N, 3)
        N = gt.shape[0]

        rmse_xyz = np.sqrt(np.mean(err ** 2, axis=0))
        rmse_total = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        mae_xyz = np.mean(np.abs(err), axis=0)
        max_err_xyz = np.max(np.abs(err), axis=0)
        std_err_xyz = np.std(err, axis=0)
        gt_rms = np.sqrt(np.mean(gt ** 2, axis=0))
        rel_rmse_xyz = rmse_xyz / np.maximum(gt_rms, 1e-6)

        header = (
            f"===================== VELOCITY METRICS =====================\n"
            f"Window: [{self.start_delay:.1f}s, {self.start_delay + self.duration:.1f}s]"
            f"  |  samples: {N}  |  dt: {self.dt:.4f}s\n"
            f"------------------------------------------------------------\n"
        )
        body = (
            f"                         vx          vy          vz\n"
            f"RMSE          (m/s) : {rmse_xyz[0]:10.4f}  {rmse_xyz[1]:10.4f}  {rmse_xyz[2]:10.4f}\n"
            f"MAE           (m/s) : {mae_xyz[0]:10.4f}  {mae_xyz[1]:10.4f}  {mae_xyz[2]:10.4f}\n"
            f"Max |err|     (m/s) : {max_err_xyz[0]:10.4f}  {max_err_xyz[1]:10.4f}  {max_err_xyz[2]:10.4f}\n"
            f"Err std       (m/s) : {std_err_xyz[0]:10.4f}  {std_err_xyz[1]:10.4f}  {std_err_xyz[2]:10.4f}\n"
            f"Relative RMSE (-)   : {rel_rmse_xyz[0]:10.4f}  {rel_rmse_xyz[1]:10.4f}  {rel_rmse_xyz[2]:10.4f}\n"
            f"------------------------------------------------------------\n"
            f"Total 3D RMSE (m/s) : {rmse_total:.4f}\n"
            f"============================================================\n"
        )
        msg = header + body

        try:
            with open(self.save_path, "w") as f:
                f.write(msg)
            print(f"[METRIC] >>> window closed, saved to {self.save_path} <<<")
        except Exception as e:
            print(f"[METRIC] Failed to save metrics: {e}")


# ============================================================
# RIEKF 实现
# ============================================================
class RIEKF:
    """
    右不变扩展卡尔曼滤波器,用于 Go2 真机/仿真状态估计 —— 带 IMU bias 估计的 18 维版本。
    状态 X_t in SE_2+(3) × R^6:
        kinematic: [R_t, v_t, p_t, d_t]       (phi, v, p, d 各 3 维)
        biases:    [bg_t, ba_t]               (gyro / accel bias)
    布局索引:
        phi=0:3   v=3:6   p=6:9   d=9:12   bg=12:15   ba=15:18
    预测输入: omega_tilde (陀螺仪), a_tilde (加速度计, 含 +g bias)
    更新输入: h_p_body (接触足在机体系位置)
    输出:     R_hat -> g_proj = R_hat^T * [0,0,-1]^T
    """
    def __init__(self, dt, sigma_g=0.01, sigma_a=0.1, sigma_v=0.01, sigma_alpha=0.001,
                 sigma_bg=1e-5, sigma_ba=1e-4,
                 contact_force_on=18.0, contact_force_off=7.0,
                 force_drop_thr=5.0, force_drop_weight=0.5,
                 touchdown_weight=0.25):
        self.dt = dt
        self.g_vec = np.array([0.0, 0.0, -9.81])

        self.Sg  = sigma_g     * np.eye(3)
        self.Sa  = sigma_a     * np.eye(3)
        self.Sv  = sigma_v     * np.eye(3)
        self.Sal = sigma_alpha * np.eye(3)
        # bias random walk (very small - biases are nearly constant)
        self.Sbg = sigma_bg * np.eye(3)
        self.Sba = sigma_ba * np.eye(3)

        # ---- 接触管理参数 ----
        # 这一层只影响观测噪声 N_tilde 的缩放,不碰状态空间,不动 P 的结构,
        # 不加任何 EKF 之外的"硬跳过"逻辑。所有"不信观测"的意图都合法地
        # 表达为"那一步 R 矩阵变大",滤波器通过 Kalman gain 自己决定听多少。
        self.contact_force_on  = float(contact_force_on)   # 双阈值上限:超过才认为触地
        self.contact_force_off = float(contact_force_off)  # 双阈值下限:低于才认为离地
        self.force_drop_thr    = float(force_drop_thr)     # 力下降超过此值判定为"正在离地"
        self.force_drop_weight = float(force_drop_weight)  # 力下降期间的权重乘子
        self.touchdown_weight  = float(touchdown_weight)   # 落地瞬间的权重乘子
        self.reset()

    def reset(self):
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.d = np.zeros(3)
        self.bg = np.zeros(3)   # gyro bias (body frame)
        self.ba = np.zeros(3)   # accel bias (body frame)
        # 接触管理状态
        self.contact_mode = False       # 当前是否判定为接触(带滞回)
        self.prev_force   = float('nan')  # 上一步的力大小,用于算 delta_f
        # debug 字段(给 main log 用,可选)
        self.last_meas_weight = 0.0     # 上一步的观测权重
        self.last_delta_f     = 0.0
        self.last_force_decreasing = False
        # 18 维 P: phi, v, p, d 各用原来的 0.1, bias 用更小的初始方差
        P_diag = np.concatenate([
            np.ones(12) * 0.1,   # phi, v, p, d
            np.ones(3)  * 1e-4,  # bg
            np.ones(3)  * 1e-3,  # ba
        ])
        self.P = np.diag(P_diag)

    @staticmethod
    def skew(v):
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])

    @staticmethod
    def exp_so3(phi):
        angle = np.linalg.norm(phi)
        if angle < 1e-8:
            return np.eye(3) + RIEKF.skew(phi)
        K = RIEKF.skew(phi / angle)
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def _Ad(self):
        """18x18 Adjoint. Kinematic 12x12 block is original RIEKF Adjoint;
        bias 6x6 block is identity (biases are trivial group elements).
        """
        R, v, p, d = self.R, self.v, self.p, self.d
        Z = np.zeros((3, 3))
        M = np.zeros((18, 18))
        # kinematic 12x12 block (identical to original)
        M[0:3,  0:3]  = R
        M[3:6,  3:6]  = R
        M[3:6,  0:3]  = self.skew(v) @ R
        M[6:9,  6:9]  = R
        M[6:9,  0:3]  = self.skew(p) @ R
        M[9:12, 9:12] = R
        M[9:12, 0:3]  = self.skew(d) @ R
        # bias blocks: identity
        M[12:15, 12:15] = np.eye(3)
        M[15:18, 15:18] = np.eye(3)
        return M

    def _A(self):
        """18x18 continuous-time linearized dynamics with IMU biases.
        Non-zero blocks:
          - v <- phi (gravity):     skew(g)
          - p <- v:                 I
          - phi <- bg (gyro bias):  -R   (key: makes bg observable)
          - v   <- ba (accel bias): -R   (key: makes ba observable)
        """
        I = np.eye(3)
        R = self.R
        A = np.zeros((18, 18))
        A[3:6, 0:3] = self.skew(self.g_vec)   # phi -> v via gravity
        A[6:9, 3:6] = I                        # v   -> p
        A[0:3, 12:15] = -R                     # phi <- bg
        A[3:6, 15:18] = -R                     # v   <- ba
        return A

    def predict(self, omega_tilde, a_tilde):
        """
        预测步
        omega_tilde: 陀螺仪读数 (含 bias), shape (3,)
        a_tilde:     加速度计读数 (含 +g, 含 bias, 静止时 ≈ [0,0,+9.81]), shape (3,)
        """
        # IMU 去偏
        omega_corr = omega_tilde - self.bg
        a_corr     = a_tilde     - self.ba

        dt = self.dt
        R, v, p = self.R, self.v, self.p

        # 世界系加速度
        a_w = R @ a_corr + self.g_vec

        # 原文件的积分形式(保持一致):v 一阶,p 只用 v*dt
        self.R = R @ self.exp_so3(omega_corr * dt)
        self.v = v + a_w * dt
        self.p = p + v * dt

        # 18 维 Cov_w
        Cov_w = np.zeros((18, 18))
        Cov_w[0:3,   0:3]   = self.Sg    # gyro white noise
        Cov_w[3:6,   3:6]   = self.Sa    # accel white noise
        Cov_w[9:12,  9:12]  = self.Sv    # anchor random walk
        Cov_w[12:15, 12:15] = self.Sbg   # gyro bias random walk
        Cov_w[15:18, 15:18] = self.Sba   # accel bias random walk

        A  = self._A()    # 18x18
        Ad = self._Ad()   # 18x18

        if np.any(np.isnan(Ad)) or np.any(np.isinf(Ad)) or np.max(np.abs(Ad)) > 1e4:
            print("🚨 Ad 矩阵异常, 最大值:", np.max(np.abs(Ad)))
            print(f"   R_max={np.max(np.abs(self.R)):.2e}  v_max={np.max(np.abs(self.v)):.2e}  "
                  f"p_max={np.max(np.abs(self.p)):.2e}  d_max={np.max(np.abs(self.d)):.2e}")
            print(f"   bg={self.bg}  ba={self.ba}")

        if np.any(np.isnan(Cov_w)) or np.any(np.isinf(Cov_w)):
            print("🚨 Cov_w 矩阵异常")

        Q_bar = Ad @ Cov_w @ Ad.T
        # 连续时间更新,和原文件一致
        self.P = self.P + (A @ self.P + self.P @ A.T + Q_bar) * dt

    def _contact_weight(self, force_norm):
        """接触管理层:根据足底力大小算出观测权重 w ∈ [0, 1]。
        这是 EKF 数学之外的独立逻辑,只用来动态缩放观测噪声 N_tilde / w —— 
        w=1 时 R 等于设计值(默认最信),w<1 时 R 放大(不信),w≈0 时 R 趋向无穷(等价于跳过)。
        状态空间不变,bias 估计路径不变,严格符合时变观测噪声的 EKF 数学。

        规则:
          1. 双阈值滞回:>on 才转接触,<off 才转摆动 → 防抖动
          2. 力下降窗口:delta_f < -thr 时观测权重 × force_drop_weight → 处理离地尾段
          3. 落地过渡:刚从摆动切接触时 × touchdown_weight → 处理冲击瞬间
        """
        # delta_f 计算
        if np.isnan(self.prev_force):
            delta_f = 0.0
            is_decreasing = False
        else:
            delta_f = float(force_norm - self.prev_force)
            is_decreasing = delta_f < -self.force_drop_thr

        # 双阈值滞回更新 contact_mode
        was_contact = self.contact_mode
        if was_contact:
            contact_now = force_norm > self.contact_force_off
        else:
            contact_now = force_norm > self.contact_force_on
        touchdown = contact_now and (not was_contact)

        # 力-权重线性插值:force_off 对应 0, force_on 对应 1
        if contact_now:
            w = (force_norm - self.contact_force_off) / max(
                self.contact_force_on - self.contact_force_off, 1e-6)
            w = float(np.clip(w, 0.0, 1.0))
            if is_decreasing:
                w *= self.force_drop_weight
            if touchdown:
                w *= self.touchdown_weight
        else:
            w = 0.0

        # 更新接触状态
        self.contact_mode = contact_now
        self.prev_force = float(force_norm)
        self.last_delta_f = delta_f
        self.last_force_decreasing = is_decreasing
        self.last_meas_weight = w
        return w

    def update(self, h_p_body, force_norm):
        """位置观测更新。
        force_norm: 足底力的 2-范数 (float, 单位 N)。接触管理层会根据它算出
                    观测权重 w,用于缩放 N_tilde。
        接口变更:原来传 bool contact,现在传 float force_norm。
        """
        w = self._contact_weight(float(force_norm))
        if w < 1e-4:
            return   # 权重几乎为零,等价于没有观测;数学上和 R=∞ 等价

        R, p, d = self.R, self.p, self.d

        # 位置观测 innovation(和原文件完全一致)
        innovation = (R @ h_p_body) - (d - p)

        # H 在 18 维上:p 列 6:9 取 -I,d 列 9:12 取 +I,bias 列全 0
        # bias 的更新通过 P 里 phi/v 与 bg/ba 的交叉协方差间接传播
        H = np.zeros((3, 18))
        H[:, 6:9]  = -np.eye(3)
        H[:, 9:12] =  np.eye(3)

        # 时变观测噪声:权重 w 越小,N_tilde 越大,滤波器越不信这步观测
        N_tilde = (R @ self.Sal @ R.T) / w

        S = H @ self.P @ H.T + N_tilde
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_xi = K @ innovation   # (18,)

        self.R = self.exp_so3(delta_xi[0:3]) @ self.R
        self.v = self.v + delta_xi[3:6]
        self.p = self.p + delta_xi[6:9]
        self.d = self.d + delta_xi[9:12]
        # 关键:应用 bias 修正
        self.bg = self.bg + delta_xi[12:15]
        self.ba = self.ba + delta_xi[15:18]
        # bias 安全裁剪(防止数值跑飞,Go2 真机 bias 正常 <0.1)
        self.bg = np.clip(self.bg, -0.02, 0.02)
        self.ba = np.clip(self.ba, -0.02, 0.02)

        I_KH = np.eye(18) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ N_tilde @ K.T

        # State Centering: RIEKF 只需要相对位置 (d - p),绝对位置无物理意义
        shift = self.p.copy()
        self.p -= shift
        self.d -= shift

    def get_g_proj(self):
        """重力投影: g_proj = R_hat^T * [0,0,-1]^T,对应 policy obs: projected_gravity"""
        return self.R.T @ np.array([0.0, 0.0, -1.0])

    def get_omega_hat(self, omega_tilde):
        """返回去偏后的 gyro,用于 policy obs"""
        return np.asarray(omega_tilde, dtype=np.float64) - self.bg


class BatchRIEKF:
    """N个env并行的RIEKF管理器"""
    def __init__(self, num_envs, dt, **kwargs):
        self.num_envs = num_envs
        self.filters  = [RIEKF(dt, **kwargs) for _ in range(num_envs)]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        for i in env_ids:
            self.filters[i].reset()

    def step(self, omega_tilde, a_tilde, h_p_body, force_norm):
        """
        omega_tilde: (N, 3) ndarray
        a_tilde:     (N, 3) ndarray
        h_p_body:    (N, 3) ndarray
        force_norm:  (N,)   float ndarray, 足底力大小 (N)
        """
        for i, f in enumerate(self.filters):
            f.predict(omega_tilde[i], a_tilde[i])
            f.update(h_p_body[i], force_norm[i])

    def get_obs(self, omega_tilde):
        """
        返回滤波后的两个 policy obs
        omega_hat: (N, 3) ndarray
        g_proj:    (N, 3) ndarray
        """
        omega_hat = np.stack([f.get_omega_hat(omega_tilde[i]) for i, f in enumerate(self.filters)])
        g_proj    = np.stack([f.get_g_proj() for f in self.filters])
        return omega_hat, g_proj
# ============================================================


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # 获取仿真底层资产句柄
    raw_env    = env.unwrapped
    robot      = raw_env.scene["robot"]
    cf_sensor  = raw_env.scene["contact_forces"]
    imu_sensor = raw_env.scene["imu_sensor"]   # ← IMU sensor（挂载在 base 上）

    # 打印body名称，确认足端名字（第一次运行后可注释掉）
    print(f"[RIEKF] robot.body_names:          {robot.body_names}")
    print(f"[RIEKF] contact_forces.body_names: {cf_sensor.body_names}")

    # Go2 足端 body 名称，根据上面打印结果确认后修改
    foot_name       = "FL_foot"   # 单足模式，优先左前足
    foot_body_id    = robot.body_names.index(foot_name)
    foot_contact_id = cf_sensor.body_names.index(foot_name)

    # 初始化 BatchRIEKF
    # dt = decimation * sim.dt = 4 * 0.005 = 0.02s
    riekf = BatchRIEKF(
        num_envs    = raw_env.num_envs,
        dt          = dt,
        sigma_g     = 0.01,
        sigma_a     = 0.1,
        sigma_v     = 0.01,
        sigma_alpha = 0.001,
    )

    # =====================================================================
    # Initialize real-time plotter + metric evaluator
    # =====================================================================
    plotter = RealtimeVelocityPlotter(window_size=args_cli.plot_window, dt=dt)
    print(f"[INFO] Real-time plotter initialized (window={args_cli.plot_window} steps)")

    metric_eval = VelocityMetricEvaluator(
        dt=dt,
        start_delay=3.0,
        duration=10.0,
        save_path="velocity_metrics_bias.txt",
    )
    print(f"[INFO] Metric evaluator initialized "
          f"(window [{metric_eval.start_delay}s, "
          f"{metric_eval.start_delay + metric_eval.duration}s])")

    def quats_to_R_batch(quats):
        """quats: (N,4) (w,x,y,z) -> R_batch: (N,3,3)"""
        w, x, y, z = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
        N = quats.shape[0]
        R = np.zeros((N, 3, 3))
        R[:,0,0] = 1 - 2*(y*y + z*z);  R[:,0,1] = 2*(x*y - w*z);  R[:,0,2] = 2*(x*z + w*y)
        R[:,1,0] = 2*(x*y + w*z);      R[:,1,1] = 1 - 2*(x*x+z*z); R[:,1,2] = 2*(y*z - w*x)
        R[:,2,0] = 2*(x*z - w*y);      R[:,2,1] = 2*(y*z + w*x);   R[:,2,2] = 1 - 2*(x*x+y*y)
        return R  # (N,3,3)

    def dones_to_numpy(dones):
        if isinstance(dones, torch.Tensor):
            return dones.detach().cpu().numpy().astype(bool)
        return np.asarray(dones).astype(bool)

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # --- 新增：EKF 延时启动计数器 ---
    ekf_delay_counter = 0
    riekf_initialized = False   # first-step attitude init from GT quat

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():

            # ----------------------------------------------------------
            # 读取 IMU sensor 数据（机体系）
            # imu_sensor.data.ang_vel_b: 角速度，对应陀螺仪读数
            # imu_sensor.data.lin_acc_b: 线加速度，含 gravity_bias=(0,0,9.81)
            #   即静止水平时输出 [0, 0, +9.81]，与真实 IMU 行为完全一致
            # ----------------------------------------------------------
            omega_tilde = imu_sensor.data.ang_vel_b.cpu().numpy()   # (N,3) 陀螺仪
            a_tilde     = imu_sensor.data.lin_acc_b.cpu().numpy()   # (N,3) 加速度计 ✅

            # 足端世界系位置 -> 转机体系（用于 RIEKF 观测更新）
            root_pos_w  = robot.data.root_pos_w.cpu().numpy()       # (N,3)
            root_quat_w = robot.data.root_quat_w.cpu().numpy()      # (N,4) (w,x,y,z)
            foot_pos_w  = robot.data.body_pos_w[:, foot_body_id, :].cpu().numpy()  # (N,3)

            R_batch  = quats_to_R_batch(root_quat_w)                              # (N,3,3)
            diff_w   = foot_pos_w - root_pos_w                                     # (N,3)
            h_p_body = np.einsum('nij,nj->ni', R_batch.transpose(0,2,1), diff_w)  # (N,3)

            # 接触判断：FL_foot 接触力范数 > 1N
            contact_force_norm = cf_sensor.data.net_forces_w[
                :, foot_contact_id, :
            ].norm(dim=-1).cpu().numpy()   # (N,) float,足底力大小
            # 不再做 bool 化,接触判断交给滤波器内部的接触管理层
            
            max_sensor_val = 50.0  # 你可以调整这个阈值
            omega_tilde = np.clip(omega_tilde, -max_sensor_val, max_sensor_val)
            a_tilde = np.clip(a_tilde, -max_sensor_val, max_sensor_val)

            # --- 安检代码开始 ---
            print("-------------")
            print(f"Omega 包含 NaN: {np.isnan(omega_tilde).any()}, 最大值: {np.nanmax(np.abs(omega_tilde)):.2f}")
            print(f"Accel 包含 NaN: {np.isnan(a_tilde).any()}, 最大值: {np.nanmax(np.abs(a_tilde)):.2f}")
            
            if np.isnan(omega_tilde).any() or np.isnan(a_tilde).any() or np.nanmax(np.abs(a_tilde)) > 500:
                print("🚨 抓到元凶：传感器输入了极其离谱的垃圾数据！(物理引擎已崩溃)")
                import sys; sys.exit()

            # RIEKF 预测 + 更新(每步直接执行,和 4foots.py 对齐,无延时启动)
            if not riekf_initialized:
                # Initialize filter attitude from GT quat (first step only).
                # Without this, R=I causes massive xy drift from gravity misalignment.
                for i, f_i in enumerate(riekf.filters):
                    f_i.R = R_batch[i].copy()
                riekf_initialized = True
            ekf_delay_counter += 1
            riekf.step(omega_tilde, a_tilde, h_p_body, contact_force_norm)

            # 获取滤波后 obs，覆盖 Isaac Lab 原始 obs
            # obs 结构（PolicyCfg 定义顺序）:
            # [base_ang_vel(0:3), projected_gravity(3:6), velocity_commands(6:9),
            #  joint_pos_rel(9:21), joint_vel_rel(21:33), last_action(33:45)]
            omega_hat, g_proj = riekf.get_obs(omega_tilde)   # (N,3), (N,3)
            obs[:, 0:3] = torch.from_numpy(omega_hat * 0.2).to(obs.device).float()  # scale=0.2
            obs[:, 3:6] = torch.from_numpy(g_proj).to(obs.device).float()

            # agent stepping

            # ---- Fixed CMD injection (同步六个算法的输入)----
            # 初始: vx=0.5, vy=-0.5
            # t=8s : vx -> -0.2  (vy 不变, 仍 -0.5)
            # t=9s : vy -> 0.2   (vx 不变, 仍 -0.2)
            # 时间用 ekf_delay_counter * dt (episode-local,每次 reset 归零)
            t_local = ekf_delay_counter * dt
            if t_local < 8.0:
                cmd_fixed = np.array([0.5, -0.5, 0.0])
            elif t_local < 9.0:
                cmd_fixed = np.array([-0.2, -0.5, 0.0])
            else:
                cmd_fixed = np.array([-0.2, 0.2, 0.0])
            # 覆盖 Isaac Lab 的 command_manager (首选路径 - 让 ObservationManager 下一帧自己写入 obs)
            try:
                _cmd_t = env.unwrapped.command_manager.get_command("base_velocity")
                _cmd_t[:, 0] = float(cmd_fixed[0])
                _cmd_t[:, 1] = float(cmd_fixed[1])
                if _cmd_t.shape[-1] >= 3:
                    _cmd_t[:, 2] = 0.0   # yaw_rate 置零
            except Exception:
                pass
            # 同时覆盖 obs[:, 6:9] (兜底 - 假设 scale=1.0)
            obs[:, 6:9] = torch.tensor(
                cmd_fixed, dtype=obs.dtype, device=obs.device
            ).unsqueeze(0).expand(obs.shape[0], -1)

            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # IMPORTANT: reset filter/plotter/metric on episode reset
            dones_np = dones_to_numpy(dones)
            done_ids = np.flatnonzero(dones_np)
            if len(done_ids) > 0:
                riekf.reset(done_ids.tolist())
                # Re-initialize attitude from fresh GT quat after episode reset
                root_quat_after_init = robot.data.root_quat_w.cpu().numpy()
                R_after_init = quats_to_R_batch(root_quat_after_init)
                for i in done_ids.tolist():
                    riekf.filters[i].R = R_after_init[i].copy()
                print(f"[RIEKF] reset filters for env_ids={done_ids.tolist()}")
                if 0 in done_ids:
                    plotter.reset()
                    metric_eval.reset()

            # ---- Real-time plot + metric update (env 0 only) ----
            # 统一坐标系:GT / EKF 转 body frame,和 cmd 对齐
            # 用 env.step() 之后的最新 quat
            root_quat_after = robot.data.root_quat_w.cpu().numpy()
            R_body0 = quats_to_R_batch(root_quat_after[0:1])[0]   # (3,3)
            ekf_vel_w = riekf.filters[0].v
            ekf_vel_b = R_body0.T @ ekf_vel_w

            # 取 base_velocity 指令(body frame),失败回退到 obs[0, 6:9]
            cmd_vel_b = cmd_fixed.copy()

            plotter.update(np.full(3, np.nan), ekf_vel_b, cmd_vel_b)
            # metric: body frame GT vs EKF (评估 EKF 估计质量)
            metric_eval.update(cmd_vel_b, ekf_vel_b)

            cmd_str = (f"CMD[{cmd_vel_b[0]:+.2f} {cmd_vel_b[1]:+.2f} {cmd_vel_b[2]:+.2f}]"
                       if cmd_vel_b is not None else "CMD[n/a]")
            print(
                f"[step {ekf_delay_counter:4d}] {cmd_str} | "
                f"EKF[{ekf_vel_b[0]:+.2f} {ekf_vel_b[1]:+.2f} {ekf_vel_b[2]:+.2f}]"
            )

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Clean up: finalize metrics and close plotter
    metric_eval.finalize()
    plotter.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()