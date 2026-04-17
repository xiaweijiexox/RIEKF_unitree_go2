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
import torch
import numpy as np

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


# ============================================================
# RIEKF 实现
# ============================================================
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# =====================================================================
# Real-time rolling-window plotter: EKF velocity (env 0)
# =====================================================================
class RealtimeVelocityPlotter:
    """Matplotlib real-time rolling-window plotter for vx/vy/vz CMD vs EKF (body frame)."""

    LABELS = ["vx", "vy", "vz"]
    COLORS_EKF = ["#ff7f0e", "#9467bd", "#8c564b"]
    COLORS_CMD = ["#000000", "#000000", "#000000"]

    def __init__(self, window_size: int = 500, dt: float = 0.02):
        self.window_size = window_size
        self.dt = dt
        self.ekf_bufs = [deque(maxlen=window_size) for _ in range(3)]
        self.cmd_bufs = [deque(maxlen=window_size) for _ in range(3)]
        self.t_buf = deque(maxlen=window_size)
        self.step_count = 0

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
            all_vals = np.concatenate([ekf_arr, cmd_arr])
            all_vals = all_vals[~np.isnan(all_vals)]
            if len(all_vals) > 0:
                ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
                margin = max(0.1, (ymax - ymin) * 0.15)
                ax.set_ylim(ymin - margin, ymax + margin)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def reset(self):
        for i in range(3):
            self.ekf_bufs[i].clear()
            self.cmd_bufs[i].clear()
        self.t_buf.clear()
        self.step_count = 0

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# =====================================================================
# Metric accumulator: GT vs EKF RMSE inside a fixed evaluation window
# =====================================================================
class VelocityMetricEvaluator:
    def __init__(self, dt: float, start_delay: float = 3.0, duration: float = 10.0,
                 save_path: str = "velocity_metrics_4foots_bias.txt"):
        self.dt = float(dt)
        self.start_delay = float(start_delay)
        self.duration = float(duration)
        self.save_path = save_path
        self.gt_list = []
        self.ekf_list = []
        self.step_count = 0
        self.finalized = False

    def reset(self):
        self.gt_list.clear()
        self.ekf_list.clear()
        self.step_count = 0
        self.finalized = False
        print("[METRIC] Evaluator reset.")

    def update(self, gt_vel: np.ndarray, ekf_vel: np.ndarray):
        if self.finalized:
            return
        t = self.step_count * self.dt
        self.step_count += 1
        if t < self.start_delay:
            return
        if t < self.start_delay + self.duration:
            self.gt_list.append(np.asarray(gt_vel, dtype=np.float64).copy())
            self.ekf_list.append(np.asarray(ekf_vel, dtype=np.float64).copy())
            return
        self.finalize()

    def finalize(self):
        if self.finalized:
            return
        self.finalized = True
        if len(self.gt_list) == 0:
            print("[METRIC] No samples — window never reached.")
            return
        gt = np.stack(self.gt_list, axis=0)
        ekf = np.stack(self.ekf_list, axis=0)
        err = ekf - gt
        N = gt.shape[0]
        rmse_xyz = np.sqrt(np.mean(err ** 2, axis=0))
        rmse_total = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        mae_xyz = np.mean(np.abs(err), axis=0)
        max_err_xyz = np.max(np.abs(err), axis=0)
        std_err_xyz = np.std(err, axis=0)
        gt_rms = np.sqrt(np.mean(gt ** 2, axis=0))
        rel_rmse_xyz = rmse_xyz / np.maximum(gt_rms, 1e-6)

        msg = (
            f"===================== VELOCITY METRICS =====================\n"
            f"Window: [{self.start_delay:.1f}s, {self.start_delay + self.duration:.1f}s]"
            f"  |  samples: {N}  |  dt: {self.dt:.4f}s\n"
            f"------------------------------------------------------------\n"
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
        try:
            with open(self.save_path, "w") as f:
                f.write(msg)
            print(f"[METRIC] >>> window closed, saved to {self.save_path} <<<")
        except Exception as e:
            print(f"[METRIC] Failed to save: {e}")


def dones_to_numpy(dones):
    if isinstance(dones, torch.Tensor):
        return dones.detach().cpu().numpy().astype(bool)
    return np.asarray(dones).astype(bool)


# ============================================================
# IMU + 四足足端接触 RIEKF
# ============================================================
class RIEKF:
    """
    右不变扩展卡尔曼滤波器，用于Go2真机/仿真状态估计
    状态 X_t in SE_3(3): [R_t, v_t, p_t, d_t]
    预测输入: omega_tilde (陀螺仪), a_tilde (加速度计)
    更新输入: h_p_body (接触足在机体系位置)
    输出: R_hat -> g_proj = R_hat^T * [0,0,-1]^T
    """
    def __init__(self, dt, sigma_g=0.01, sigma_a=0.1, sigma_v=0.01, sigma_alpha=0.001,
                 sigma_bg=1e-5, sigma_ba=1e-4):
        self.dt = dt
        self.g_vec = np.array([0.0, 0.0, -9.81])

        # 各向异性过程噪声:z 方向放大 3 倍(MEMS IMU z 轴本身更差 + 重力耦合)
        self.Sg  = sigma_g     * np.eye(3)
        self.Sa  = sigma_a     * np.diag([1.0, 1.0, 3.0])
        self.Sv  = sigma_v     * np.diag([1.0, 1.0, 3.0])
        self.Sal = sigma_alpha * np.eye(3)
        # bias random walk (very small — biases are nearly constant)
        self.Sbg = sigma_bg * np.eye(3)
        self.Sba = sigma_ba * np.eye(3)
        self.reset()

    def reset(self):
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.d = np.zeros((4, 3))           # 4 个足端锚点(世界系)
        self.bg = np.zeros(3)               # gyro bias (body frame)
        self.ba = np.zeros(3)               # accel bias (body frame)
        self.prev_contact = np.zeros(4, dtype=bool)
        # 状态维度 27: [phi(3), v(3), p(3), d0(3), d1(3), d2(3), d3(3), bg(3), ba(3)]
        #  索引: phi=0:3  v=3:6  p=6:9  d0=9:12  d1=12:15  d2=15:18  d3=18:21
        #        bg=21:24  ba=24:27
        # bias 初始方差小(假设 IMU 已近似校准),由接触观测慢慢在线估计
        P_diag = np.concatenate([
            np.ones(21) * 0.1,    # phi, v, p, d0..d3
            np.ones(3)  * 1e-4,   # bg
            np.ones(3)  * 1e-3,   # ba
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
        """27x27 Adjoint for 4-foot RIEKF with IMU biases.
        State layout: [phi, v, p, d0, d1, d2, d3, bg, ba].
        - Kinematic blocks (0..21) follow Barrau-Bonnabel Adjoint for SE_2+(3) + anchors
        - Bias blocks (21..27) are identity (biases are trivial group elements)
        """
        R = self.R
        v, p = self.v, self.p
        M = np.zeros((27, 27))
        # Diagonal R blocks for phi, v, p, d0..d3
        M[0:3,   0:3]   = R
        M[3:6,   3:6]   = R
        M[6:9,   6:9]   = R
        M[9:12,  9:12]  = R
        M[12:15, 12:15] = R
        M[15:18, 15:18] = R
        M[18:21, 18:21] = R
        # Off-diagonal coupling: skew(v)@R, skew(p)@R, skew(d_k)@R in first column block
        M[3:6, 0:3] = self.skew(v) @ R
        M[6:9, 0:3] = self.skew(p) @ R
        for k in range(4):
            M[9 + 3*k:12 + 3*k, 0:3] = self.skew(self.d[k]) @ R
        # Bias blocks: identity (no Adjoint coupling)
        M[21:24, 21:24] = np.eye(3)
        M[24:27, 24:27] = np.eye(3)
        return M

    def _A(self):
        """27x27 continuous-time linearized dynamics with IMU biases.
        Non-zero blocks:
          - v <- phi (gravity):     skew(g)
          - p <- v:                 I
          - phi <- bg (gyro bias):  -R   (key: makes bg observable)
          - v   <- ba (accel bias): -R   (key: makes ba observable)
        Anchors d_k and biases have zero self-dynamics (random walks in Q).
        """
        I = np.eye(3)
        R = self.R
        A = np.zeros((27, 27))
        A[3:6, 0:3] = self.skew(self.g_vec)   # phi -> v via gravity
        A[6:9, 3:6] = I                        # v   -> p
        # Bias coupling — this is what lets the filter learn bg / ba
        A[0:3, 21:24] = -R                     # phi <- bg
        A[3:6, 24:27] = -R                     # v   <- ba
        return A

    def predict(self, omega_tilde, a_tilde):
        """
        预测步
        omega_tilde: 陀螺仪读数 (含 bias), shape (3,)
        a_tilde:     加速度计读数(含 +g, 含 bias), shape (3,)
        """
        # IMU 去偏:使用 bias 估计值修正原始读数
        omega_corr = omega_tilde - self.bg
        a_corr     = a_tilde     - self.ba
        self.omega_tilde = omega_corr  # update 中 z 速度约束可能会用到(这里已弃用)

        dt = self.dt
        R, v, p = self.R, self.v, self.p

        # 世界系加速度 (含 g 修正)
        a_w = R @ a_corr + self.g_vec

        # 正确的二阶积分:位置必须包含 0.5·a·dt²
        self.R = R @ self.exp_so3(omega_corr * dt)
        self.v = v + a_w * dt
        self.p = p + v * dt + 0.5 * a_w * dt * dt

        # 27 维过程噪声矩阵:phi, v, p, d0..d3, bg, ba
        Cov_w = np.zeros((27, 27))
        Cov_w[0:3,   0:3]   = self.Sg      # gyro white noise
        Cov_w[3:6,   3:6]   = self.Sa      # accel white noise (已各向异性)
        Cov_w[6:9,   6:9]   = self.Sv      # p random walk
        for k in range(4):
            Cov_w[9 + 3*k:12 + 3*k, 9 + 3*k:12 + 3*k] = self.Sal
        Cov_w[21:24, 21:24] = self.Sbg     # gyro bias random walk
        Cov_w[24:27, 24:27] = self.Sba     # accel bias random walk

        A  = self._A()    # 27x27
        Ad = self._Ad()   # 27x27

        if np.any(np.isnan(Ad)) or np.any(np.isinf(Ad)) or np.max(np.abs(Ad)) > 1e4:
            print("🚨 Ad 矩阵异常, 最大值:", np.max(np.abs(Ad)))
            print(f"   v={np.max(np.abs(self.v)):.2e}  p={np.max(np.abs(self.p)):.2e}  "
                  f"d={np.max(np.abs(self.d)):.2e}  bg={self.bg}  ba={self.ba}")

        Q_bar = Ad @ Cov_w @ Ad.T
        F = np.eye(27) + A * dt
        self.P = F @ self.P @ F.T + Q_bar * dt

        # 状态发散检测,自动软重置(保留姿态和 bias,只清动力学状态)
        if np.any(np.isnan(self.v)) or np.any(np.isinf(self.v)) or np.max(np.abs(self.v)) > 50.0:
            self.v = np.zeros(3)
            self.p = np.zeros(3)
            self.d = np.zeros((4, 3))
            # 注意:不清 bg/ba,已经学到的东西要保留
            P_diag = np.concatenate([
                np.ones(21) * 0.1,
                np.ones(3)  * 1e-4,
                np.ones(3)  * 1e-3,
            ])
            self.P = np.diag(P_diag)
            self.prev_contact = np.zeros(4, dtype=bool)

    def update(self, h_p_body, contact, leg_idx):
        """
        位置观测:假设支撑足在世界系静止 → 机体位置 p 应该等于 锚点 d_leg - R@h_p_body
        每条腿观测**只修正自己的 d_leg**,不污染其他腿的锚点(原版 bug)。
        """
        # 新接触:重置该腿的锚点位置(不触发 update)
        if contact and not self.prev_contact[leg_idx]:
            self.d[leg_idx] = self.p + self.R @ h_p_body
        self.prev_contact[leg_idx] = contact
        if not contact:
            return

        R, p = self.R, self.p

        # ---- 位置约束: innovation = R@h_p_body - (d_leg - p) ----
        innovation_p = (R @ h_p_body) - (self.d[leg_idx] - p)

        # H 在 27 维状态上:只碰 p (列 6:9) 和对应的 d_leg (列 9+3*leg_idx : 12+3*leg_idx)
        # bias 列 (21:27) 全零 — bg/ba 的更新通过 P 里 phi/v 与 bias 的交叉协方差间接传播
        H_p = np.zeros((3, 27))
        H_p[:, 6:9] = -np.eye(3)
        d_start = 9 + 3 * leg_idx
        H_p[:, d_start:d_start + 3] = np.eye(3)

        # 观测噪声:z 方向放大 5 倍(落地冲击时足底垂直窜动更严重)
        N_p = R @ (self.Sal * np.diag([1.0, 1.0, 5.0])) @ R.T

        S = H_p @ self.P @ H_p.T + N_p
        K = self.P @ H_p.T @ np.linalg.inv(S)
        delta_xi = K @ innovation_p   # (27,)

        # 应用修正
        self.R = self.exp_so3(delta_xi[0:3]) @ self.R
        self.v += delta_xi[3:6]
        self.p += delta_xi[6:9]
        for k in range(4):
            self.d[k] += delta_xi[9 + 3*k : 12 + 3*k]
        # 关键:把修正量应用到 bias 估计上
        self.bg += delta_xi[21:24]
        self.ba += delta_xi[24:27]
        # bias 安全裁剪(防止数值跑飞,Go2 真机 bias 正常量级 <0.1)
        self.bg = np.clip(self.bg, -0.02, 0.02)
        self.ba = np.clip(self.ba, -0.02, 0.02)

        # Joseph 形式协方差更新(数值稳定)
        I_KH = np.eye(27) - K @ H_p
        self.P = I_KH @ self.P @ I_KH.T + K @ N_p @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(27) * 1e-8

        # NOTE: 原来的 "速度约束 v = -ω × r_foot" 被移除。
        # 原因:该约束的 z 分量几乎总是 ≈ 0 (ω 主要是 yaw/pitch,r_foot 主要向下),
        # 等于在告诉滤波器 "vz ≈ 0",这会把跳跃/落地时真实的 vz 强行压平 —— 正是你看到
        # "vz 跟不上" 的根本原因。四足接触位置观测本身已经能很好地约束水平速度,
        # 不需要这个有害的速度约束。

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

    def step(self, omega_tilde, a_tilde, h_p_body_list, contact_list):
        for i, f in enumerate(self.filters):
            f.predict(omega_tilde[i], a_tilde[i])
            for leg_idx, (h_p_body, contact) in enumerate(zip(h_p_body_list, contact_list)):
                f.update(h_p_body[i], contact[i], leg_idx)

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
    # foot_name       = "FL_foot"   # 单足模式，优先左前足
    # foot_body_id    = robot.body_names.index(foot_name)
    # foot_contact_id = cf_sensor.body_names.index(foot_name)

    foot_names       = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_body_ids    = [robot.body_names.index(n) for n in foot_names]
    foot_contact_ids = [cf_sensor.body_names.index(n) for n in foot_names]

    print(f"foot_body_ids: {foot_body_ids}")
    print(f"foot_contact_ids: {foot_contact_ids}")

    # 初始化 BatchRIEKF
    # dt = decimation * sim.dt = 4 * 0.005 = 0.02s
    riekf = BatchRIEKF(
        num_envs    = raw_env.num_envs,
        dt          = dt,
        sigma_g     = 0.01,
        sigma_a     = 0.1,
        sigma_v     = 0.01,
        sigma_alpha = 0.001,
        sigma_bg    = 1e-5,   # gyro bias random walk(仿真 IMU 干净, 取小值)
        sigma_ba    = 1e-4,   # accel bias random walk
    )

    # ---- Real-time plotter + metric evaluator ----
    plotter = RealtimeVelocityPlotter(
        window_size=args_cli.plot_window if hasattr(args_cli, "plot_window") else 500,
        dt=dt,
    )
    print(f"[INFO] Real-time plotter initialized")

    metric_eval = VelocityMetricEvaluator(
        dt=dt,
        start_delay=3.0,
        duration=10.0,
        save_path="velocity_metrics_4foots_bias.txt",
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

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
# --- 新增：EKF 延时启动计数器 ---

    ekf_delay_counter = 0

    riekf_initialized = False

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
            # foot_pos_w  = robot.data.body_pos_w[:, foot_body_id, :].cpu().numpy()  # (N,3)

            # R_batch  = quats_to_R_batch(root_quat_w)                              # (N,3,3)
            # diff_w   = foot_pos_w - root_pos_w                                     # (N,3)
            # h_p_body = np.einsum('nij,nj->ni', R_batch.transpose(0,2,1), diff_w)  # (N,3)

            # # 接触判断：FL_foot 接触力范数 > 1N
            # contact_force_norm = cf_sensor.data.net_forces_w[
            #     :, foot_contact_id, :
            # ].norm(dim=-1).cpu().numpy()   # (N,)
            # contact = contact_force_norm > 1.0  # (N,) bool


            R_batch = quats_to_R_batch(root_quat_w)
            h_p_body_list = []
            contact_list  = []
            for bid, cid in zip(foot_body_ids, foot_contact_ids):
                fp_w   = robot.data.body_pos_w[:, bid, :].cpu().numpy()
                diff   = fp_w - root_pos_w
                h_p_b  = np.einsum('nij,nj->ni', R_batch.transpose(0,2,1), diff)
                h_p_body_list.append(h_p_b)
                cf_norm = cf_sensor.data.net_forces_w[:, cid, :].norm(dim=-1).cpu().numpy()
                # 20N 阈值:Go2 单腿静态载荷 ~37N,20N 能过滤落地冲击瞬间的假接触和
                # 离地前的力衰减段,避免在脏接触窗口刷新锚点污染 z 估计
                contact_list.append(cf_norm > 20.0)


            
            max_sensor_val = 20.0
            omega_tilde = np.clip(omega_tilde, -max_sensor_val, max_sensor_val)
            a_tilde = np.clip(a_tilde, -max_sensor_val, max_sensor_val)

            # 异常静默安检
            if np.isnan(omega_tilde).any() or np.isnan(a_tilde).any():
                print("🚨 sensor NaN detected — aborting")
                import sys; sys.exit(1)

            # RIEKF 预测 + 更新
            ekf_delay_counter += 1

            if not riekf_initialized:
                R_init = quats_to_R_batch(root_quat_w)
                for i, f in enumerate(riekf.filters):
                    f.R = R_init[i].copy()
                    f.prev_contact = np.zeros(4, dtype=bool)
                riekf_initialized = True

            imu_ok = np.nanmax(np.abs(a_tilde)) < 30.0
            if imu_ok:
                riekf.step(omega_tilde, a_tilde, h_p_body_list, contact_list)

            # 获取滤波后 obs,覆盖 Isaac Lab 原始 obs
            # obs 结构(PolicyCfg 定义顺序):
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
                # reset filters and re-init their orientation with fresh GT quat
                riekf.reset(done_ids.tolist())
                root_quat_after = robot.data.root_quat_w.cpu().numpy()
                R_after = quats_to_R_batch(root_quat_after)
                for i in done_ids.tolist():
                    riekf.filters[i].R = R_after[i].copy()
                print(f"[RIEKF] reset filters for env_ids={done_ids.tolist()}")
                if 0 in done_ids:
                    plotter.reset()
                    metric_eval.reset()

            # ---- Real-time plot + metric update (env 0 only) ----
            # 把 GT/EKF 转 body frame,和 cmd (本来就是 body frame) 对齐。
            R_body0 = quats_to_R_batch(root_quat_w[0:1])[0]   # (3,3) env 0 的姿态
            ekf_vel_w = riekf.filters[0].v
            ekf_vel_b = R_body0.T @ ekf_vel_w

            # 取 base_velocity 指令(body frame),失败回退到 obs[0, 6:9]
            cmd_vel_b = cmd_fixed.copy()

            plotter.update(np.full(3, np.nan), ekf_vel_b, cmd_vel_b)
            metric_eval.update(cmd_vel_b, ekf_vel_b)

            # terminal log
            bg_dbg = riekf.filters[0].bg
            ba_dbg = riekf.filters[0].ba
            cmd_str = (f"CMD[{cmd_vel_b[0]:+.2f} {cmd_vel_b[1]:+.2f} {cmd_vel_b[2]:+.2f}]"
                       if cmd_vel_b is not None else "CMD[n/a]")
            print(
                f"[step {ekf_delay_counter:4d}] "
                f"{cmd_str} | "
                f"EKF[{ekf_vel_b[0]:+.2f} {ekf_vel_b[1]:+.2f} {ekf_vel_b[2]:+.2f}] | "
                f"bg={bg_dbg.round(4)} ba={ba_dbg.round(3)}"
            )

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

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