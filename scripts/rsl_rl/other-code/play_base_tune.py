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
# Real-time rolling-window plotter: GT vs EKF velocity (env 0)
# =====================================================================
class RealtimeVelocityPlotter:
    """Matplotlib real-time rolling-window plotter for vx/vy/vz GT vs EKF.

    Uses interactive mode (`plt.ion()`) and partial redraw so the plot
    stays responsive even at high control frequencies.
    """

    LABELS = ["vx", "vy", "vz"]
    COLORS_GT = ["#1f77b4", "#2ca02c", "#d62728"]   # blue, green, red
    COLORS_EKF = ["#ff7f0e", "#9467bd", "#8c564b"]  # orange, purple, brown

    def __init__(self, window_size: int = 500, dt: float = 0.02, redraw_every: int = 10):
        self.window_size = window_size
        self.dt = dt
        self.redraw_every = int(redraw_every)  # 每 N 次 update 才真正重绘一次

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

        数据每步都 append 到环形缓冲,但实际的 matplotlib 重绘只在
        每 `redraw_every` 步执行一次 —— 重绘是 TkAgg 后端最慢的部分,
        节流后主循环从 ~50ms/step 降到 ~5ms/step,消除数据滞后。

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

        # 节流:未到刷新周期就直接返回
        if self.step_count % self.redraw_every != 0:
            return

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

        # efficient redraw
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
                 save_path: str = "velocity_metrics.txt"):
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
    右不变扩展卡尔曼滤波器，用于Go2真机/仿真状态估计
    状态 X_t in SE_3(3): [R_t, v_t, p_t, d_t]
    预测输入: omega_tilde (陀螺仪), a_tilde (加速度计)
    更新输入: h_p_body (接触足在机体系位置)
    输出: R_hat -> g_proj = R_hat^T * [0,0,-1]^T
    """
    def __init__(self, dt, sigma_g=0.005, sigma_a=0.5, sigma_v=0.001, sigma_alpha=0.05):
        # ─────────────────────────────────────────────────────────────
        # 参数调优说明(针对"水平快速运动下 EKF 相位滞后"):
        #   sigma_a   0.1  → 0.5   加速度过程噪声 ↑5x,让滤波器更信 IMU 积分
        #                         的高频成分,响应更快(主修复项)
        #   sigma_alpha 0.001→0.05  接触观测噪声 ↑50x,放低接触 FK 的权重,
        #                         降低它对速度高频的压制(主修复项)
        #   sigma_g   0.01 → 0.005 陀螺噪声 ↓,IMU 仿真里几乎无噪,姿态积分
        #                         更稳,间接减少 R·a_tilde 的投影误差
        #   sigma_v   0.01 → 0.001 足端锚点随机游走 ↓,锚点更"钉死",让 EKF
        #                         把短时误差归到速度通道而不是锚点漂移
        # ─────────────────────────────────────────────────────────────
        self.dt = dt
        self.g_vec = np.array([0.0, 0.0, -9.81])

        self.Sg  = sigma_g     * np.eye(3)
        self.Sa  = sigma_a     * np.eye(3)
        self.Sv  = sigma_v     * np.eye(3)
        self.Sal = sigma_alpha * np.eye(3)
        self.reset()

    def reset(self):
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.d = np.zeros(3)
        self.P = np.eye(12) * 0.1

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
        R, v, p, d = self.R, self.v, self.p, self.d
        Z = np.zeros((3, 3))
        return np.block([
            [R,                 Z, Z, Z],
            [self.skew(v) @ R, R, Z, Z],
            [self.skew(p) @ R, Z, R, Z],
            [self.skew(d) @ R, Z, Z, R],
        ])

    def _A(self):
        Z, I = np.zeros((3, 3)), np.eye(3)
        return np.block([
            [Z,                    Z, Z, Z],
            [self.skew(self.g_vec), Z, Z, Z],
            [Z,                    I, Z, Z],
            [Z,                    Z, Z, Z],
        ])

    def predict(self, omega_tilde, a_tilde):
        """
        预测步
        omega_tilde: 陀螺仪读数, shape (3,)
        a_tilde:     加速度计读数（含 +g bias，即静止时 ≈ [0,0,+9.81]）, shape (3,)
        """
        dt = self.dt
        R, v, p = self.R, self.v, self.p

        # 👉 修复核心：a_tilde 已经是比力 (Specific Force)，直接用！
        # 物理学公式: dv = (R * a_measured + g_world) * dt
        self.R = R @ self.exp_so3(omega_tilde * dt)
        self.v = v + (R @ a_tilde + self.g_vec) * dt
        self.p = p + v * dt

        Z = np.zeros((3, 3))
        Cov_w = np.block([
            [self.Sg, Z,       Z, Z      ],
            [Z,       self.Sa, Z, Z      ],
            [Z,       Z,       Z, Z      ],
            [Z,       Z,       Z, self.Sv],
        ])

        A     = self._A()
        Ad    = self._Ad()

        Q_bar = Ad @ Cov_w @ Ad.T
        self.P = self.P + (A @ self.P + self.P @ A.T + Q_bar) * dt

    def update(self, h_p_body, contact):
        if not contact:
            return

        R, p, d = self.R, self.p, self.d

        # 👉 修复核心：把误差转回世界坐标系，且必须是 [测量值 - 预期值]
        innovation = (R @ h_p_body) - (d - p)

        Z, I = np.zeros((3, 3)), np.eye(3)
        H = np.block([Z, Z, -I, I])

        N_tilde = R @ self.Sal @ R.T

        S = H @ self.P @ H.T + N_tilde
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_xi = K @ innovation

        self.R = self.exp_so3(delta_xi[0:3]) @ self.R
        self.v = self.v + delta_xi[3:6]
        self.p = self.p + delta_xi[6:9]
        self.d = self.d + delta_xi[9:12]

        I_KH   = np.eye(12) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ N_tilde @ K.T
        
        # 👉 终极护城河：状态归零（State Centering）
        # RIEKF 只需要相对位置 (d - p)，绝对位置是没有意义的。
        # 每步把 p 和 d 强行拉回原点附近，防止机器人走远后浮点数精度爆炸导致 Ad 矩阵崩溃！
        shift = self.p.copy()
        self.p -= shift
        self.d -= shift

    def get_g_proj(self):
        """重力投影: g_proj = R_hat^T * [0,0,-1]^T，对应 policy obs: projected_gravity"""
        return self.R.T @ np.array([0.0, 0.0, -1.0])

    def get_omega_hat(self, omega_tilde):
        return omega_tilde


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

    def step(self, omega_tilde, a_tilde, h_p_body, contact):
        """
        omega_tilde: (N, 3) ndarray
        a_tilde:     (N, 3) ndarray  ← 直接来自 imu_sensor.data.lin_acc_b
        h_p_body:    (N, 3) ndarray
        contact:     (N,)   bool ndarray
        """
        for i, f in enumerate(self.filters):
            f.predict(omega_tilde[i], a_tilde[i])
            f.update(h_p_body[i], contact[i])

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
    # 参数说明见 RIEKF.__init__ 注释
    riekf = BatchRIEKF(
        num_envs    = raw_env.num_envs,
        dt          = dt,
        sigma_g     = 0.005,   # was 0.01
        sigma_a     = 0.5,     # was 0.1   ← 放大过程噪声,减小相位滞后
        sigma_v     = 0.001,   # was 0.01
        sigma_alpha = 0.05,    # was 0.001 ← 放大观测噪声,减小相位滞后
    )

    # =====================================================================
    # Initialize real-time plotter + metric evaluator
    # =====================================================================
    plotter = RealtimeVelocityPlotter(window_size=args_cli.plot_window, dt=dt, redraw_every=10)
    print(f"[INFO] Real-time plotter initialized (window={args_cli.plot_window} steps, redraw every 10 steps)")

    metric_eval = VelocityMetricEvaluator(
        dt=dt,
        start_delay=3.0,
        duration=10.0,
        save_path="velocity_metrics.txt",
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
            ].norm(dim=-1).cpu().numpy()   # (N,)
            contact = contact_force_norm > 1.0  # (N,) bool
            
            max_sensor_val = 50.0  # 你可以调整这个阈值
            omega_tilde = np.clip(omega_tilde, -max_sensor_val, max_sensor_val)
            a_tilde = np.clip(a_tilde, -max_sensor_val, max_sensor_val)

            # --- 安检(静默):只在真崩溃时才报 ---
            if np.isnan(omega_tilde).any() or np.isnan(a_tilde).any() or np.nanmax(np.abs(a_tilde)) > 500:
                print(f"🚨 传感器崩溃: omega_nan={np.isnan(omega_tilde).any()}, "
                      f"accel_nan={np.isnan(a_tilde).any()}, "
                      f"accel_max={np.nanmax(np.abs(a_tilde)):.2f}")
                import sys; sys.exit()

            # RIEKF 预测 + 更新
            # riekf.step(omega_tilde, a_tilde, h_p_body, contact)

            # --- EKF 延时启动逻辑 ---
            ekf_delay_counter += 1

            # 前 100 步在空中和落地缓冲期让 EKF "闭关锁国",避免吸收冲击数据
            if ekf_delay_counter > 100:
                riekf.step(omega_tilde, a_tilde, h_p_body, contact)

            # 获取滤波后 obs，覆盖 Isaac Lab 原始 obs
            # obs 结构（PolicyCfg 定义顺序）:
            # [base_ang_vel(0:3), projected_gravity(3:6), velocity_commands(6:9),
            #  joint_pos_rel(9:21), joint_vel_rel(21:33), last_action(33:45)]
            omega_hat, g_proj = riekf.get_obs(omega_tilde)   # (N,3), (N,3)
            obs[:, 0:3] = torch.from_numpy(omega_hat * 0.2).to(obs.device).float()  # scale=0.2
            obs[:, 3:6] = torch.from_numpy(g_proj).to(obs.device).float()

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # IMPORTANT: reset filter/plotter/metric on episode reset
            dones_np = dones_to_numpy(dones)
            done_ids = np.flatnonzero(dones_np)
            if len(done_ids) > 0:
                riekf.reset(done_ids.tolist())
                print(f"[RIEKF] reset filters for env_ids={done_ids.tolist()}")
                if 0 in done_ids:
                    plotter.reset()
                    metric_eval.reset()

            # ---- Real-time plot + metric update (env 0 only) ----
            gt_vel  = robot.data.root_lin_vel_w[0].cpu().numpy()   # ground truth
            ekf_vel = riekf.filters[0].v                            # RIEKF 估计
            plotter.update(gt_vel, ekf_vel)
            metric_eval.update(gt_vel, ekf_vel)

            # --- wall-clock 探针:每 100 步报一次实际循环周期,用来验证滞后是否消除 ---
            # 正常情况下 wall_dt 应接近 env.step_dt (默认 0.02s = 20ms)
            # 如果 wall_dt 明显大于 dt,说明循环被 plot/print/其他 IO 拖慢,
            # EKF 会用错误的 dt 积分导致整体时间平移(波形滞后)
            if ekf_delay_counter % 100 == 0:
                now = time.time()
                if hasattr(main, "_last_probe_t"):
                    wall_dt = (now - main._last_probe_t) / 100.0
                    print(
                        f"[step {ekf_delay_counter:5d}] "
                        f"wall_dt={wall_dt*1000:5.1f}ms (expect {dt*1000:.1f}ms) | "
                        f"GT  v=[{gt_vel[0]:+.2f} {gt_vel[1]:+.2f} {gt_vel[2]:+.2f}] | "
                        f"EKF v=[{ekf_vel[0]:+.2f} {ekf_vel[1]:+.2f} {ekf_vel[2]:+.2f}]"
                    )
                main._last_probe_t = now

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