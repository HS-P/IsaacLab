# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import sys
import os

# 현재 train.py가 있는 디렉토리 기준 상위/하위 경로 설정:
ext_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exts", "berkeley_humanoid"))
if ext_module_path not in sys.path:
    sys.path.insert(0, ext_module_path)
"""Script to train RL agent with RSL-RL."""
"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=8192, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
from isaaclab.app import AppLauncher
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
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
# Import extensions to set up environment tasks
import berkeley_humanoid.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import torch
import torch.nn as nn
from isaaclab.envs.mdp.env_encoder import EnvEncoderMLP
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import atexit
import wandb
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from isaaclab.envs.mdp.env_encoder import EnvEncoderMLP

def cleanup_wandb(signum=None, frame=None):
    """wandb와 환경을 안전하게 종료하는 함수"""
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

def main():
    """Train with RSL-RL agent."""
    try:
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
            args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        
        # wandb 초기화
        wandb.init(
            project="berkeley-humanoid",
            name=agent_cfg.experiment_name,
            config={
                "task": args_cli.task,
                "num_envs": args_cli.num_envs,
                "max_iterations": args_cli.max_iterations
            }
        )

        # 종료 핸들러 등록
        atexit.register(cleanup_wandb)
        signal.signal(signal.SIGINT, cleanup_wandb)
        signal.signal(signal.SIGTERM, cleanup_wandb)

        # Logging directory 설정
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        if args_cli.max_iterations:
            agent_cfg.max_iterations = args_cli.max_iterations

        # 환경 생성 및 랩핑
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = RslRlVecEnvWrapper(env)
        
        # runner 생성
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)

        # # checkpoint 로드 후 모델 재초기화 (구조를 현재와 맞춤)
        # if agent_cfg.resume:
        #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        #     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        #     state_dict = torch.load(resume_path, map_location=agent_cfg.device)
        #     # 새로운 모델 인스턴스 생성
        #     policy_model = ModifiedPolicyNetwork(
        #         obs_dim=1,
        #         env_input_dim=10,
        #         hidden_dim=512,
        #         action_dim=1,
        #         z_dim=8
        #     ).to(agent_cfg.device)
        #     # strict=False를 사용하여 필수 키가 일치하지 않는 경우에도 무시하도록 처리
        #     policy_model.load_state_dict(state_dict, strict=False)
        #     runner.alg.policy = policy_model
        # else:
        #     policy_model = ModifiedPolicyNetwork(
        #         obs_dim=1,
        #         env_input_dim=10,
        #         hidden_dim=512,
        #         action_dim=1,
        #         z_dim=8
        #     ).to(agent_cfg.device)
        #     runner.alg.policy = policy_model

        # if not hasattr(runner.alg.policy, 'env_input_dim'):
        #     runner.alg.policy.env_input_dim = 10

        # 환경에 seed 설정
        env.seed(agent_cfg.seed)

        # 설정 덤프
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        # wandb logging callback 정의 (변경 없음)
        def wandb_log_callback(log_data):
            if log_data is None:
                return
            try:
                metrics = {}
                train_metrics = [
                    "Value function loss",
                    "Surrogate loss", 
                    "Mean action noise std",
                    "Mean reward",
                    "Mean episode length"
                ]
                for key, value in log_data.items():
                    # 만약 value가 torch.Tensor라면 반드시 CPU로 옮기고 scalar 또는 list로 변환
                    if isinstance(value, torch.Tensor):
                        # 스칼라인 경우 .item()을 사용
                        if value.ndim == 0:
                            metrics[f"train/{key}"] = value.cpu().item()
                        else:
                            metrics[f"train/{key}"] = value.detach().cpu().numpy().tolist()
                    else:
                        metrics[f"train/{key}"] = value

                    if key.startswith("Episode_Reward/"):
                        # 마찬가지로 스칼라 변환
                        if isinstance(value, torch.Tensor):
                            metrics[f"rewards/{key}"] = value.cpu().item() if value.ndim == 0 else value.detach().cpu().numpy().tolist()
                        else:
                            metrics[f"rewards/{key}"] = value  
                    elif key.startswith("Curriculum/"):
                        if isinstance(value, torch.Tensor):
                            metrics[f"command_velocity/{key}"] = value.cpu().item() if value.ndim == 0 else value.detach().cpu().numpy().tolist()
                        else:
                            metrics[f"command_velocity/{key}"] = value

                if hasattr(env.unwrapped, 'reward_manager') and hasattr(env.unwrapped, 'max_episode_length'):
                    try:
                        term_cfg = env.unwrapped.reward_manager.get_term_cfg('joint_pos_pitch_exp')
                        rew = env.unwrapped.reward_manager._episode_sums['joint_pos_pitch_exp']
                        # 반드시 CPU로 옮기고 scalar로 변환
                        joint_pos_pitch_mean_reward = torch.mean(rew).cpu().item()
                        max_episode_length = env.unwrapped.max_episode_length  # 이미 일반 숫자라면 그대로 사용
                        metrics.update({
                            "curriculum/joint_pos_pitch_mean_reward": joint_pos_pitch_mean_reward,
                            "curriculum/max_episode_length": max_episode_length,
                            "curriculum/joint_pos_pitch_reward": joint_pos_pitch_mean_reward / max_episode_length,
                            "curriculum/term_weight": term_cfg.weight,
                            "curriculum/step_dt": env.unwrapped.step_dt
                        })

                        if wandb.run is not None and metrics:
                            wandb.log(metrics, commit=False)
                    except Exception as e:
                        print(f"Warning: Failed to log curriculum metrics: {str(e)}")
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {str(e)}")
        
        env.unwrapped.log_callback = wandb_log_callback
        runner.log_callback = wandb_log_callback
        # ---------------------------------------------------------------------
        # Start training
        print("Runner 준비 완료")
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, 
            init_at_random_ep_len=True
        )

        wandb.finish()
        env.close()
    except Exception as e:
        print(f"Error: An error occurred during training: {str(e)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
