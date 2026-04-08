"""
train.py — Train a PPO agent for satellite collision avoidance.

Usage:
    python train.py                        # default 500k steps
    python train.py --timesteps 2000000    # longer run
    python train.py --lr 1e-4 --timesteps 1000000
"""

import os, argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback
)
from sat_env import SatelliteAvoidanceEnv


class MetricsCallback(BaseCallback):
    """Print collision / fuel stats every N steps."""
    def __init__(self, every=10_000, verbose=0):
        super().__init__(verbose)
        self.every = every

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            infos = self.locals.get("infos", [])
            if infos:
                colls = [i.get("collisions", 0) for i in infos]
                fuel  = [i.get("fuel_mean", 0) for i in infos]
                print(f"  [step {self.num_timesteps:>8}]  "
                      f"collisions={sum(colls):.0f}  fuel_mean={sum(fuel)/len(fuel):.1f}")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = DummyVecEnv([SatelliteAvoidanceEnv])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
    )

    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=args.save_dir,
                           name_prefix="sat_ppo"),
        MetricsCallback(every=10_000),
    ]

    print(f"Training PPO for {args.timesteps:,} timesteps …")
    print(f"Obs dim = {model.observation_space.shape}, Act dim = {model.action_space.shape}")

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=True)

    model.save(os.path.join(args.save_dir, "sat_ppo_final"))
    env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
    print(f"\n✓ Saved model + normalizer to {args.save_dir}/")


if __name__ == "__main__":
    main()
