"""
export_weights.py — Extract trained PPO actor weights → JSON for browser inference.
No TF.js needed: the browser runs a raw matrix-multiply forward pass.

Usage:
    python export_weights.py
    python export_weights.py --model ./checkpoints/sat_ppo_final --output ../trained_weights.json
"""

import json, os, argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sat_env import SatelliteAvoidanceEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="./checkpoints/sat_ppo_final")
    p.add_argument("--norm",   default="./checkpoints/vecnormalize.pkl")
    p.add_argument("--output", default="../trained_weights.json")
    args = p.parse_args()

    model = PPO.load(args.model)

    # ── extract actor (policy) network weights ──
    layers = []
    for module in model.policy.mlp_extractor.policy_net:
        if hasattr(module, "weight"):
            layers.append({
                "weight": module.weight.detach().cpu().numpy().tolist(),
                "bias":   module.bias.detach().cpu().numpy().tolist(),
                "activation": "tanh",
            })
    # final action head (no activation — we apply tanh in browser)
    act = model.policy.action_net
    layers.append({
        "weight": act.weight.detach().cpu().numpy().tolist(),
        "bias":   act.bias.detach().cpu().numpy().tolist(),
        "activation": "none",
    })

    # ── extract observation normalization stats ──
    norm_stats = None
    if os.path.exists(args.norm):
        env = DummyVecEnv([SatelliteAvoidanceEnv])
        env = VecNormalize.load(args.norm, env)
        norm_stats = {
            "obs_mean": env.obs_rms.mean.tolist(),
            "obs_std":  np.sqrt(env.obs_rms.var + 1e-8).tolist(),
            "clip_obs": float(env.clip_obs),
        }

    payload = {
        "layers": layers,
        "normalization": norm_stats,
        "config": {
            "n_sats": 8,
            "k_threats": 3,
            "obs_per_sat": 32,
            "max_dv": 0.04,
            "caution_r": 1.2,
        },
    }

    with open(args.output, "w") as f:
        json.dump(payload, f)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"✓ Exported {len(layers)} layers → {args.output} ({size_kb:.0f} KB)")
    if norm_stats:
        print(f"  Included obs normalization (mean/std from VecNormalize)")


if __name__ == "__main__":
    main()
