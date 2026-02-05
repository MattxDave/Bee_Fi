# train_utils.py
import csv
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int | None):
    """
    Set seeds across Python, NumPy, and PyTorch for reproducible runs.
    If seed is None, this is a no-op.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CUDA math deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Small helper in case you want to centralize device selection."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Config (sensible defaults)
# ---------------------------
DEFAULT_TRAIN = {
    "episodes": 100,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "gamma": 0.99,
}

# Defaults aligned with the outside-grid environment
DEFAULT_ENV = {
    "num_bees": 5,
    "num_flowers": 12,
    "grid_size": 20,
    "max_steps": 800,
    "time_window_min": 10,
    "time_window_max": 80,
    "harvest_radius": 3.0,  # a bit larger helps early learning
    "lambda_z": 0.1,
    "knn_k": 3,
    # Orbit & placement outside the grid
    "orbit_scale": 1.2,  # >1.0 puts orbit outside
    "flower_edge_band": 3,  # spawn near edges
    "spawn_on_orbit_ratio": 0.8,  # some flowers near orbits
    # Shaping
    "shaping_weight": 0.05,  # reward for moving closer
    "anti_spam_pen": -0.005,  # tiny penalty for blind harvests
}

DEFAULT_OUTPUT_PATH = "outputs"


def _merge_defaults(d: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    if d:
        # preserve all default keys and override with provided values
        for k in defaults:
            if k in d:
                out[k] = d[k]
        # include any extra user-specified keys
        for k, v in d.items():
            if k not in out:
                out[k] = v
    return out


def load_config(
    path: str = "/mnt/w/Bee/bee1/test/config.yaml",
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Load YAML and return (train_cfg, env_cfg, output_path).
    If the file is missing or partial, return sensible defaults.
    """
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        print(f"[load_config] {path} not found; using defaults.")

    # accept either 'train:' or legacy 'training:'
    train_raw = data.get("train", data.get("training", {})) or {}
    env_raw = data.get("env", {}) or {}
    out_raw = data.get("output", {}) or {}

    train_cfg = _merge_defaults(train_raw, DEFAULT_TRAIN)
    env_cfg = _merge_defaults(env_raw, DEFAULT_ENV)
    output_path = out_raw.get(
        "path", data.get("training", {}).get("model_output_path", DEFAULT_OUTPUT_PATH)
    )

    return train_cfg, env_cfg, output_path


# ---------------------------
# Model I/O + metrics
# ---------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_models(actor, critic, tag: str, out_dir: str):
    ensure_dir(out_dir)
    a_path = os.path.join(out_dir, f"{tag}_actor.pt")
    c_path = os.path.join(out_dir, f"{tag}_critic.pt")
    torch.save(actor.state_dict(), a_path)
    torch.save(critic.state_dict(), c_path)
    print(f"[save_models] wrote {a_path} and {c_path}")


def load_models(actor, critic, tag: str, out_dir: str) -> bool:
    """
    Load actor/critic weights saved by save_models(..., tag, out_dir).
    Returns True if both were loaded, else False.
    """
    a_path = os.path.join(out_dir, f"{tag}_actor.pt")
    c_path = os.path.join(out_dir, f"{tag}_critic.pt")

    ok = True
    if os.path.exists(a_path):
        actor.load_state_dict(torch.load(a_path, map_location="cpu"))
        actor.eval()
        print(f"[load_models] loaded {a_path}")
    else:
        print(f"[load_models] missing {a_path}")
        ok = False

    if os.path.exists(c_path):
        critic.load_state_dict(torch.load(c_path, map_location="cpu"))
        critic.eval()
        print(f"[load_models] loaded {c_path}")
    else:
        print(f"[load_models] missing {c_path}")
        ok = False

    return ok


def save_reward_history(history, out_dir: str, fname: str = "rewards.csv"):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(history, 1):
            w.writerow([i, float(r)])
    print(f"[save_reward_history] wrote {path}")
