# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass, field


def config_value(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is not None:
        return raw
    return default


TARGET_WIN_RATE = 0.60
NODE_COUNT = 8

LOCAL_MODEL_PATH = config_value("LOCAL_MODEL_PATH", config_value("COTE_LOCAL_MODEL_PATH", ""))
LOCAL_MODEL_DTYPE = config_value("LOCAL_MODEL_DTYPE", config_value("COTE_LOCAL_MODEL_DTYPE", "auto"))
LOCAL_MODEL_DEVICE_MAP = config_value("LOCAL_MODEL_DEVICE_MAP", config_value("COTE_LOCAL_MODEL_DEVICE_MAP", "auto"))
LOCAL_MODEL_MAX_NEW_TOKENS = 256


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class HyperParams:
    # Equation (13): alpha_T + alpha_P = 1.
    alpha_t: float = env_float("COTE_ALPHA_T", 0.5)
    alpha_p: float = env_float("COTE_ALPHA_P", 0.5)

    # EL-BDPEA parameters from Algorithm 1.
    population_size: int = env_int("COTE_NP", 20)
    elite_ratio: float = env_float("COTE_RHO_E", 0.10)
    crossover_rate: float = env_float("COTE_PCROSS", 0.80)
    mutation_rate: float = env_float("COTE_PMUT", 0.10)
    repeated_samples: int = env_int("COTE_M_SAMPLES", 5)
    suspicion_delta: float = env_float("COTE_DELTA", 1.5)

    # Phase D parameters from Algorithm 2.
    eta_w: float = env_float("COTE_ETA_W", 1e-3)
    eta_p: float = env_float("COTE_ETA_P", 5e-4)
    theta_grad: float = env_float("COTE_THETA_GRAD", 1.0)
    edge_threshold: float = env_float("COTE_EDGE_THRESHOLD", 0.02)

    # Prompt vector geometry.
    prompt_dim: int = env_int("COTE_PROMPT_DIM", 64)
    prompt_bound: float = env_float("COTE_PROMPT_BOUND", 3.0)
    initial_sigma: float = env_float("COTE_INITIAL_SIGMA", 0.08)
    sigma_success_up: float = env_float("COTE_SIGMA_SUCCESS_UP", 1.05)
    sigma_success_down: float = env_float("COTE_SIGMA_SUCCESS_DOWN", 0.95)
    sigma_min: float = env_float("COTE_SIGMA_MIN", 0.01)
    sigma_max: float = env_float("COTE_SIGMA_MAX", 0.50)

    # Fitness coefficients matching the paper's channels.
    lambda_q: float = env_float("COTE_LAMBDA_Q", 0.35)
    gamma_q: float = env_float("COTE_GAMMA_Q", 1.0)
    lambda_i: float = env_float("COTE_LAMBDA_I", 0.30)
    lambda_edge_i: float = env_float("COTE_LAMBDA_EDGE_I", 0.20)
    lambda_d: float = env_float("COTE_LAMBDA_D", 0.20)
    lambda_c: float = env_float("COTE_LAMBDA_C", 0.08)
    lambda_l: float = env_float("COTE_LAMBDA_L", 1e-4)
    lambda_b: float = env_float("COTE_LAMBDA_B", 0.02)
    lambda_p: float = env_float("COTE_LAMBDA_P", 1e-4)
    eta_gt: float = env_float("COTE_ETA_GT", 0.25)
    gamma_reward: float = env_float("COTE_GAMMA_REWARD", 0.99)

    # Runtime guards for locally deployed model inference.
    use_local_model: bool = env_bool("COTE_USE_LOCAL_MODEL", True)
    node_local_model: bool = env_bool("COTE_NODE_LOCAL_MODEL", True)
    edge_local_model: bool = env_bool("COTE_EDGE_LOCAL_MODEL", True)
    evolve: bool = env_bool("COTE_EVOLVE", True)
    sample_action: bool = env_bool("COTE_SAMPLE_ACTION", False)
    local_model_budget: int = env_int("COTE_LOCAL_MODEL_BUDGET", 0)
    node_local_model_budget: int = env_int("COTE_NODE_LOCAL_MODEL_BUDGET", 0)
    node_local_score_scale: float = env_float("COTE_NODE_LOCAL_SCORE_SCALE", 18.0)
    edge_local_model_budget: int = env_int("COTE_EDGE_LOCAL_MODEL_BUDGET", 56)
    local_model_min_actions: int = env_int("COTE_LOCAL_MODEL_MIN_ACTIONS", 2)
    disable_edge_messages: bool = env_bool("COTE_DISABLE_EDGE_MESSAGES", False)
    prompt_evolve: bool = env_bool("COTE_PROMPT_EVOLVE", True)
    soft_prompt_train: bool = env_bool("COTE_SOFT_PROMPT_TRAIN", True)
    soft_prompt_steps: int = env_int("COTE_SOFT_PROMPT_STEPS", 1)
    soft_prompt_tokens: int = env_int("COTE_SOFT_PROMPT_TOKENS", 4)
    soft_prompt_lm_loss: bool = env_bool("COTE_SOFT_PROMPT_LM_LOSS", False)
    soft_prompt_lm_loss_weight: float = env_float("COTE_SOFT_PROMPT_LM_LOSS_WEIGHT", 0.05)
    topology_update: bool = env_bool("COTE_TOPOLOGY_UPDATE", True)
    topology_prune: bool = env_bool("COTE_TOPOLOGY_PRUNE", True)
    opt_mode: str = os.environ.get("COTE_OPT_MODE", "joint").strip().lower()

    # Short-benchmark module scales. Defaults keep the full COTE policy intact;
    # ablation/method variants lower the online contribution of removed modules.
    node_score_scale: float = env_float("COTE_NODE_SCORE_SCALE", 1.0)
    semantic_score_scale: float = env_float("COTE_SEMANTIC_SCORE_SCALE", 1.0)
    rule_score_scale: float = env_float("COTE_RULE_SCORE_SCALE", 1.0)
    finish_guard_bonus: float = env_float("COTE_FINISH_GUARD_BONUS", 0.0)
    pass_guard_bonus: float = env_float("COTE_PASS_GUARD_BONUS", 0.0)
    block_guard_bonus: float = env_float("COTE_BLOCK_GUARD_BONUS", 0.0)
    expert_score_scale: float = env_float("COTE_EXPERT_SCORE_SCALE", 0.0)
    action_dropout_rate: float = env_float("COTE_ACTION_DROPOUT_RATE", 0.0)

    # Ablation switches for the paper's three gradient/fitness channels.
    reward_channel: bool = env_bool("COTE_REWARD_CHANNEL", True)
    error_channel: bool = env_bool("COTE_ERROR_CHANNEL", True)
    belief_channel: bool = env_bool("COTE_BELIEF_CHANNEL", True)

    # Strict paper reproduction mode. Defaults off so the current runnable
    # benchmark path remains unchanged unless explicitly requested.
    strict_repro: bool = field(default_factory=lambda: env_bool("COTE_STRICT_REPRO", False))
    strict_replay_particles: int = field(default_factory=lambda: env_int("COTE_STRICT_REPLAY_PARTICLES", 512))
    strict_message_samples: int = field(default_factory=lambda: env_int("COTE_STRICT_MESSAGE_SAMPLES", 5))
    strict_candidate_replays: int = field(default_factory=lambda: env_int("COTE_STRICT_CANDIDATE_REPLAYS", 5))
    strict_duplicate_required: bool = field(default_factory=lambda: env_bool("COTE_STRICT_DUPLICATE_REQUIRED", True))
