from dataclasses import dataclass


@dataclass
class ActorCriticConfig:
    eps: float = 0.2
    gamma: float = 0.925
    ld: float = 0.625
    tgt_discount: float = 0.95

    td_loss_coef: float = 2.0
    ent_loss_coef: float = 0.01

@dataclass
class WorldModelConfig:
    sequence_length: int = 20
    num_updates: int = 500
    num_minibatches: int = 3

@dataclass
class TokenizerConfig:
    codebook_size: int = 4096
    num_updates: int = 25

@dataclass
class WandBConfig:
    enable: bool = True
    project_name: str = "twm_reproduce"
    exp_name: str = "twm"
    group_name: str = "twm"

@dataclass
class TrainConfig:
    seed: int = 0
    total_env_interactions: int = 1_000_000
    warmup_interactions: int = 200_000

    replay_buffer_size: int = 128_000

    batch_size: int = 48
    rollout_horizon: int = 96
    wm_rollout_horizon: int = 20
    num_epochs: int = 4
    learning_rate: float = 0.00045
    max_grad_norm: float = 0.5

    burn_in_horizon: int = 5

    num_minibatches: int = 8
    num_updates: int = 150

    wandb_config: WandBConfig = WandBConfig()
    ac_config: ActorCriticConfig = ActorCriticConfig()
    wm_config: WorldModelConfig = WorldModelConfig()
    token_config: TokenizerConfig = TokenizerConfig()
