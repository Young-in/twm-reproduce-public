from dataclasses import dataclass


@dataclass
class ActorCriticParams:
    eps: float = 0.2

    td_loss_coef: float = 2.0
    ent_loss_coef: float = 0.01


@dataclass
class ActorCriticConfig:
    params: ActorCriticParams = ActorCriticParams()

    num_epochs: int = 4
    num_minibatches: int = 8
    num_updates: int = 150

    gamma: float = 0.925
    ld: float = 0.625
    tgt_discount: float = 0.95

    learning_rate: float = 0.00045


@dataclass
class WorldModelParams:
    tokens_per_block: int = 82
    max_blocks: int = 20
    vocab_size: int = 4096
    n_positions: int = 82 * 20
    n_embd: int = 128
    n_layer: int = 3
    n_head: int = 8
    n_inner = None  # defaults to 4 * n_embd
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1


@dataclass
class WorldModelConfig:
    params: WorldModelParams = WorldModelParams()

    num_updates: int = 500
    num_minibatches: int = 3

    learning_rate: float = 0.001


@dataclass
class TokenizerParams:
    codebook_size: int = 4096


@dataclass
class TokenizerConfig:
    params: TokenizerParams = TokenizerParams()

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
    max_grad_norm: float = 0.5

    burn_in_horizon: int = 5

    wandb_config: WandBConfig = WandBConfig()
    ac_config: ActorCriticConfig = ActorCriticConfig()
    wm_config: WorldModelConfig = WorldModelConfig()
    token_config: TokenizerConfig = TokenizerConfig()
