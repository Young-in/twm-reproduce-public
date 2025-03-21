from dataclasses import dataclass


@dataclass
class ActorCriticConfig:
    eps: float = 0.2
    gamma: float = 0.925
    ld: float = 0.625
    tgt_discount: float = 0.95

    td_loss_coef: float = 0.5
    ent_loss_coef: float = 0.01


@dataclass
class TrainConfig:
    seed: int = 0
    total_env_interactions: int = 1_000_000
    batch_size: int = 48
    rollout_horizon: int = 96
    num_epochs: int = 4
    learning_rate: float = 0.00045
    max_grad_norm: float = 0.5

    num_minibatches: int = 8

    ac_config: ActorCriticConfig = ActorCriticConfig()
