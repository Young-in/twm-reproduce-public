from dataclasses import dataclass


@dataclass
class ActorCriticConfig:
    eps: float = 0.2
    gamma: float = 0.925
    ld: float = 0.625

    td_loss_coef: float = 2.0
    ent_loss_coef: float = 0.01


@dataclass
class TrainConfig:
    batch_size: int = 48
    rollout_horizon: int = 8
    num_epochs: int = 10
    learning_rate: float = 0.001
    max_grad_norm: float = 0.5

    ac_config: ActorCriticConfig = ActorCriticConfig()
