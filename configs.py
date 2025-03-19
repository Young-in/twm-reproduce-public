from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size: int = 48
    rollout_horizon: int = 8
    num_epochs: int = 10
    learning_rate: float = 0.001
    max_grad_norm: float = 0.5
