from dataclasses import asdict
from flax import nnx
import jax.numpy as jnp

from nets.impala_cnn import ImpalaCNN
from nets.actor_critic import ActorCritic
from nets.rnn import RNN
from configs import ActorCriticConfig


class Agent(nnx.Module):
    def __init__(
        self,
        num_actions: int,
        ac_config: ActorCriticConfig = ActorCriticConfig(),
        *,
        rngs: nnx.Rngs
    ):
        self.encoder = ImpalaCNN(
            channels=[64, 64, 128],
            rngs=rngs,
        )
        self.rnn = RNN(
            in_features=8 * 8 * 128,
            hidden_features=256,
            rngs=rngs,
        )
        self.actor_critic = ActorCritic(
            input_dim=(8 * 8 * 128) + 256,
            num_actions=num_actions,
            **asdict(ac_config),
            rngs=rngs,
        )

    def __call__(self, obs, prev_state):
        B, T, *_ = obs.shape
        z = self.encoder(obs)
        z = nnx.relu(z)
        z = z.reshape((B, T, -1))

        prev_state, y = self.rnn(prev_state, z)

        pi, v = self.actor_critic(jnp.concatenate([z, y], axis=-1))
        return pi, v, prev_state


def main():
    model = Agent(num_actions=10, rngs=nnx.Rngs(0))

    x = jnp.ones((16, 20, 63, 63, 3))
    y = model(x)


if __name__ == "__main__":
    main()
