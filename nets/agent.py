from flax import nnx
import jax.numpy as jnp

from nets.impala_cnn import ImpalaCNN
from nets.actor_critic import ActorCritic


class Agent(nnx.Module):
    def __init__(self, num_actions: int, *, rngs: nnx.Rngs):
        self.encoder = ImpalaCNN(
            channels=[64, 64, 128],
            rngs=rngs,
        )
        self.rnn = nnx.RNN(
            nnx.GRUCell(
                in_features=8 * 8 * 128,
                hidden_features=256,
                rngs=rngs,
            ),
            return_carry=True,
            rngs=rngs,
        )
        self.actor_critic = ActorCritic(
            input_dim=(8 * 8 * 128) + 256,
            num_actions=num_actions,
            rngs=rngs,
        )
        self.prev_state = None

    def __call__(self, obs):
        B, T, *_ = obs.shape
        z = self.encoder(obs)
        z = z.reshape((B, T, -1))

        if self.prev_state is None:
            self.prev_state = self.rnn.cell.initialize_carry((B, z.shape[-1]))

        self.prev_state, y = self.rnn(z, initial_carry=self.prev_state)
        pi, v = self.actor_critic(jnp.concatenate([z, y], axis=-1))
        return pi, v


def main():
    model = Agent(num_actions=10, rngs=nnx.Rngs(0))

    x = jnp.ones((16, 20, 63, 63, 3))
    y = model(x)


if __name__ == "__main__":
    main()
