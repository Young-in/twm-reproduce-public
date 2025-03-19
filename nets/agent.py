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
        self.rnn_cell = nnx.GRUCell(
            in_features=8 * 8 * 128,
            hidden_features=256,
            rngs=rngs,
        )
        self.actor_critic = ActorCritic(
            input_dim=(8 * 8 * 128) + 256,
            num_actions=num_actions,
            rngs=rngs,
        )

    def __call__(self, obs, prev_state):
        B, T, *_ = obs.shape
        z = self.encoder(obs)
        z = z.reshape((B, T, -1))

        def rnn_step(prev_state, x):
            prev_state, y = self.rnn_cell(prev_state, x)
            return prev_state, y

        prev_state, y = nnx.scan(
            rnn_step,
            in_axes=(nnx.transforms.iteration.Carry, 1),
            out_axes=(nnx.transforms.iteration.Carry, 1),
        )(prev_state, z)

        pi, v = self.actor_critic(jnp.concatenate([z, y], axis=-1))
        return pi, v, prev_state


def main():
    model = Agent(num_actions=10, rngs=nnx.Rngs(0))

    x = jnp.ones((16, 20, 63, 63, 3))
    y = model(x)


if __name__ == "__main__":
    main()
