from dataclasses import asdict
import functools
from flax import nnx
import jax.numpy as jnp

from nets.impala_cnn import ImpalaCNN
from nets.actor_critic import ActorCritic
from nets.rnn import RNN
from nets.nnt import NearestNeighborTokenizer
from nets.patch_mlp import PatchMLP
from configs import ActorCriticConfig


class Agent(nnx.Module):
    def __init__(
        self,
        num_actions: int,
        ac_config: ActorCriticConfig = ActorCriticConfig(),
        *,
        rngs: nnx.Rngs
    ):
        self.encoder = NearestNeighborTokenizer(
            codebook_size=4096,
        )
        self.embedding = nnx.Embed(
            num_embeddings=4096,
            features=128,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(
            num_features=9 * 9 * 128,
            rngs=rngs,
        )
        self.linear = nnx.Linear(
            in_features=9 * 9 * 128,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
        )
        self.rnn = RNN(
            in_features=256,
            hidden_features=256,
            rngs=rngs,
        )
        self.actor_critic = ActorCritic(
            input_dim=(9 * 9 * 128) + 256,
            num_actions=num_actions,
            **asdict(ac_config),
            rngs=rngs,
        )

    def __call__(self, obs, reset, prev_state, codebook, codebook_size):
        B, T, *_ = obs.shape
        z, codebook, codebook_size = self.encoder(
            obs.reshape(B * T, *_), codebook, codebook_size
        )
        z = self.embedding(z)
        z = nnx.relu(z)
        z = z.reshape((B, T, -1))

        r_in = self.norm(z)
        r_in = self.linear(r_in)
        r_in = nnx.relu(r_in)

        prev_state, y = self.rnn(prev_state, r_in, reset)
        y = nnx.relu(y)

        state = jnp.concatenate([z, y], axis=-1)

        pi, v = self.actor_critic(state)
        return pi, v, prev_state, codebook, codebook_size

    @nnx.jit
    def loss(
        self,
        obs,
        reset,
        prev_state,
        codebook,
        codebook_size,
        action,
        old_pi_log_prob,
        adv,
        tgt,
    ):
        B, T, *_ = obs.shape
        z, _, _ = self.encoder(obs.reshape(B * T, *_), codebook, codebook_size)
        z = self.embedding(z)
        z = nnx.relu(z)
        z = z.reshape((B, T, -1))

        r_in = self.norm(z)
        r_in = self.linear(r_in)
        r_in = nnx.relu(r_in)

        prev_state, y = self.rnn(prev_state, r_in, reset)
        y = nnx.relu(y)

        state = jnp.concatenate([z, y], axis=-1)

        return self.actor_critic.loss(state, action, old_pi_log_prob, adv, tgt)


def main():
    model = Agent(num_actions=10, rngs=nnx.Rngs(0))

    x = jnp.ones((16, 20, 63, 63, 3))
    y = model(x)


if __name__ == "__main__":
    main()
