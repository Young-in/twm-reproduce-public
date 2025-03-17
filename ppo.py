import distrax
from flax import nnx
import jax.numpy as jnp

from nets.impala_cnn import ImpalaCNN


class Actor(nnx.Module):
    def __init__(
        self, input_dim: int, intermediate_dim: int, num_actions: int, *, rngs: nnx.Rngs
    ):
        self.norm1 = nnx.LayerNorm(
            num_features=input_dim,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=input_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=num_actions,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm1(x)
        x = nnx.relu(self.linear1(x))
        x = x + nnx.relu(self.linear2(x))
        x = x + nnx.relu(self.linear3(x))
        x = self.norm2(x)
        x = self.linear4(x)
        x = distrax.Categorical(logits=x)
        return x


class Critic(nnx.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(
            num_features=input_dim,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=input_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=1,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm1(x)
        x = nnx.relu(self.linear1(x))
        x = x + nnx.relu(self.linear2(x))
        x = x + nnx.relu(self.linear3(x))
        x = self.norm2(x)
        x = self.linear4(x)
        return jnp.squeeze(x, axis=-1)


class ActorCritic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        *,
        eps: float = 0.2,
        gamma: float = 0.925,
        ld: float = 0.625,
        rngs: nnx.Rngs
    ):
        intermediate_dim = 2048
        self.eps = eps
        self.gamma = gamma
        self.ld = ld

        self.actor = Actor(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            num_actions=num_actions,
            rngs=rngs,
        )
        self.critic = Critic(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            rngs=rngs,
        )

    def __call__(self, state):
        return self.actor(state), self.critic(state)

    def loss(self, state, action, reward, done, old_pi_log_prob, old_value):
        # state: (B, T + 1, 8 * 8 * 128 + 256)
        # action: (B, T)
        # reward: (B, T)
        # done: (B, T)
        # old_pi_log_prob: (B, T)
        # old_value: (B, T + 1)

        delta = reward + (1 - done) * self.gamma * old_value[:, 1:] - old_value[:, :-1]

        adv = nnx.scan(
            lambda d, dt: self.gamma * self.ld * d + dt, 0, delta, reverse=True
        )

        adv = (1 - done) * adv + done * delta

        tgt = adv + old_value[:, :-1]

        pi, v = self(state)
        log_r = pi.log_prob(action) - old_pi_log_prob

        policy_loss = -jnp.minimum(
            jnp.exp(log_r) * adv,
            jnp.clip(jnp.exp(log_r), 1 - self.eps, 1 + self.eps) * adv,
        ).mean()
        value_loss = jnp.square(v - tgt).mean()
        ent_loss = -pi.entropy().mean()

        return policy_loss + value_loss + ent_loss


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
        carry, y = self.rnn(z, initial_carry=self.prev_state)
        pi, v = self.actor_critic(jnp.concatenate([z, y], axis=-1))
        return pi, v


def main():
    model = Agent(num_actions=10, rngs=nnx.Rngs(0))

    x = jnp.ones((16, 20, 63, 63, 3))
    y = model(x)


if __name__ == "__main__":
    main()
