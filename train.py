import pyrallis

import jax
from flax import nnx
import optax

from craftax import craftax_env

from nets.agent import Agent
from configs import TrainConfig


def rollout(env, env_state, env_params, agent, batch_size, horizon, rollout_rng):
    agent.eval()
    agent_state = agent.rnn_cell.initialize_carry((batch_size, 256))

    def one_step(state, rng):
        env_state, agent_state = state
        obs = nnx.vmap(env.get_obs)(env_state)
        obs = obs[:, None, ...]  # Add time axis

        pi, value, agent_state = agent(obs, agent_state)
        rng, sample_rng = jax.random.split(rng)
        action, log_prob = pi.sample_and_log_prob(seed=sample_rng)

        # Remove time axis
        action = action.squeeze(axis=1)
        log_prob = log_prob.squeeze(axis=1)

        rng, _rng = jax.random.split(rng)
        step_rngs = jax.random.split(_rng, batch_size)
        obs, env_state, reward, done, info = nnx.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_rngs, env_state, action, env_params)

        return (env_state, agent_state), (obs, action, log_prob, value, reward, done)

    rollout_rngs = jax.random.split(rollout_rng, horizon)
    (env_state, agent_state), (obs, action, log_prob, value, reward, done) = nnx.scan(
        one_step,
        out_axes=(nnx.transforms.iteration.Carry, 1),
        length=horizon,
    )((env_state, agent_state), rollout_rngs)

    return env_state, (obs, action, log_prob, value, reward, done)


@pyrallis.wrap()
def main(cfg: TrainConfig):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, 3)

    # Create environment
    env = craftax_env.make_craftax_env_from_name(
        "Craftax-Classic-Pixels-v1", auto_reset=True
    )
    env_params = env.default_params

    agent = Agent(num_actions=env.action_space(env_params).n, rngs=nnx.Rngs(0))

    rng, _rng = jax.random.split(rng)
    env_rngs = jax.random.split(_rng, cfg.batch_size)
    obs, env_state = nnx.vmap(env.reset, in_axes=(0, None))(env_rngs, env_params)

    for step in range(10):
        print(f"{step=}")
        rng, rollout_rng = jax.random.split(rng)
        with jax.checking_leaks():
            env_state, (obs, action, log_prob, value, reward, done) = rollout(
                env,
                env_state,
                env_params,
                agent,
                cfg.batch_size,
                cfg.rollout_horizon,
                rollout_rng,
            )

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=cfg.learning_rate),
    )
    train_state = nnx.Optimizer(agent, tx)


if __name__ == "__main__":
    main()
