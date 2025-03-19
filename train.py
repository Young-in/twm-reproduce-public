import functools
import time
import pyrallis

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from craftax import craftax_env

from env.wrapper import AutoResetEnvWrapper, BatchEnvWrapper
from nets.agent import Agent
from configs import TrainConfig


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

    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, cfg.batch_size)

    @functools.partial(nnx.jit, static_argnums=(3,))
    def rollout(agent, curr_obs, env_state, horizon, rollout_rng):
        agent_state = jnp.zeros((curr_obs.shape[0], 256))

        def one_step(state, rng):
            obs, env_state, agent_state = state

            pi, value, agent_state = agent(obs[:, None, ...], agent_state)
            rng, sample_rng = jax.random.split(rng)
            action, log_prob = pi.sample_and_log_prob(seed=sample_rng)

            action = action.squeeze(axis=1)
            log_prob = log_prob.squeeze(axis=1)
            value = value.squeeze(axis=1)

            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, info = env.step(
                step_rng, env_state, action, env_params
            )
            return (next_obs, env_state, agent_state), (
                obs,
                action,
                log_prob,
                value,
                reward,
                done,
            )

        (curr_obs, env_state, agent_state), (
            obs,
            action,
            log_prob,
            value,
            reward,
            done,
        ) = nnx.scan(
            one_step, out_axes=(nnx.transforms.iteration.Carry, 1), length=horizon
        )(
            (
                curr_obs,
                env_state,
                agent_state,
            ),
            jax.random.split(rollout_rng, horizon),
        )
        _, last_value, _ = agent(curr_obs[:, None, ...], agent_state)
        return (curr_obs, env_state), (
            obs,
            action,
            log_prob,
            jnp.concatenate((value, last_value), axis=1),
            reward,
            done,
        )

    agent = Agent(
        num_actions=env.action_space(env_params).n,
        ac_config=cfg.ac_config,
        rngs=nnx.Rngs(0),
    )

    nnx.display(agent)

    rng, env_rng = jax.random.split(rng)

    start_time = time.time()
    curr_obs, env_state = env.reset(env_rng, env_params)
    end_time = time.time()
    print(f"Reset time: {end_time - start_time:.2f}s")

    for step in range(10):
        print(f"{step=}")
        rng, rollout_rng = jax.random.split(rng)

        (curr_obs, env_state), (obs, action, log_prob, value, reward, done) = rollout(
            agent,
            curr_obs,
            env_state,
            cfg.rollout_horizon,
            rollout_rng,
        )

        agent_state = jnp.zeros((curr_obs.shape[0], 256))
        agent.loss(obs, agent_state, action, reward, done, log_prob, value)

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=cfg.learning_rate),
    )
    train_state = nnx.Optimizer(agent, tx)


if __name__ == "__main__":
    main()
