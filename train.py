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


def rollout(agent, env, env_params, curr_obs, env_state, horizon, rollout_rng):
    agent_state = jnp.zeros((curr_obs.shape[0], 256))

    rollout_rngs = jax.random.split(rollout_rng, horizon)
    obs = curr_obs
    obss = [obs]
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    for i in range(horizon):
        obs = obs[:, None, ...]

        pi, value, agent_state = agent(obs, agent_state)
        rng, sample_rng = jax.random.split(rollout_rngs[i])
        action, log_prob = pi.sample_and_log_prob(seed=sample_rng)

        action = action.squeeze(axis=1)
        log_prob = log_prob.squeeze(axis=1)

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(
            step_rng, env_state, action, env_params
        )
        obss.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)

    return (obs, env_state), (
        jnp.concatenate(obss),
        jnp.concatenate(actions),
        jnp.concatenate(log_probs),
        jnp.concatenate(values),
        jnp.concatenate(rewards),
        jnp.concatenate(dones),
    )


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

    agent = Agent(
        num_actions=env.action_space(env_params).n,
        ac_config=cfg.ac_config,
        rngs=nnx.Rngs(0),
    )

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
            env,
            env_params,
            curr_obs,
            env_state,
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
