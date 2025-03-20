from dataclasses import asdict
import functools
import time
import pyrallis
import wandb

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from craftax import craftax_env

from env.wrapper import AutoResetEnvWrapper, BatchEnvWrapper
from nets.agent import Agent
from configs import TrainConfig
from utils.gae import calc_adv_tgt

@pyrallis.wrap()
def main(cfg: TrainConfig):
    wandb.init(
        project="twm_reproduce",
        config=asdict(cfg),
        name="model-free",
    )
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

    @functools.partial(nnx.jit, static_argnames=("horizon",))
    def rollout(
        agent, agent_state, curr_obs, curr_done, env_state, horizon, rollout_rng
    ):
        def one_step(state, rng):
            obs, done, env_state, agent_state = state

            pi, value, agent_state = jax.lax.stop_gradient(
                agent(obs[:, None, ...], done[:, None, ...], agent_state)
            )
            rng, sample_rng = jax.random.split(rng)
            action, log_prob = pi.sample_and_log_prob(seed=sample_rng)

            action = action.squeeze(axis=1)
            log_prob = log_prob.squeeze(axis=1)
            value = value.squeeze(axis=1)

            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, info = env.step(
                step_rng, env_state, action, env_params
            )
            return (next_obs, done, env_state, agent_state), (
                obs,
                action,
                log_prob,
                value,
                reward,
                done,
            )

        (curr_obs, curr_done, env_state, agent_state), (
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
                curr_done,
                env_state,
                agent_state,
            ),
            jax.random.split(rollout_rng, horizon),
        )
        _, last_value, _ = jax.lax.stop_gradient(
            agent(curr_obs[:, None, ...], curr_done[:, None, ...], agent_state)
        )
        return (curr_obs, curr_done, env_state, agent_state), (
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

    agent_state = jnp.zeros((cfg.batch_size, 256))
    curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=cfg.learning_rate),
    )
    train_state = nnx.Optimizer(agent, tx)

    env_interactions = 0

    step = 0
    while env_interactions < cfg.total_env_interactions:
        print(f"{step=}")
        step += 1
        rng, rollout_rng = jax.random.split(rng)

        (curr_obs, next_done, env_state, next_agent_state), (
            obs,
            action,
            log_prob,
            value,
            reward,
            done,
        ) = rollout(
            agent,
            agent_state,
            curr_obs,
            curr_done,
            env_state,
            cfg.rollout_horizon,
            rollout_rng,
        )

        env_interactions += cfg.batch_size * cfg.rollout_horizon

        reset = jnp.concatenate((curr_done[:, None], done[:, :-1]), axis=1)

        adv, tgt = calc_adv_tgt(reward, done, value, cfg.ac_config.gamma, cfg.ac_config.ld)

        for epoch in range(cfg.num_epochs):
            for i in range(cfg.num_minibatches):
                start_idx = i * (cfg.batch_size // cfg.num_minibatches)
                end_idx = (i + 1) * (cfg.batch_size // cfg.num_minibatches)
                loss_fn = lambda model: model.loss(
                    obs[start_idx:end_idx],
                    reset[start_idx:end_idx],
                    agent_state[start_idx:end_idx],
                    action[start_idx:end_idx],
                    log_prob[start_idx:end_idx],
                    adv[start_idx:end_idx],
                    tgt[start_idx:end_idx],
                )

                loss, grads = nnx.value_and_grad(loss_fn)(train_state.model)
                print(f"loss before backprop: {loss}")

                train_state.update(grads=grads)

                wandb.log(
                    {
                        "loss": loss,
                    }
                )

        agent_state = next_agent_state
        curr_done = next_done


if __name__ == "__main__":
    main()
