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

from env.wrapper import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper
from nets.agent import Agent
from configs import TrainConfig
from utils.gae import calc_adv_tgt


@pyrallis.wrap()
def main(cfg: TrainConfig):
    wandb.init(
        project=cfg.wandb_config.project_name,
        config=asdict(cfg),
        name=f"{cfg.wandb_config.exp_name}_s{cfg.seed}",
        group=cfg.wandb_config.group_name,
    )
    rng = jax.random.PRNGKey(cfg.seed)

    # Create environment
    env = craftax_env.make_craftax_env_from_name(
        "Craftax-Classic-Pixels-v1", auto_reset=True
    )
    env_params = env.default_params

    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, cfg.batch_size)

    @functools.partial(nnx.jit, static_argnames=("horizon",))
    def rollout(
        agent,
        agent_state,
        curr_obs,
        curr_done,
        env_state,
        horizon,
        rollout_rng,
    ):
        def one_step(state, rng):
            obs, done, env_state, agent_state = state

            pi, value, agent_state = jax.lax.stop_gradient(
                agent(
                    obs[:, None, ...],
                    done[:, None, ...],
                    agent_state,
                )
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
                info,
            )

        (curr_obs, curr_done, env_state, agent_state), (
            obs,
            action,
            log_prob,
            value,
            reward,
            done,
            info,
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
            agent(
                curr_obs[:, None, ...],
                curr_done[:, None, ...],
                agent_state,
            )
        )
        return (curr_obs, curr_done, env_state, agent_state), (
            obs,
            action,
            log_prob,
            jnp.concatenate((value, last_value), axis=1),
            reward,
            done,
            info,
        )

    agent = Agent(
        num_actions=env.action_space(env_params).n,
        ac_config=cfg.ac_config,
        rngs=nnx.Rngs(cfg.seed),
    )

    nnx.display(agent)

    rng, env_rng = jax.random.split(rng)

    start_time = time.time()
    curr_obs, env_state = env.reset(env_rng, env_params)
    end_time = time.time()
    print(f"Reset time: {end_time - start_time:.2f}s")

    agent_state = agent.rnn.initialize_carry(cfg.batch_size)
    codebook = jnp.zeros((4096, 7, 7, 3)) - 1
    codebook_size = jnp.array(0)
    curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

    def lr_schedule(count):
        return cfg.learning_rate * (
            1.0
            - (count // (cfg.num_minibatches * cfg.num_epochs))
            / (cfg.total_env_interactions // (cfg.batch_size * cfg.rollout_horizon))
        )

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )
    train_state = nnx.Optimizer(agent, tx)

    tgt_mean = 0
    tgt_std = 0
    debiasing = 0

    for step in range(
        0, cfg.total_env_interactions, cfg.batch_size * cfg.rollout_horizon
    ):
        print(f"{step=}")
        rng, rollout_rng = jax.random.split(rng)

        (curr_obs, next_done, env_state, next_agent_state), (
            obs,
            action,
            log_prob,
            value,
            reward,
            done,
            info,
        ) = rollout(
            agent,
            agent_state,
            curr_obs,
            curr_done,
            env_state,
            cfg.rollout_horizon,
            rollout_rng,
        )

        wandb.log(
            {
                "rollout_reward": reward.mean(),
                "rollout_done": done.mean(),
                "rollout_log_prob": log_prob.mean(),
                "rollout_value": value.mean(),
                "target_mean": tgt_mean,
                "target_std": tgt_std,
                "debiasing": debiasing,
            },
            step=step + cfg.batch_size * cfg.rollout_horizon,
        )

        if info["returned_episode"].any():
            avg_episode_returns = jnp.average(
                info["returned_episode_returns"], weights=info["returned_episode"]
            )

            wandb.log(
                {
                    "rollout_return": avg_episode_returns,
                    "rollout_ends": info["returned_episode"].sum(),
                },
                step=step + cfg.batch_size * cfg.rollout_horizon,
            )

        reset = jnp.concatenate((curr_done[:, None], done[:, :-1]), axis=1)

        value = value * jnp.maximum(
            tgt_std / jnp.maximum(debiasing, 1e-1), 1e-1
        ) + tgt_mean / jnp.maximum(debiasing, 1e-2)

        adv, tgt = calc_adv_tgt(
            reward, done, value, cfg.ac_config.gamma, cfg.ac_config.ld
        )

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        mini_logs = []

        for epoch in range(cfg.num_epochs):
            for i in range(cfg.num_minibatches):
                start_idx = i * (cfg.batch_size // cfg.num_minibatches)
                end_idx = (i + 1) * (cfg.batch_size // cfg.num_minibatches)

                tgt_mini = tgt[start_idx:end_idx]
                tgt_mean = (
                    cfg.ac_config.tgt_discount * tgt_mean
                    + (1 - cfg.ac_config.tgt_discount) * tgt_mini.mean()
                )
                tgt_std = (
                    cfg.ac_config.tgt_discount * tgt_std
                    + (1 - cfg.ac_config.tgt_discount) * tgt_mini.std()
                )

                debiasing = (
                    cfg.ac_config.tgt_discount * debiasing
                    + (1 - cfg.ac_config.tgt_discount) * 1
                )

                tgt_mini = (
                    tgt_mini - tgt_mean / jnp.maximum(debiasing, 1e-2)
                ) / jnp.maximum(tgt_std / jnp.maximum(debiasing, 1e-1), 1e-1)

                loss_fn = lambda model: model.loss(
                    obs[start_idx:end_idx],
                    reset[start_idx:end_idx],
                    agent_state[start_idx:end_idx],
                    action[start_idx:end_idx],
                    log_prob[start_idx:end_idx],
                    adv[start_idx:end_idx],
                    tgt_mini,
                )

                (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                    train_state.model
                )

                train_state.update(grads=grads)

                mini_logs.append(
                    {
                        "loss": loss,
                        **metrics,
                    }
                )

        logs = {}

        for k in mini_logs[0].keys():
            logs[k] = jnp.array([l[k] for l in mini_logs]).mean()

        wandb.log(logs, step=step + cfg.batch_size * cfg.rollout_horizon)

        agent_state = next_agent_state
        curr_done = next_done


if __name__ == "__main__":
    main()
