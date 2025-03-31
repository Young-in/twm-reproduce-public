from dataclasses import asdict
import functools
import time
import pyrallis
import wandb

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.train_state import TrainState
import flashbax as fbx
import optax

from craftax import craftax_env

from configs import TrainConfig
from env.wrapper import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper
from nets.agent import Agent
from nets.nnt import NearestNeighborTokenizer
from nets.configuration import GPT2WorldModelConfig
from nets.configuration import GPT2WorldModelConfig
from nets.world_model import FlaxGPT2WorldModel
from utils.gae import calc_adv_tgt


def wm_rollout(
    agent,
    agent_state,
    curr_obs,
    curr_done,
    world_model,
    world_model_params,
    horizon,
    rollout_rng,
):
    def one_step(state, rng):
        obs, done, agent_state, past_key_values = state

        pi, value, agent_state = jax.lax.stop_gradient(
            agent(obs[:, None, ...], done[:, None, ...], agent_state)
        )
        rng, sample_rng = jax.random.split(rng)
        action, log_prob = pi.sample_and_log_prob(seed=sample_rng)

        action = action.squeeze(axis=1)
        log_prob = log_prob.squeeze(axis=1)
        value = value.squeeze(axis=1)

        @functools.partial(jax.jit, static_argnums=(1, 2))
        def imagine_state(key, world_model, params, state_action_ids, past_key_values):
            # TODO get config from world_model
            input_ids = state_action_ids[:, -config.tokens_per_block :]
            total_seq_len = state_action_ids.shape[1]
            position_ids = jnp.broadcast_to(
                jnp.arange(total_seq_len - config.tokens_per_block, total_seq_len)[
                    None, :
                ],
                input_ids.shape,
            )
            outputs = world_model(
                input_ids,
                position_ids=position_ids,
                params=params,
                past_key_values=past_key_values,
            )

            tokens_per_state = config.tokens_per_block - 1
            next_state_logits = outputs.observation_logits[:, -tokens_per_state:]
            next_state_ids = jax.random.categorical(key, next_state_logits)

            return next_state_ids, outputs.past_key_values

        # TODO: Encode obs + action into state_action_ids
        state_action_ids = None

        rng, step_rng = jax.random.split(rng)
        next_state_ids, past_key_values = imagine_state(
            step_rng,
            world_model,
            world_model_params,
            state_action_ids,
            past_key_values,
        )

        # TODO: Decode obs
        next_obs = None
        done = None

        return (next_obs, done, agent_state, past_key_values), (
            obs,
            action,
            log_prob,
            value,
            reward,
            done,
            info,
        )

    @functools.partial(jax.jit, static_argnums=(0, 1, 2))
    def init_cache(world_model, batch_size):
        # TODO get config from world_model
        return world_model.init_cache(batch_size, config.max_tokens)

    batch_size = curr_obs.shape[0]
    past_key_values = init_cache(world_model, batch_size)
    (curr_obs, curr_done, agent_state, _), (
        obs,
        action,
        log_prob,
        value,
        reward,
        done,
        info,
    ) = nnx.scan(one_step, out_axes=(nnx.transforms.iteration.Carry, 1))(
        (curr_obs, curr_done, agent_state, past_key_values),
        jax.random.split(rollout_rng, horizon),
    )

    _, last_value, _ = jax.lax.stop_gradient(
        agent(curr_obs[:, None, ...], curr_done[:, None, ...], agent_state)
    )

    return (
        obs,
        action,
        log_prob,
        jnp.concatenate((value, last_value), axis=1),
        reward,
        done,
        info,
    )


@pyrallis.wrap()
def main(cfg: TrainConfig):
    if cfg.wandb_config.enable:
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
    codebook = jnp.zeros((cfg.token_config.codebook_size, 7, 7, 3)) - 1
    codebook_size = jnp.array(0)
    curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

    tokenizer = NearestNeighborTokenizer(cfg.token_config.codebook_size)

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
    )
    policy_train_state = nnx.Optimizer(agent, tx)

    # Create world model
    # TODO: Create GPT2WorldModelConfig from cfg.wm_config
    config = GPT2WorldModelConfig(
        num_actions=17,
        tokens_per_block=82,
        max_blocks=20,
        vocab_size=4096,
        n_positions=82 * 20,
        n_embd=128,
        n_layer=3,
        n_head=8,
        n_inner=None,  # defaults to 4 * n_embd
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    input_shape = (cfg.batch_size, config.max_tokens)
    world_model = FlaxGPT2WorldModel(config, input_shape, cfg.seed)
    rng, init_weights_rng = jax.random.split(rng)
    world_model_params = world_model.init_weights(init_weights_rng, input_shape)
    # TODO add lr=0.001 to cfg.wm_config
    world_model_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=0.001, eps=1e-5),
    )
    world_model_train_state = TrainState.create(
        apply_fn=world_model.apply, params=world_model_params, tx=world_model_tx
    )

    # Create replay buffer
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        buffer = fbx.make_trajectory_buffer(
            add_batch_size=cfg.batch_size,
            sample_batch_size=cfg.batch_size,
            sample_sequence_length=cfg.wm_rollout_horizon,
            period=1,
            min_length_time_axis=cfg.wm_rollout_horizon,
            max_size=cfg.replay_buffer_size,
        )

        buffer_state = buffer.init(
            {
                "obs": jnp.zeros((63, 63, 3), dtype=jnp.float32),
                "action": jnp.zeros((), dtype=jnp.int32),
                "reward": jnp.zeros((), dtype=jnp.float32),
                "done": jnp.zeros((), dtype=jnp.bool),
            }
        )

    tgt_mean = 0
    tgt_std = 0
    debiasing = 0

    for step in range(
        0, cfg.total_env_interactions, cfg.batch_size * cfg.rollout_horizon
    ):
        print(f"{step=}")

        # 1. Collect data from environment

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

        with jax.default_device(cpu):
            buffer_state = buffer.add(
                buffer_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
            )

        if cfg.wandb_config.enable:
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

        # 2. Update policy on environment data

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
                    policy_train_state.model
                )

                policy_train_state.update(grads=grads)

                mini_logs.append(
                    {
                        "loss": loss,
                        **metrics,
                    }
                )

        logs = {}

        for k in mini_logs[0].keys():
            logs[k] = jnp.array([l[k] for l in mini_logs]).mean()

        if cfg.wandb_config.enable:
            wandb.log(logs, step=step + cfg.batch_size * cfg.rollout_horizon)

        agent_state = next_agent_state
        curr_done = next_done

        # 3. Update world model

        # Update tokenizer
        for _ in range(cfg.token_config.num_updates):
            rng, sample_rng = jax.random.split(rng)
            data = buffer.sample(buffer_state, sample_rng)
            
            obs = data.experience["obs"]

            codebook, codebook_size = tokenizer.update(obs, codebook, codebook_size)
        

        for _ in range(cfg.wm_config.num_updates):
            rng, sample_rng = jax.random.split(rng)
            data = buffer.sample(buffer_state, sample_rng)

            obs = data.experience["obs"]
            action = data.experience["action"]
            reward = data.experience["reward"]
            done = data.experience["done"]

            # TODO: Train world model
            token = tokenizer(obs, codebook)

            # TODO: Convert obs + action into state_action_ids
            state_action_ids = None

            @functools.partial(jax.jit, static_argnums=(0,))
            def loss_fn(
                world_model,
                params,
                dropout_key,
                state_action_ids,
                rewards,
                terminations,
            ):
                return world_model.loss(
                    params, dropout_key, state_action_ids, rewards, terminations
                )

            rng, dropout_rng = jax.random.split(rng)
            grads = jax.grad(loss_fn)(
                world_model,
                world_model_train_state.params,
                dropout_rng,
                state_action_ids,
                reward,
                done,
            )
            world_model_train_state = world_model_train_state.apply_gradients(
                grads=grads
            )

            # TODO: Convert obs + action into state_action_ids
            state_action_ids = None

            @functools.partial(jax.jit, static_argnums=(0,))
            def loss_fn(
                world_model,
                params,
                dropout_key,
                state_action_ids,
                rewards,
                terminations,
            ):
                return world_model.loss(
                    params, dropout_key, state_action_ids, rewards, terminations
                )

            rng, dropout_rng = jax.random.split(rng)
            grads = jax.grad(loss_fn)(
                world_model,
                world_model_train_state.params,
                dropout_rng,
                state_action_ids,
                reward,
                done,
            )
            world_model_train_state = world_model_train_state.apply_gradients(
                grads=grads
            )

        # 4. Update policy on imagined data

        if step + cfg.batch_size * cfg.rollout_horizon >= cfg.warmup_interactions:
            for _ in range(cfg.num_updates):
                # TODO: create new agent_state using burn-in frames
                imagination_agent_state = None

                rng, sample_rng = jax.random.split(rng)
                data = buffer.sample(buffer_state, sample_rng)

                obs = data.experience["obs"]
                action = data.experience["action"]
                reward = data.experience["reward"]
                done = data.experience["done"]

                rng, rollout_rng = jax.random.split(rng)
                (
                    obs,
                    action,
                    log_prob,
                    value,
                    reward,
                    done,
                    info,
                ) = wm_rollout(
                    agent,
                    imagination_agent_state,
                    obs,
                    done,
                    world_model,
                    cfg.wm_rollout_horizon,
                    rollout_rng,
                )

                value = value * jnp.maximum(
                    tgt_std / jnp.maximum(debiasing, 1e-1), 1e-1
                ) + tgt_mean / jnp.maximum(debiasing, 1e-2)

                adv, tgt = calc_adv_tgt(
                    reward, done, value, cfg.ac_config.gamma, cfg.ac_config.ld
                )

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                mini_logs = []

                start_idx = 0
                end_idx = cfg.batch_size

                # TODO: Not sure moving averages are shared with online learning
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
                    imagination_agent_state[start_idx:end_idx],
                    action[start_idx:end_idx],
                    log_prob[start_idx:end_idx],
                    adv[start_idx:end_idx],
                    tgt_mini,
                )

                (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                    policy_train_state.model
                )

                policy_train_state.update(grads=grads)

                mini_logs.append(
                    {
                        "loss": loss,
                        **metrics,
                    }
                )


if __name__ == "__main__":
    main()
