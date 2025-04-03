from functools import partial

import jax
import jax.numpy as jnp
import tqdm

from nets.configuration import GPT2WorldModelConfig
from nets.world_model import FlaxGPT2WorldModelModule
from utils.meaasure_time import MeasureTime


def generate_initial_state(key, config, batch_size):
    state_ids_key, action_ids_key = jax.random.split(key)

    state_tokens_per_block = config.tokens_per_block - 1
    state_ids = jax.random.randint(
        state_ids_key,
        shape=(batch_size, state_tokens_per_block),
        minval=0,
        maxval=config.vocab_size,
    )

    action_ids = jax.random.randint(
        action_ids_key, shape=(batch_size, 1), minval=0, maxval=config.num_actions
    )

    return jnp.concatenate([state_ids, action_ids], axis=1)


def generate_action(key, config, batch_size, state_action_ids):
    action_ids = jax.random.randint(
        key, shape=(batch_size, 1), minval=0, maxval=config.num_actions
    )

    return jnp.concatenate([state_action_ids, action_ids], axis=1)


@MeasureTime
@partial(jax.jit, static_argnums=(1, 2))
def imagine_state(key, world_model, config, params, state_action_ids, past_key_values):
    input_ids = state_action_ids[:, -config.tokens_per_block :]
    total_seq_len = state_action_ids.shape[1]
    position_ids = jnp.broadcast_to(
        jnp.arange(total_seq_len - config.tokens_per_block, total_seq_len)[None, :],
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

    sequence = jnp.concatenate([input_ids, next_state_ids], axis=1)

    return sequence, outputs.past_key_values


@partial(jax.jit, static_argnums=(0, 1, 2))
def init_cache(world_model, config, batch_size):
    return world_model.init_cache(batch_size, config.max_tokens)


def measure_inference_time(
    seed=0,
    iterations=10,
    batch_size=48,
    rollout_horizon=20,
    tokens_per_block=82,
    max_blocks=20,
    vocab_size=4096,
):
    config = GPT2WorldModelConfig(
        num_actions=17,
        tokens_per_block=tokens_per_block,
        max_blocks=max_blocks,
        vocab_size=vocab_size,
        n_positions=tokens_per_block * max_blocks,
        n_embd=128,
        n_layer=3,
        n_head=8,
        n_inner=None,  # defaults to 4 * n_embd
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )

    input_shape = (batch_size, config.max_tokens)
    world_model = FlaxGPT2WorldModelModule(config)

    key = jax.random.PRNGKey(seed)
    key, init_weights_key, initial_state_key, imagine_key = jax.random.split(key, 4)
    params = world_model.init_weights(init_weights_key, input_shape)

    input_ids = generate_initial_state(initial_state_key, config, batch_size)
    outputs = imagine_state(
        imagine_key, world_model, config, params, input_ids, past_key_values=None
    )
    jax.block_until_ready(outputs)

    MeasureTime.reset()

    for _ in tqdm.trange(iterations):
        key, initial_state_key = jax.random.split(key)
        sequence = generate_initial_state(initial_state_key, config, batch_size)
        past_key_values = init_cache(world_model, config, batch_size)

        for t in range(rollout_horizon):
            key, imagine_key, action_key = jax.random.split(key, 3)
            sequence, past_key_values = imagine_state(
                imagine_key, world_model, config, params, sequence, past_key_values
            )
            sequence = generate_action(action_key, config, batch_size, sequence)

    MeasureTime.print_stats()


if __name__ == "__main__":
    measure_inference_time()
