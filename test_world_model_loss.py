from functools import partial

import jax

from nets.world_model import FlaxGPT2WorldModel
from nets.configuration import GPT2WorldModelConfig


@partial(jax.jit, static_argnums=(0,))
def loss_fn(world_model, params, dropout_key, state_action_ids, rewards, terminations):
    return world_model.loss(
        params, dropout_key, state_action_ids, rewards, terminations
    )


def main(seed=0, batch_size=48, tokens_per_block=82, max_blocks=20, vocab_size=4096):
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
    world_model = FlaxGPT2WorldModel(config, input_shape, seed)

    key = jax.random.PRNGKey(seed)
    key, init_weights_key, state_ids_key = jax.random.split(key, 3)

    params = world_model.init_weights(init_weights_key, input_shape)

    state_action_ids = jax.random.randint(
        state_ids_key,
        shape=(batch_size, (config.max_blocks + 1) * config.tokens_per_block),
        minval=0,
        maxval=vocab_size,
    )
    for block_idx in range(config.max_blocks):
        block_end_idx = (block_idx + 1) * config.tokens_per_block - 1
        key, action_key = jax.random.split(key)
        action_ids = jax.random.randint(
            action_key, shape=(batch_size,), minval=0, maxval=config.num_actions
        )
        state_action_ids = state_action_ids.at[:, block_end_idx].set(action_ids)

    key, reward_key, termination_key, dropout_key = jax.random.split(key, 4)
    rewards = jax.random.randint(
        reward_key, shape=(batch_size, config.max_blocks), minval=0, maxval=2
    )
    terminations = jax.random.randint(
        termination_key, shape=(batch_size, config.max_blocks), minval=0, maxval=2
    )
    loss = loss_fn(
        world_model, params, dropout_key, state_action_ids, rewards, terminations
    )
    print(loss)


if __name__ == "__main__":
    main()
