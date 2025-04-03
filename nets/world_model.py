from typing import Optional, Tuple

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
import flax.linen as nn
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
from jax import lax
import jax.numpy as jnp
import optax
from transformers.models.gpt2.modeling_flax_gpt2 import (
    FlaxGPT2Attention,
    FlaxGPT2Block,
    FlaxGPT2BlockCollection,
    FlaxGPT2MLP,
    FlaxGPT2Module,
    FlaxGPT2Model,
)
from transformers.models.llama.modeling_flax_llama import FlaxLlamaRotaryEmbedding

from nets.configuration import GPT2WorldModelConfig
from nets.mask import nonflex_block_causal_mask
from nets.outputs import FlaxGPT2WorldModelOutput
from nets.slicer import (
    ActionHead,
    Embedder,
    ObservationHead,
    slice_observations,
)


class FlaxGPT2BlockAttention(FlaxGPT2Attention):
    """
    Uses block attention mask and RoPE
    """

    def setup(self):
        super().setup()

        if self.causal:
            self.block_mask = nonflex_block_causal_mask(
                self.config.max_position_embeddings, self.config.tokens_per_block
            )

        self.rotary_emb = FlaxLlamaRotaryEmbedding(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        position_ids: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        if not is_cross_attention:
            qkv_out = self.c_attn(hidden_states)
            query, key, value = jnp.split(qkv_out, 3, axis=2)
        else:
            q_out = self.q_attn(hidden_states)
            (query,) = jnp.split(q_out, 1, axis=2)
            kv_out = self.c_attn(key_value_states)
            key, value = jnp.split(kv_out, 2, axis=2)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query, key = self.rotary_emb(query, key, position_ids)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.causal:
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                block_mask = lax.dynamic_slice(
                    self.block_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                block_mask = self.block_mask[:, :, :query_length, :key_length]
            block_mask = jnp.broadcast_to(
                block_mask, (batch_size,) + block_mask.shape[1:]
            )

        if self.causal:
            attention_mask = block_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key, value, attention_mask = self._concatenate_to_cache(
                key, value, query, attention_mask
            )

        # transform boolean mask into float mask
        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                    self.dtype
                ),
            )
        else:
            attention_bias = None

        # TODO use flex attention
        # usual dot product attention
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxGPT2WorldModelBlock(FlaxGPT2Block):
    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = (
            self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
        )

        self.ln_1 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.ln_2 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon, dtype=self.dtype
        )

        if self.config.add_cross_attention:
            self.crossattention = FlaxGPT2BlockAttention(
                config=self.config,
                dtype=self.dtype,
                causal=False,
                is_cross_attention=True,
            )
            self.ln_cross_attn = nn.LayerNorm(
                epsilon=self.config.layer_norm_epsilon, dtype=self.dtype
            )

        self.mlp = FlaxGPT2MLP(self.config, inner_dim, dtype=self.dtype)

        self.attn = FlaxGPT2BlockAttention(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        position_ids: jnp.ndarray,
        attention_mask=None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        Identical to parent class implementation, but passes position_ids
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = attn_outputs[0]  # output_attn: a, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[1:]
            )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(
            hidden_states, deterministic=deterministic
        )
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + outputs

        return outputs


class FlaxGPT2WorldModelBlockCollection(FlaxGPT2BlockCollection):
    def setup(self):
        self.blocks = [
            FlaxGPT2WorldModelBlock(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        position_ids: jnp.ndarray,
        attention_mask=None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Identical to parent class implementation, but passes position_ids
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                position_ids,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # this contains possible `None` values - `FlaxGPT2Module` will filter them out
        outputs = (
            hidden_states,
            all_hidden_states,
            all_attentions,
            all_cross_attentions,
        )

        return outputs


class FlaxGPT2WorldModelModule(FlaxGPT2Module):
    """
    FlaxGPT2Module with separate state and action embeddings, RoPE, and output reward + termination heads

    Based on IRIS https://github.com/eloialonso/iris/blob/24326aaaa283c527f42b89b44cfdecf2665a7a16/src/models/world_model.py#L25
    """

    config: GPT2WorldModelConfig

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.ln_f = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon, dtype=self.dtype
        )

        self.h = FlaxGPT2WorldModelBlockCollection(self.config, dtype=self.dtype)

        action_emb = nn.Embed(
            self.config.num_actions,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
        )

        self.embedder = Embedder(
            tokens_per_block=self.config.tokens_per_block,
            max_blocks=self.config.max_blocks,
            observation_embedding=self.wte,
            action_embedding=action_emb,
        )

        self.observation_head = ObservationHead(
            tokens_per_block=self.config.tokens_per_block,
            head_module=nn.Sequential(
                [
                    nn.Dense(self.embed_dim),
                    nn.relu,
                    nn.Dense(self.config.vocab_size),
                ]
            ),
        )

        self.reward_head = ActionHead(
            tokens_per_block=self.config.tokens_per_block,
            head_module=nn.Sequential(
                [
                    nn.Dense(self.embed_dim),
                    nn.relu,
                    nn.Dense(2),
                ]
            ),
        )

        self.termination_head = ActionHead(
            tokens_per_block=self.config.tokens_per_block,
            head_module=nn.Sequential(
                [
                    nn.Dense(self.embed_dim),
                    nn.relu,
                    nn.Dense(2),
                ]
            ),
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.embedder(input_ids.astype("i4"))

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            position_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        observation_logits = self.observation_head(hidden_states)
        reward_logits = self.reward_head(hidden_states)
        termination_logits = self.termination_head(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        outputs += (observation_logits, reward_logits, termination_logits)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxGPT2WorldModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
            observation_logits=observation_logits,
            reward_logits=reward_logits,
            termination_logits=termination_logits,
        )

    def _loss(self, state_action_ids, rewards, terminations):
        input_ids = state_action_ids[:, : -self.config.tokens_per_block]
        attention_mask = jnp.ones_like(input_ids)
        outputs = self(input_ids, attention_mask, deterministic=False)

        observation_labels = state_action_ids[:, self.config.tokens_per_block :]
        observation_labels = slice_observations(
            observation_labels, self.config.tokens_per_block
        )
        observation_labels = nn.one_hot(observation_labels, self.config.vocab_size)

        observation_loss = optax.softmax_cross_entropy(
            outputs.observation_logits, observation_labels
        ).sum()

        reward_labels = nn.one_hot(rewards, 2)
        reward_loss = optax.softmax_cross_entropy(
            outputs.reward_logits, reward_labels
        ).sum()

        termination_labels = nn.one_hot(terminations, 2)
        termination_loss = optax.softmax_cross_entropy(
            outputs.termination_logits, termination_labels
        ).sum()

        loss = observation_loss + reward_loss + termination_loss
        return loss

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.init(
            rngs, input_ids, attention_mask, position_ids, return_dict=False
        )

        return module_init_outputs["params"]

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
        )

        init_variables = self.init(
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return unfreeze(init_variables["cache"])

    def loss(self, params: dict, dropout_rng: jax.random.PRNGKey, *args, **kwargs):
        rngs = {"dropout": dropout_rng}

        inputs = {"params": params}
        outputs = self.apply(inputs, *args, **kwargs, method="_loss", rngs=rngs)
        return outputs
