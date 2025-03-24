import jax

from flax import nnx

from nets.impala_cnn import ImpalaCNN


def test_impala_cnn():
    cnn = ImpalaCNN(channels=[64, 64, 128], rngs=nnx.Rngs(0))

    import jax.numpy as jnp

    x = jnp.ones((16, 63, 63, 3))
    y = cnn(x)

    assert y.shape == (16, 8, 8, 128)

def test_nearest_neighbor_tokenizer():
    from nets.nnt import NearestNeighborTokenizer
    from craftax import craftax_env
    from env.wrapper import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper

    nnt = NearestNeighborTokenizer(codebook_size=4096)


    env = craftax_env.make_craftax_env_from_name(
        "Craftax-Classic-Pixels-v1", auto_reset=True
    )
    env_params = env.default_params

    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, 48)

    env_rng = jax.random.PRNGKey(0)
    curr_obs, env_state = env.reset(env_rng, env_params)
    y = nnt(curr_obs)
    import ipdb
    ipdb.set_trace()

    assert y.shape == (16, 9, 9)

def main():
    # test_impala_cnn()
    test_nearest_neighbor_tokenizer()

    print("All tests passed!")


if __name__ == "__main__":
    main()
