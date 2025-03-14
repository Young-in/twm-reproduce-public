from flax import nnx

from nets.impala_cnn import ImpalaCNN


def test_impala_cnn():
    cnn = ImpalaCNN(channels=[64, 64, 128], rngs=nnx.Rngs(0))

    import jax.numpy as jnp

    x = jnp.ones((16, 63, 63, 3))
    y = cnn(x)

    assert y.shape == (16, 8, 8, 128)


def main():
    test_impala_cnn()


if __name__ == "__main__":
    main()
