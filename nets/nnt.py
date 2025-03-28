from flax import nnx
import jax.numpy as jnp


class NearestNeighborTokenizer:
    def __init__(self, codebook_size: int = 4096):
        self.codebook_size = codebook_size
        self.patch_size = 7
        self.grid_size = 9
        self.threshold = 0.75

    def __call__(self, x, codebook, codebook_size):
        *_, H, W, C = x.shape

        x = x.reshape(
            -1, self.grid_size, self.patch_size, self.grid_size, self.patch_size, C
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.patch_size, self.patch_size, C)

        def update_codebook(codebook_info, patch):
            codebook, current_size = codebook_info
            diff = codebook - patch
            diff = jnp.square(diff).sum(axis=(-3, -2, -1))

            should_update = (diff > self.threshold).all()
            codebook = jnp.where(should_update, codebook.at[current_size].set(patch), codebook)
            current_size += should_update

            return (codebook, current_size), None

        (codebook, codebook_size), _ = nnx.scan(
            update_codebook,
            in_axes=(nnx.transforms.iteration.Carry, 0),
        )((codebook, codebook_size), x)


        diff = x[:, None] - codebook[None]
        diff = jnp.square(diff).sum(axis=(-3, -2, -1))
        idx = jnp.argmin(diff, axis=-1)

        idx = idx.reshape(-1, self.grid_size, self.grid_size)
        return idx, codebook, codebook_size
    
    def decode(self, x, codebook):
        *_, H, W = x.shape
        
        x = jnp.take(codebook, x, axis=0)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.grid_size * self.patch_size, self.grid_size * self.patch_size, x.shape[-1])

        return x
