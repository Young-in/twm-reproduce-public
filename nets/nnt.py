import jax.numpy as jnp


class NearestNeighborTokenizer:
    def __init__(self, codebook_size: int = 4096):
        self.codebook_size = codebook_size
        self.patch_size = 7
        self.grid_size = 9
        self.threshold = 0.75

        self.codebook = (
            jnp.zeros((codebook_size, self.patch_size, self.patch_size, 3)) - 1
        )  # -1 for preventing minimum distance of unassigned patches
        self.current_size = 0

    def __call__(self, x):
        *_, H, W, C = x.shape

        x = x.reshape(
            -1, self.grid_size, self.patch_size, self.grid_size, self.patch_size, C
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.patch_size, self.patch_size, C)

        diff = x[:, None] - self.codebook[None]
        diff = jnp.square(diff).sum(axis=(-3, -2, -1))
        min_diff = jnp.min(diff, axis=-1)

        new_codes = []

        for i in zip(*(min_diff > self.threshold).nonzero()):
            if len(new_codes) >= self.codebook_size - self.current_size:
                break
            for code in new_codes:
                if jnp.square(code - x[i]).sum() < self.threshold:
                    break
            else:
                new_codes.append(x[i])
        
        self.codebook = self.codebook.at[self.current_size : self.current_size + len(new_codes)].set(jnp.stack(new_codes))
        self.current_size += len(new_codes)

        diff = x[:, None] - self.codebook[None]
        diff = jnp.square(diff).sum(axis=(-3, -2, -1))
        idx = jnp.argmin(diff, axis=-1)

        idx = idx.reshape(-1, self.grid_size, self.grid_size)
        return idx