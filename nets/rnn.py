from flax import nnx


class RNN(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        self.rnn_cell = nnx.GRUCell(
            in_features=in_features,
            hidden_features=hidden_features,
            rngs=rngs,
        )

    def __call__(self, prev_state, z):
        prev_state, y = nnx.scan(
            self.rnn_cell,
            in_axes=(nnx.transforms.iteration.Carry, 1),
            out_axes=(nnx.transforms.iteration.Carry, 1),
        )(prev_state, z)

        return prev_state, y
    
    def initialize_carry(self, batch_size):
        return self.rnn_cell.initialize_carry((batch_size, self.hidden_features))
