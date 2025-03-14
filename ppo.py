from flax import nnx

class Actor(nnx.Module):
    def __init__(self, input_dim: int, intermediate_dim: int,  num_actions: int, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(
            num_features=input_dim,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=input_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=num_actions,
            rngs=rngs,
        )
    
    def __call__(self, x):
        x = self.norm1(x)
        x = nnx.relu(self.linear1(x))
        x = x + nnx.relu(self.linear2(x))
        x = x + nnx.relu(self.linear3(x))
        x = self.norm2(x)
        x = self.linear4(x)
        return x


class Critic(nnx.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(
            num_features=input_dim,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=input_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=1,
            rngs=rngs,
        )
    
    def __call__(self, x):
        x = self.norm1(x)
        x = nnx.relu(self.linear1(x))
        x = x + nnx.relu(self.linear2(x))
        x = x + nnx.relu(self.linear3(x))
        x = self.norm2(x)
        x = self.linear4(x)
        return x
        
class ActorCritic(nnx.Module):
    def __init__(self, input_dim: int, num_actions: int, rngs: nnx.Rngs):
        intermediate_dim = 2048

        self.actor = Actor(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            num_actions=num_actions,
            rngs=rngs,
        )
        self.critic = Critic(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            rngs=rngs,
        )

def main():
    model = ActorCritic(input_dim=8192 + 256, num_actions= 10, rngs=nnx.Rngs(0))

if __name__ == "__main__":
    main()