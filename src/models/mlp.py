import torch as th
import torch.nn as nn
import torch.nn.functional as F

DEVICE = th.device("cuda:0") if th.cuda.is_available() else th.device("cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256, device=DEVICE)
        self.linear2 = nn.Linear(256, 32, device=DEVICE)
        self.linear3 = nn.Linear(32, output_dim, device=DEVICE)

        self.log_Z = th.zeros(1, requires_grad=True, device=DEVICE)

    def forward(self, x, _):
        x = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = F.softmax(x)

        return None, x
    
    def sample_actions(self, env, _):
        states = th.from_numpy(env.obs()).unsqueeze(0).to(DEVICE)
        masks = th.from_numpy(env.get_masks()).float().to(DEVICE)

        masks = masks.unsqueeze(0)
        with th.no_grad():
            _, policy = self(states, _)
            return (masks * policy + (1 - masks) * -(1e9)).argmax(dim=1).unsqueeze(1)