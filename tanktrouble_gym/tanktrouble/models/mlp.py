# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F

# DEVICE = th.device("cuda:0") if th.cuda.is_available() else th.device("cpu")

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, 256, device=DEVICE)
#         self.linear2 = nn.Linear(256, 32, device=DEVICE)
#         self.linear3 = nn.Linear(32, output_dim, device=DEVICE)

#     def forward(self, x):
#         print(x, self.linear1)
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
#         x = F.relu(x)
#         x = self.linear3(x)
#         x = F.relu(x)
#         x = F.log_softmax(x)

#         return x
