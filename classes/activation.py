import torch
import torch.nn as nn

# Gaussian Error Activation Function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # this is and approximate formula for the gaussian error function
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))