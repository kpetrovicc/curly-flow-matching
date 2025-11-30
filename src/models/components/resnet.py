import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.SELU(inplace=True)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        # Add & activate
        return self.act(out + identity)


class ResNetMLP(nn.Module):
    def __init__(self, dim, out_dim=None, width=64, time_varying=False, num_blocks=4):
        """
        A small ResNet-style MLP.
        
        Args:
            dim (int): input feature dimension
            out_dim (int, optional): output dimension; defaults to `dim`
            width (int): hidden width for all layers
            time_varying (bool): if True, x is concatenated with a time scalar
            num_blocks (int): number of residual blocks to stack
        """
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim

        # First layer: expand to width
        self.input_fc = nn.Linear(dim + (1 if time_varying else 0), width)
        self.input_act = nn.SELU(inplace=True)

        # Residual tower
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(num_blocks)])

        # Final projection
        self.output_fc = nn.Linear(width, out_dim)

    def forward(self, x, t=None):
        """
        Args:
            x (Tensor): shape (batch, dim)
            t (Tensor, optional): shape (batch, 1) if time_varying=True
        """
        out = self.input_act(self.input_fc(x))
        for block in self.blocks:
            out = block(out)
        return self.output_fc(out)
