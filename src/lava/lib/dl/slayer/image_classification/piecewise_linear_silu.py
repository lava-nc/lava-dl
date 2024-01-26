from torch import nn

class PiecewiseLinearSiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        x = x.clamp(-5., x.max().item())
        x[x<-0.5] = (x[x<-0.5] + 5) * -0.1
        return x