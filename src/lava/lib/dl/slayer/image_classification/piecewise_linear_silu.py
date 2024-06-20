from torch import nn

class PiecewiseLinearSiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        x = x.clamp(-5.)
        x[x<-0.5] = (x[x<-0.5] + 5) * -0.1
        return x


class QuantizedPiecewiseLinearSiLU(nn.Module):
    def __init__(self, 
                 inp_exp=8,
                 weight_exp=6,
                 *args, 
                 **kwargs):
        super().__init__()
        self.inp_exp = inp_exp
        self.weight_exp = weight_exp
        exp = inp_exp + weight_exp 
        self.exp = exp
        self.m01 = int(-0.1 * 2**exp) / 2**exp
        self.m05 = int(-0.5 * 2**exp) / 2**exp
        self.m5 = int(-5. * 2**exp) / 2**exp
        self.p5 = int(5 * 2**exp) / 2**exp

    def forward(self, x):
        x = x.clamp(self.m5)
        x[x<self.m05] = (x[x<self.m05] + self.p5) * self.m01
        return ((x*2**self.inp_exp).int() / 2**self.inp_exp).float()

