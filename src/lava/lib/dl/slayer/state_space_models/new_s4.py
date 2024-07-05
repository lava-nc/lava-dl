import math
from typing import Mapping, Optional, Union
import torch
from torch import nn
from einops import repeat, rearrange
from ssm_utils import dplr, nplr

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie: bool = True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

        
    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
        return X



def log_vandermonde(v, x, L):
    """
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    x = torch.log(x)
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = torch.einsum('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
    return 2*vandermonde_prod.real

def log_vandermonde_transpose(u, v, x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = torch.einsum('... l, ... n, ... n l -> ... n', u.to(x), v.to(x), vandermonde_matrix) # (... L)
    return vandermonde_prod





class S4D(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_state: int,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 rank: int = 1,
                 ssm_init: str = 'legs',
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} not implemented")
        self.dropout = dropout
        self.rank = rank
        self.ssm_init = ssm_init


              # Initialize dt, A, B, C
        inv_dt = self.init_dt()
        A, _, B, C = self.init_ssm_dplr()
        self.drop = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        self.repeat = self.d_model // A.size(0)

        # register params
        self.register_parameter("C", nn.Parameter(torch.view_as_real(C.conj().resolve_conj())))
        self.register_parameter("B", nn.Parameter(torch.view_as_real(B)))
        self.register_parameter("A_real", nn.Parameter(torch.log(-A.real)))
        self.register_parameter("A_imag", nn.Parameter(-A.imag))
        self.register_parameter("inv_dt", nn.Parameter(inv_dt))


    def forward(self, x):
        L = x.size(-1)
        k = self.compute_kernel(L=L)
        # TODO kernel dropout and y dropout
        
        k_f = torch.fft.rfft(k, n=2*L) # (C H L)
        x_f = torch.fft.rfft(x, n=2*L) # (B H L)
        y_f = torch.einsum('bhl,chl->bchl', x_f, k_f)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)

        # Squeeze second dimension
        y = rearrange(y, 'b c h l -> b (c h) l')

        # TODO too much dropout?
        y = self.drop(y)
        # TODO maybe transpose, droput better with transpose
        y = self.activation(y)

        return y
        
    def setup_step(self):
        pass

    def step(self, x, state):
        pass

    def default_state(self):
        pass

    def compute_kernel(self, L):
        dt, A, B, C = self._get_params()
        dtA = dt * A

        # Combine B and C
        C = (B[:, None, :, :] * C).view(-1, self.d_model, self.d_state)


        # Main kernel
        # Zero-order hold if you want other options you need to add them 
        dtA = torch.exp(dtA)
        C = C * (dtA-1.) / A
        K = log_vandermonde(C, dtA, L) # (H L)

        return K
    
    def _get_params(self):
        """Process the internal parameters."""
        A = -torch.exp(self.A_real) - 1j * self.A_imag
        B = torch.view_as_complex(self.B) # (1 S N)
        C = torch.view_as_complex(self.C) # (C H N)
    
        dt = torch.exp(self.inv_dt) # (H N)

        # Incorporate dt into A and B
        A = repeat(A, 't n -> (v t) n', v=self.repeat)  # (H N)
        B = repeat(B, 'b t n -> b (v t) n', v=self.repeat)  # (1 H N)

        return dt, A, B, C


    def init_dt(self, dt_min=0.001, dt_max=0.1):
        # Generate dt
        shape = (self.d_model, 1)
        # Initialize log dt
        # torch.manual_seed(1)
        inv_dt = torch.rand(*shape, dtype=torch.float) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        return inv_dt

    def init_ssm_dplr(self):
        """Returns DPLR (A, P, B, C) parameters for init options."""
        A, P, B, _ =  self.ssm()

        # Broadcast C to have H channels
        # torch.manual_seed(0)
        C = torch.randn(1, self.d_model, self.d_state, dtype=torch.cfloat)

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.d_model // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.d_model // P.size(-2)).clone().contiguous()
        A = repeat(A, 't n -> (v t) n', v=self.d_model // A.size(-2)).clone().contiguous()


           # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.d_model, self.d_state))) # (C, H, N) 
        B = B.unsqueeze(0) # (1, H, N)
        return A, P, B, C

    def ssm(self):
        """Dispatcher to create single SSM initialization
        N: state size
        R: rank (for DPLR parameterization)
        H: number of independent SSM copies
        """

        # Because these complex parameterizations assume conjugate symmetry,
        # double the value of self.d_state for convenience
        d_state = self.d_state * 2

        if self.ssm_init == "diag" or self.ssm_init == "dlpr":
            A, P, B, V = dplr(N=d_state, rank=self.rank, H=self.d_model)
        else:
            A, P, B, V = nplr(self.ssm_init, d_state, self.rank)
            A = repeat(A, 'n -> s n', s=self.d_model)
            P = repeat(P, 'r n -> r s n', s=self.d_model)
            B = repeat(B, 'n -> s n', s=self.d_model)
            V = repeat(V, 'n m -> s n m', s=self.d_model)
        return A, P, B, V

