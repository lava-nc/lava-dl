from logging import log
import torch
import numpy as np
from einops import repeat
import math


def transition(measure, N, **measure_args):
    """A, B transition matrices for different measures.

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Legendre (translated)
    if measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure in ['fourier', 'fout']:
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal."""

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
        P *= 2**(-0.5) # Halve the rank correct just like the original matrix was halved
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
    else: raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P

def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True, B_clip=2.0):
    """Constructs NPLR form of HiPPO matrices.

    Returns w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B

    measure: Name of HiPPO method.
    N: Size of recurrent A matrix (also known as `d_state` elsewhere).
    dtype: Single or double precision.
    diagonalize_precision: Calculate diagonalization in double precision.
    B_clip: Clip values of B, can help with stability. None for no clipping.
    """

    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype) # (r N)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0,0]*torch.eye(N))**2) / N) > 1e-5: # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)


    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    W_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    W_im, V = torch.linalg.eigh(AP*-1j) # (..., N) (..., N, N)
    if diagonalize_precision: W_im, V = W_im.to(cdtype), V.to(cdtype)
    W = W_re + 1j * W_im
    # Check: V W V^{-1} = A
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))


    # Only keep half of each conjugate pair
    _, idx = torch.sort(W.imag)
    W_sorted = W[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, :N//2]
    W = W_sorted[:N//2]  # Only keep negative imaginary components
    #assert W[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if W[-1].abs() < 1e-4:
        V[:, -1] = 0.
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2)
    if ((err := torch.sum((2*_AP.real-AP)**2)/N) > 1e-5):
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    # C = initial_C(measure, N, dtype=dtype)
    B = torch.einsum('ij, j -> i', V_inv, B.to(V)) # V^* B
    # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
    P = torch.einsum('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P

    if B_clip is not None:
        B = B.real + 1j*torch.clamp(B.imag, min=-B_clip, max=B_clip)

    # W represents the imaginary part of the DPLR form: A = W - PP^*
    # Downstream classes just call this A for simplicity,
    # which is also more consistent with the diagonal case
    return W, P, B, V

def dplr(
    init='hippo',
    N=64, rank=1, H=1,
    dtype=torch.float,
    real_random=False,
    real_scale=1.0,
    imag_random=False,
    imag_scale=1.0,
    B_random=False,
    B_init='constant',
    B_scale=1.0,
    P_scale=1.0,
    normalize=False,
):
    """Directly construct a DPLR matrix.

    Args:
    - init: (str) ['rand', 'lin', inv', 'real', 'hippo'] Choices for initialization of A.
          Most of these affect the imaginary part of A, except for 'real'.
    - real_random: (bool) Initialize A.real in -U[0, 1]. Otherwise, initialize to -1/2.
    - real_scale: (float) Scaling factor of real part of A.
    - imag_random: (bool) Initialize A.imag randomly.
    - imag_scale: (bool) Scaling factor of imaginary part of A.
    - B_init: (str) ['constant' | 'random' | 'alternating' | 'unit-cw' | 'unit-ccw' | 'hippo']
          Choices for initialization of B.
    - B_scale: (float) Scaling factor for B
    - P_scale: (float) Scaling factor for P
    - normalize: (bool) Apply an automatic normalization factor on B
    """
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)

    # Construct real part of diagonal A (must be non-negative)
    if real_random:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    real_part = real_scale * real_part

    # Construct imaginary part of diagonal A (must be non-negative)
    if imag_random:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    if init in ['random', 'rand']:
        imag_part = torch.exp(torch.randn(H, N//2))
    elif init == 'real':
        imag_part = 0 * imag_part
        if real_random:
            real_part = torch.rand(H, N//2) * N//2
        else:
            # This is the S4D-Real method described in the S4D paper
            # The A matrix is diag(-1, -2, ..., -N), which are the eigenvalues of the HiPPO matrix
            real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif init in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif init in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif init in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif init in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2
    elif init in ['legs', 'hippo']:
        A, _, _, _ = nplr('legs', N)
        imag_part = -A.imag  # Positive
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part

    # Construct diagonal A
    A = -real_part - 1j * imag_part  # Force negative real and imag
    assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)  # Allow some tolerance for numerical precision on real part

    # Initialize B
    if B_random:
        log.warning("'B_random' is deprecated in favor of B_init='random' and will be deprecated in a future version.")
    if init in ['legs', 'hippo']:
        log.info(f'Initializing with S4D-LegS and ignoring argument {B_init=}')
        # Special initialization using the HiPPO B matrix
        # Note that theory (from S4D paper) says that B should be halved
        # to match DPLR but we drop this 0.5 factor for simplicity
        _, P, B, _ = nplr('legs', N, B_clip=2.0)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'constant':
        B = torch.ones(H, N//2, dtype=dtype)
    elif B_init == 'random':
        B = torch.randn(H, N//2, dtype=dtype)
    elif B_init == 'alternating':  # Seems to track 'constant' exactly for some reason
        B = torch.ones(H, N//4, 2, dtype=dtype)
        B[:, :, 1] *= -1
        B = B.view(H, N//2)
    elif B_init == 'unit-cw':
        z = torch.tensor(torch.exp(-2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'unit-ccw':
        z = torch.tensor(torch.exp(2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    else: raise NotImplementedError
    B *= B_scale

    # Experimental feature that appeared in earlier versions of HTTYH (not extensively tested)
    # Seems more principled for normalization theoretically, but seemed to hurt on PathX
    if normalize:
        norm = -B/A # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    # Initialize P
    if B_init in ['legs', 'hippo']:
        # P constructed earlier
        P = repeat(P, 'r n -> r h n', h=H).clone().contiguous()
    else:
        P = torch.randn(rank, H, N//2, dtype=dtype)
        P = P * P_scale

    # Initialize V (only used in testing)
    V = torch.eye(N, dtype=dtype)[:, :N//2]
    V = repeat(V, 'n m -> h n m', h=H)

    return A, P, B, V

