# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Resonator dynamics."""

import os
import torch
from torch.utils.cpp_extension import load

from ...utils.int_utils import right_shift_to_zero
from ...utils.utils import staticproperty
from ... import jitconfig


class Accelerated:
    # This provides a static handle to jit cuda accelerated module
    # They are callable as
    # Accelerated.resonator.dynamics(...) -> handles autograd
    # Accelerated.resonator.fwd(...)
    # Accelerated.resonator.bwd(...)
    module = None

    @staticproperty
    def resonator():
        if Accelerated.module is None:
            if jitconfig.VERBOSE is True:
                print(
                    'Resonator accelerated module does not exist. '
                    'Initializing with JIT compilation.'
                )
            if not torch.cuda.is_available():
                raise Exception(
                    'CUDA acceleration of Resonator failed. '
                    'CUDA is not available in the system.'
                )
            if jitconfig.TORCH_CUDA_ARCH_LIST is not None:
                os.environ['TORCH_CUDA_ARCH_LIST'] = \
                    jitconfig.TORCH_CUDA_ARCH_LIST
            Accelerated.module = load(
                name='dynamics',
                sources=[
                    os.path.dirname(os.path.abspath(__file__))
                    + '/resonator.cu'
                ],
            )
        return Accelerated.module


def dynamics(
    real_input, imag_input,
    sin_decay, cos_decay,
    real_state, imag_state,
    w_scale, threshold=None, debug=False
):
    """Resonator dynamics. It automatically switches between CUDA and CPU
    implementation depending on available hardware.

    .. math::
        \\mathfrak{Re}(z[t]) &= (1-\\alpha)(\\cos\\phi\\ \\mathfrak{Re}(z[t-1])
            - \\sin\\phi\\ \\mathfrak{Im}(z[t-1]))
            + \\mathfrak{Re}(x[t])\\

        \\mathfrak{Im}(z[t]) &= (1-\\alpha)(\\sin\\phi\\ \\mathfrak{Re}(z[t-1])
            + \\cos\\phi\\ \\mathfrak{Im}(z[t-1]))
            + \\mathfrak{Im}(x[t])

    Note
    ----
    If threshod is not None, it implements Izhikevich spike and reset dynamics.

    .. math::
        s[t] &= \\mathfrak{Im}(z[t]) \\geq \\vartheta \\

        \\mathfrak{Re}(z[t]) &= 0

    Parameters
    ----------
    real_input : torch tensor
        real input tensor.
    imag_input : torch tensor
        imaginary input tensor.
    sin_decay : torch tensor
        sin decay tensor. Note: it is unscaled integer value here.
    cos_decay : torch tensor
        cos decay tensor. Note: it is unscaled integer value here.
    real_state : torch tensor
        real dynamics state.
    imag_state : torch tensor
        imaginary dyanmics state.
    w_scale : int
        parameter scaling for integer calculations.
    threshold : float or None, optional
        threshold for spiking dynamics. Default is None.
    debug : bool, optional
        enable/disable debug mode. Default is False.

    Returns
    -------
    torch tensor
        real dynamics.
    torch tensor
        imaginary dynamics.
    """
    if threshold is None:
        threshold = -1   # -1 means no reset mechanism
    _ResDynamics.DEBUG = debug

    if torch.numel(real_state) == 1:
        real_state = real_state * torch.ones(
            real_input.shape[:-1]
        ).to(real_input.device)
    if torch.numel(imag_state) == 1:
        imag_state = imag_state * torch.ones(
            imag_input.shape[:-1]
        ).to(imag_input.device)

    if real_input.is_cuda is False or debug is True:
        real, imag = _ResDynamics.apply(
            real_input, imag_input,
            sin_decay, cos_decay,
            real_state, imag_state,
            threshold, w_scale
        )
    else:
        real, imag = Accelerated.resonator.dynamics(
            real_input.contiguous(), imag_input.contiguous(),
            sin_decay.contiguous(), cos_decay.contiguous(),
            real_state.contiguous(), imag_state.contiguous(),
            threshold, w_scale
        )

    return real, imag


def persistent_real_state(real_state, spike):
    """Handles refractory state changes due to spike in last time bin.

    Parameters
    ----------
    real_state : torch tensor
        real state in last time bin.
    spike : torch tensor
        spike state in last time bin.

    Returns
    -------
    torch tensor
        persistent real state to store for next time.
    """
    spike = (spike > 0).to(real_state.dtype)
    return real_state * (1 - spike)


def persistent_imag_state(imag_state, spike, threshold, scale):
    """Handles refractory state change due to spike in last time bin.

    Parameters
    ----------
    imag_state : torch tensor
        imaginary sate in last time bin.
    spike : torch tensor
        spike state in last time bin.
    threshold : float
        threshold value.
    scale : int
        parameter scaling value.

    Returns
    -------
    torch tensor
        persistent state to store for next time.
    """
    spike = (spike > 0).to(imag_state.dtype)
    return spike * (threshold - 1 / scale) + (1 - spike) * imag_state


class _ResDynamics(torch.autograd.Function):
    DEBUG = False

    @staticmethod
    def forward(
        ctx, real_input, imag_input,
        sin_decay, cos_decay,
        real_state, imag_state,
        threshold, w_scale
    ):
        """ """
        real, imag = _res_dynamics_fwd(
            real_input, imag_input,
            sin_decay, cos_decay,
            real_state, imag_state,
            threshold, w_scale, dtype=torch.int64
        )

        if _ResDynamics.DEBUG is True and real_input.is_cuda is True:
            _real, _imag = Accelerated.resonator.fwd(
                real_input, imag_input,
                sin_decay, cos_decay,
                real_state, imag_state,
                threshold, w_scale
            )
            # print('Fwd Checking')
            for i in range(real.shape[1]):
                if torch.norm(real[0, i] - _real[0, i]) > 1e-6:
                    print('real:', i, torch.norm(real[0, i] - _real[0, i]))
                    print(real[0, i, :50] * w_scale)
                    print(_real[0, i, :50] * w_scale)
                    print('imag:', i)
                    print(imag[0, i, :50] * w_scale)
                    print('real_input:', i)
                    print(real_input[0, i, :50] * w_scale)
                    print('imag_input:', i)
                    print(imag_input[0, i, :50] * w_scale)
                    print(
                        torch.norm(real - _real),
                        real[real != _real] * w_scale,
                        _real[real != _real] * w_scale
                    )
                    raise Exception
                if torch.norm(imag[0, i] - _imag[0, i]) > 1e-6:
                    print('real:', i)
                    print(real[0, i, :50] * w_scale)
                    print('imag:', i, torch.norm(imag[0, i] - _imag[0, i]))
                    print(imag[0, i, :50] * w_scale)
                    print(_imag[0, i, :50] * w_scale)
                    print('real_input:', i)
                    print(real_input[0, i, :50] * w_scale)
                    print('imag_input:', i)
                    print(imag_input[0, i, :50] * w_scale)
                    print(
                        torch.norm(imag - _imag),
                        imag[imag != _imag] * w_scale,
                        _imag[imag != _imag] * w_scale
                    )
                    raise Exception

        ctx.save_for_backward(real, imag, sin_decay, cos_decay)

        return real, imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        """ """
        real, imag, sin_decay, cos_decay = ctx.saved_tensors

        grad_real_input, grad_imag_input, grad_sin_decay, grad_cos_decay = \
            _res_dynamics_bwd(
                grad_real, grad_imag, real, imag, sin_decay, cos_decay
            )

        if _ResDynamics.DEBUG is True and grad_real.is_cuda is True:
            _grad_real_input, _grad_imag_input, \
                _grad_sin_decay, _grad_cos_decay = Accelerated.resonator.bwd(
                    grad_real, grad_imag, real, imag, sin_decay, cos_decay
                )

            # print('Bwd Checking')
            for i in range(grad_real_input.shape[1]):
                if torch.norm(
                    grad_real_input[0, i] - _grad_real_input[0, i]
                ) / torch.numel(grad_real_input[0, i]) > 1e-6:
                    print('grad_real_input:', i, torch.norm(
                        grad_real_input[0, i] - _grad_real_input[0, i]
                    ))
                    print(grad_real_input[0, i, -50:])
                    print(_grad_real_input[0, i, -50:])
                    print('grad_imag_input:', i)
                    print(grad_imag_input[0, i, -50:])
                    print(_grad_imag_input[0, i, -50:])
                    print('grad_real:', i)
                    print(grad_real[0, i, -50:])
                    print('grad_imag:', i)
                    print(grad_imag[0, i, -50:])
                    print(
                        torch.norm(grad_real_input - _grad_real_input),
                        grad_real_input[grad_real_input != _grad_real_input],
                        _grad_real_input[grad_real_input != _grad_real_input]
                    )
                    raise Exception
                if torch.norm(
                    grad_imag_input[0, i] - _grad_imag_input[0, i]
                ) / torch.numel(grad_imag_input[0, i]) > 1e-6:
                    print('grad_real_input:', i)
                    print(grad_real_input[0, i, -50:])
                    print('grad_imag_input:', i, torch.norm(
                        grad_imag_input[0, i] - _grad_imag_input[0, i]
                    ))
                    print(grad_imag_input[0, i, -50:])
                    print(_grad_imag_input[0, i, -50:])
                    print('grad_real:', i)
                    print(grad_real[0, i, -50:])
                    print('grad_imag:', i)
                    print(grad_imag[0, i, -50:])
                    print(
                        torch.norm(grad_imag_input - _grad_imag_input),
                        grad_imag_input[grad_imag_input != _grad_imag_input],
                        _grad_imag_input[grad_imag_input != _grad_imag_input]
                    )
                    raise Exception
                if torch.norm(
                    grad_sin_decay - _grad_sin_decay
                ) / torch.numel(grad_sin_decay) > 1e-2:
                    print('grad_sin_decay:', i, torch.norm(
                        grad_sin_decay - _grad_sin_decay
                    ))
                    print(grad_sin_decay[:50])
                    print(_grad_sin_decay[:50])
                    print(
                        torch.norm(grad_sin_decay - _grad_sin_decay),
                        grad_sin_decay[grad_sin_decay != _grad_sin_decay],
                        _grad_sin_decay[grad_sin_decay != _grad_sin_decay],
                    )
                    raise Exception
                if torch.norm(
                    grad_cos_decay - _grad_cos_decay
                ) / torch.numel(grad_cos_decay) > 1e-2:
                    print('grad_cos_decay:', i, torch.norm(
                        grad_cos_decay - _grad_cos_decay
                    ))
                    print(grad_cos_decay[:50])
                    print(_grad_cos_decay[:50])
                    print(
                        torch.norm(grad_cos_decay - _grad_cos_decay),
                        grad_cos_decay[grad_cos_decay != _grad_cos_decay],
                        _grad_cos_decay[grad_cos_decay != _grad_cos_decay],
                    )
                    raise Exception

        return grad_real_input, grad_imag_input, \
            grad_sin_decay, grad_cos_decay, None, None, None, None


def _res_dynamics_fwd(
    real_input, imag_input,
    sin_decay, cos_decay,
    real_state, imag_state,
    threshold, w_scale, dtype=torch.int32
):
    """ """
    dtype = torch.int64
    device = real_state.device

    real_old = (real_state * w_scale).clone().detach().to(dtype).to(device)
    imag_old = (imag_state * w_scale).clone().detach().to(dtype).to(device)

    sin_decay_int = (sin_decay).clone().detach().to(dtype).to(device)
    cos_decay_int = (cos_decay).clone().detach().to(dtype).to(device)

    real = torch.zeros_like(real_input)
    imag = torch.zeros_like(imag_input)

    threshold *= w_scale

    num_steps = real_input.shape[-1]
    for n in range(num_steps):
        real_new = right_shift_to_zero(cos_decay_int * real_old, 12) \
            - right_shift_to_zero(sin_decay_int * imag_old, 12) \
            + (w_scale * real_input[..., n]).to(dtype)
        imag_new = right_shift_to_zero(sin_decay_int * real_old, 12) \
            + right_shift_to_zero(cos_decay_int * imag_old, 12) \
            + (w_scale * imag_input[..., n]).to(dtype)

        if threshold >= 0:
            spike_new = (imag_new >= threshold).to(dtype)
            real_old = ((1 - spike_new) * real_new).to(dtype)
            imag_old = (
                spike_new * (threshold - 1) + (1 - spike_new) * imag_new
            ).to(dtype)
        else:
            real_old = real_new
            imag_old = imag_new

        real[..., n] = real_new / w_scale
        imag[..., n] = imag_new / w_scale

    return real, imag


def _res_dynamics_bwd(grad_real, grad_imag, real, imag, sin_decay, cos_decay):
    """ """
    grad_real_input = torch.zeros_like(grad_real)
    grad_imag_input = torch.zeros_like(grad_imag)

    # cos_decay /= (1 << 12)
    # sin_decay /= (1 << 12)
    cos_decay = cos_decay / (1 << 12)
    sin_decay = sin_decay / (1 << 12)

    num_steps = grad_real.shape[-1]

    grad_real_input[..., num_steps - 1] = grad_real[..., num_steps - 1]
    grad_imag_input[..., num_steps - 1] = grad_imag[..., num_steps - 1]

    for n in range(num_steps - 1)[::-1]:
        grad_real_input[..., n] = cos_decay * grad_real_input[..., n + 1] \
            + sin_decay * grad_imag_input[..., n + 1] + grad_real[..., n]
        grad_imag_input[..., n] = -sin_decay * grad_real_input[..., n + 1] \
            + cos_decay * grad_imag_input[..., n + 1] + grad_imag[..., n]

        # TODO: this has to be commented
        if (
            torch.sum(torch.isnan(grad_real_input[..., n])) > 0
            or torch.sum(torch.isnan(grad_imag_input[..., n])) > 0
            or torch.sum(torch.isinf(grad_real_input[..., n])) > 0
            or torch.sum(torch.isinf(grad_imag_input[..., n])) > 0
        ):
            print(f'n={n}')
            print(cos_decay[:12], sin_decay[:12])
            print((cos_decay**2 + sin_decay**2)[:12])

            print(grad_real_input[..., n + 1])
            print(grad_imag_input[..., n + 1])
            print(grad_real[..., n])
            print(grad_imag[..., n])

            print(grad_real_input[..., n + 2])
            print(grad_imag_input[..., n + 2])
            print(grad_real[..., n + 1])
            print(grad_imag[..., n + 1])

            raise Exception('Stopping here.')

    if torch.sum(torch.isnan(grad_real)) > 0:
        raise Exception('grad_real has NaN.')
    if torch.sum(torch.isnan(grad_imag)) > 0:
        raise Exception('grad_imag has NaN.')

    if torch.sum(torch.isnan(grad_real_input)) > 0:
        raise Exception('grad_real_input has NaN.')
    if torch.sum(torch.isnan(grad_imag_input)) > 0:
        raise Exception('grad_imag_input has NaN.')

    grad_sin_decay = -grad_real_input[..., 1:] * imag[..., :-1] \
        + grad_imag_input[..., 1:] * real[..., :-1]
    grad_cos_decay = grad_real_input[..., 1:] * real[..., :-1] \
        + grad_imag_input[..., 1:] * imag[..., :-1]

    if torch.sum(torch.isnan(grad_sin_decay)) > 0:
        raise Exception('grad_sin_decay has NaN.')
    if torch.sum(torch.isnan(grad_cos_decay)) > 0:
        raise Exception('grad_cos_decay has NaN.')

    if torch.numel(cos_decay) == 1:  # shared parameters
        grad_sin_decay = torch.sum(
            grad_sin_decay.flatten(),
            dim=0,
            keepdim=True,
        )
        grad_cos_decay = torch.sum(
            grad_cos_decay.flatten(),
            dim=0,
            keepdim=True,
        )
    else:  # sum across batch and time dimension
        grad_sin_decay = torch.sum(grad_sin_decay, dim=[0, -1])
        grad_cos_decay = torch.sum(grad_cos_decay, dim=[0, -1])
        if len(grad_sin_decay.shape) != 1:
            grad_sin_decay = torch.sum(
                grad_sin_decay.reshape(grad_sin_decay.shape[0], -1),
                dim=1,
            )
            grad_cos_decay = torch.sum(
                grad_cos_decay.reshape(grad_cos_decay.shape[0], -1),
                dim=1,
            )

    return grad_real_input, grad_imag_input, grad_sin_decay, grad_cos_decay
