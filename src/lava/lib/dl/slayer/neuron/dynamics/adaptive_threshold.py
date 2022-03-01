# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Adaptive Threshold and Refractory dynamics"""

import os
import torch
from torch.utils.cpp_extension import load

from ...utils.int_utils import right_shift_to_zero
from ...utils.utils import staticproperty
from ... import jitconfig


class Accelerated:
    """ """
    # This provides a static handle to jit cuda accelerated module
    # They are callable as
    # Accelerated.adaptive_threshold.dynamics(...) -> handles autograd
    # Accelerated.adaptive_threshold.fwd(...)
    # Accelerated.adaptive_threshold.bwd(...)
    module = None

    @staticproperty
    def adaptive_threshold():
        """ """
        if Accelerated.module is None:
            if jitconfig.VERBOSE is True:
                print(
                    'Adaptive Threshold accelerated module does not exist. '
                    'Initializing with JIT compilation.'
                )
            if not torch.cuda.is_available():
                raise Exception(
                    'CUDA acceleration of Adaptive Threshold failed. '
                    'CUDA is not available in the system.'
                )
            if jitconfig.TORCH_CUDA_ARCH_LIST is not None:
                os.environ['TORCH_CUDA_ARCH_LIST'] = \
                    jitconfig.TORCH_CUDA_ARCH_LIST
            Accelerated.module = load(
                name='dynamics',
                sources=[
                    os.path.dirname(os.path.abspath(__file__))
                    + '/adaptive_threshold.cu'
                ],
            )
        return Accelerated.module


def dynamics(
    input,
    ref_state, ref_decay, th_state,
    th_decay, th_scale, th0, w_scale, debug=False
):
    """Adaptive Threshold and Refractory Dynamics. It automatically switches
    between CUDA and CPU implementation depending on available hardware.

    .. math::
        \\vartheta[t] &= (1-\\alpha_{\\vartheta})\\,(\\vartheta[t-1] -
                         \\vartheta_0) + \\vartheta_0 \\

        r[t] &= (1-\\alpha_r)\\,r[t-1] \\

        s[t] &= (x[t] - r[t]) \\geq \\vartheta[t] \\

        r[t] &= r[t] + 2\\,\\vartheta[t] \\

        \\vartheta[t] &= \\vartheta[t] + \\vartheta_{\\text{step}}

    Parameters
    ----------
    input : torch tensor
        input tensor. It assumes leaky dynamics is already applied.
    ref_state : torch tensor
        refractory state.
    ref_decay : torch tensor
        refractory decay. Note: it is unscaled integer value here.
    th_state : torch tensor
        threshold state.
    th_decay : torch tensor
        threshold decay. Note: it is unscaled integer value here.
    th_scale : float
        threshold step after spike.
    th0 : float
        stable threshold value.
    w_scale : int
        parameter scaling for integer calculations.
    debug : bool, optional
        enable/disable debug mode. Default is False.

    Returns
    -------
    threshold : torch tensor
        adaptive threshold dynamics.
    refractory : torch tensor
        refractory dynamics.
    """
    _ATHDynamics.DEBUG = debug

    if torch.numel(ref_state) == 1:
        ref_state = ref_state * torch.ones(
            input.shape[:-1]
        ).to(input.device)
    if torch.numel(th_state) == 1:
        th_state = th_state * torch.ones(
            input.shape[:-1]
        ).to(input.device)

    if input.is_cuda is False or debug is True:
        threshold, refractory = _ATHDynamics.apply(
            input,
            ref_state, ref_decay,
            th_state, th_decay, th_scale, th0,
            w_scale
        )
    else:
        threshold, refractory = Accelerated.adaptive_threshold.dynamics(
            input.contiguous(),
            ref_state.contiguous(), ref_decay.contiguous(),
            th_state.contiguous(), th_decay.contiguous(),
            th_scale, th0,
            w_scale
        )

    return threshold, refractory


reset_persistent_ref_state = True
reset_persistent_th_state = True


def persistent_ref_state(ref_state, spike, th_state):
    """Handles refractory state changes due to spike in the last time bin.

    Parameters
    ----------
    ref_state : torch tensor
        refractory state in last time bin.
    spike : torch tensor
        spike state in last time bin.
    th_state : torch tensor
        threshold state in last time bin.

    Returns
    -------
    torch tensor
        persistent refractory state to store for next time.
    """
    spike = (spike > 0).to(ref_state.dtype)
    return ref_state + 2 * spike * th_state


def persistent_th_state(th_state, spike, th_step):
    """Handles threshold state changes due to spike in last time bin.

    Parameters
    ----------
    th_state : torch tensor
        threshold state in last time bin.
    spike : torch tensor
        spike state in last time bin.
    th_step : float
        threshold step value of dynamics.

    Returns
    -------
    torch tensor
        persistent threshold state to store for next time.
    """
    spike = (spike > 0).to(th_state.dtype)
    return th_state + spike * th_step


class _ATHDynamics(torch.autograd.Function):
    """ """
    DEBUG = False

    @staticmethod
    def forward(
        ctx,
        input, ref_state, ref_decay,
        th_state, th_decay, th_scale, th0,
        w_scale
    ):
        """ """
        threshold, refractory = _ath_dynamics_fwd(
            input,
            ref_state, ref_decay,
            th_state, th_decay, th_scale, th0,
            w_scale, dtype=torch.int64
        )

        if _ATHDynamics.DEBUG is True and input.is_cuda is True:
            _threshold, _refractory, *_ = Accelerated.adaptive_threshold.fwd(
                input,
                ref_state, ref_decay,
                th_state, th_decay, th_scale, th0,
                w_scale
            )
            # print('Fwd Checking')
            for i in range(threshold.shape[1]):
                if (
                    torch.norm(threshold[0, i] - _threshold[0, i]) > 1e-6
                    or torch.norm(refractory[0, i] - _refractory[0, i]) > 1e-6
                ):
                    print('threshold:', i, torch.norm(
                        threshold[0, i] - _threshold[0, i]
                    ))
                    print(threshold[0, i, :50] * w_scale)
                    print(_threshold[0, i, :50] * w_scale)
                    print('refractory:', i, torch.norm(
                        refractory[0, i] - _refractory[0, i]
                    ))
                    print(refractory[0, i, :50] * w_scale)
                    print(_refractory[0, i, :50] * w_scale)
                    print(
                        refractory[refractory != _refractory].flatten()
                        * w_scale
                    )
                    print(
                        _refractory[refractory != _refractory].flatten()
                        * w_scale
                    )

                    raise Exception

        return threshold, refractory

    @staticmethod
    # def backward(ctx, grad_threshold, grad_refractory):
    def backward(*_):
        """ """
        return None, None, None, None, None, None, None, None


def _ath_dynamics_fwd(
    input,
    ref_state, ref_decay,
    th_state, th_decay, th_scale, th0,
    w_scale, dtype=torch.int32
):
    threshold_old = (th_state * w_scale)\
        .clone().detach().to(dtype).to(input.device)
    refractory_old = (ref_state * w_scale)\
        .clone().detach().to(dtype).to(input.device)
    th_decay_int = (1 << 12) - th_decay\
        .clone().detach().to(dtype).to(input.device)
    ref_decay_int = (1 << 12) - ref_decay\
        .clone().detach().to(dtype).to(input.device)
    threshold = torch.zeros_like(input)
    refractory = torch.zeros_like(input)

    th_scale = int(th_scale * w_scale)
    th0 = int(th0 * w_scale)

    for n in range(input.shape[-1]):
        refractory_new = right_shift_to_zero(
            refractory_old * ref_decay_int,
            12
        )
        threshold_new = right_shift_to_zero(
            (threshold_old - th0) * th_decay_int,
            12
        ) + th0
        output_new = (w_scale * input[..., n]).to(dtype)
        spike_new = (output_new >= (threshold_new + refractory_new))
        threshold_old = threshold_new + th_scale * (spike_new > 0.5)
        refractory_old = refractory_new + 2 * threshold_new * (spike_new > 0.5)

        threshold[..., n] = threshold_new / w_scale
        refractory[..., n] = refractory_new / w_scale

    return threshold, refractory
