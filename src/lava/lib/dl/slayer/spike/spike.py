# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Spike mechanism implementation."""

import torch
from ..axon.delay import delay


def _spike_backward(
    voltage, threshold, tau_rho, scale_rho,
    graded_spike=False
):
    """
    """
    if graded_spike is False:
        return scale_rho / 2 / tau_rho * torch.exp(
            - torch.abs(voltage - threshold) / tau_rho
        )

    return scale_rho * torch.exp(
        -torch.clamp(voltage - threshold, max=0) / tau_rho
    )


class Spike(torch.autograd.Function):
    """Spiking mechanism with autograd link.

    .. math::
        f_s(x) &= \\mathcal{H}(x - \\vartheta)

    Parameters
    ----------
    voltage : torch tensor
        neuron voltage.
    threshold : float or torch tensor
        neuron threshold
    tau_rho : float
        gradient relaxation
    scale_rho : float
        gradient scale
    graded_spike : bool
        flag for graded spike
    voltage_last : torch tensor
        voltage at t=-1
    scale : int
        variable scale value.

    Returns
    -------
    torch tensor
        spike tensor

    Examples
    --------

    >>> spike = Spike.apply(v, th, tau_rho, scale_rho, False, 0, 1)
    """
    # this could be used to make a choice of surrogate derivative.
    # Smooth functions other than decaying exponential seem unsuitable for now.
    derivative = None

    @staticmethod
    def forward(
        ctx,
        voltage, threshold,
        tau_rho, scale_rho,
        graded_spike,
        voltage_last, scale
    ):
        """
        """
        device = voltage.device
        dtype = voltage.dtype

        # spike function is formally defined as
        # f_s(v) = \mathcal{H}(v - \vartheta)

        if graded_spike is True:
            if voltage.is_cuda is True:
                voltage_old = delay(voltage)
            else:
                voltage_old = torch.zeros_like(voltage)
                voltage_old[..., 1:] = voltage[..., :-1]
            voltage_old[..., 0] = voltage_last
            spikes = ((voltage >= threshold) * voltage / scale).to(dtype)
        else:
            spikes = (voltage >= threshold).to(dtype)

        graded_spike = 1 if graded_spike is True else 0
        if torch.is_tensor(threshold) is False:
            threshold = torch.tensor(threshold, device=device, dtype=dtype)
        ctx.save_for_backward(
            voltage,
            torch.autograd.Variable(threshold, requires_grad=False),
            torch.autograd.Variable(
                torch.tensor(tau_rho, device=device, dtype=dtype),
                requires_grad=False
            ),
            torch.autograd.Variable(
                torch.tensor(scale_rho, device=device, dtype=dtype),
                requires_grad=False
            ),
            torch.autograd.Variable(
                torch.tensor(graded_spike, device=device, dtype=dtype),
                requires_grad=False
            ),
        )

        return spikes

    @staticmethod
    def backward(ctx, grad_spikes):
        """
        """
        voltage, threshold, tau_rho, scale_rho, graded_spike \
            = ctx.saved_tensors
        graded_spike = True if graded_spike > 0.5 else False
        # print(voltage)
        # print(_spike_backward(voltage, threshold, tau_rho, scale_rho))
        return (
            _spike_backward(
                voltage, threshold, tau_rho, scale_rho, graded_spike
            ) * grad_spikes,
            None, None, None, None, None, None
        )
