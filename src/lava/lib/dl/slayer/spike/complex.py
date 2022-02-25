# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Complex spike (Phase threshold) implementation."""

import torch
from ..axon.delay import delay


class Spike(torch.autograd.Function):
    """Complex spike function with autograd link.

    .. math::
        f_s(z) &= \\mathcal{H}(|z| - \\vartheta)\\,\\delta(\\arg(z))

    Parameters
    ----------
    real : torch tensor
        real component of neuron response.
    imag : torch tensor
        imaginary component of neuron response.
    threshold : torch tensor or float
        neuron threshold.
    tau_rho : float
        gradient relaxation
    scale_rho : float
        gradient scale
    graded_spike : bool
        flag for graded spike
    imag_last : torch tensor
        imaginary response at t=-1
    scale : int
        variable scale value.

    Returns
    -------
    torch tensor
        spike tensor

    Examples
    --------

    >>> spike = Spike.apply(re, im, th, tau_rho, scale_rho, False, 0, 1)
    """
    # this could be used to make a choice of surrogate derivative.
    # Smooth functions other than decaying exponential seem unsuitable for now.
    derivative = None

    @staticmethod
    def forward(
        ctx,
        real, imag, threshold,
        tau_rho, scale_rho,
        graded_spike,
        imag_last,
        scale
    ):
        """
        """
        device = real.device
        dtype = real.dtype

        # spike function is formally defined as
        # f_s(u + iv) = \mathcal{H}(u - \vartheta) \delta(v)

        if imag.is_cuda is True:
            # imag_del = slayerCuda.shift(imag.contiguous(), 1, 1)
            imag_del = delay(imag)
        else:
            imag_del = torch.zeros_like(imag)
            imag_del[..., 1:] = imag[..., :-1]
        imag_del[..., 0] = imag_last

        if graded_spike is True:
            spikes = (
                # (real - threshold) * # real[t] - real[t-1]
                # is not a good idea as
                # the dynamics will be crossing zero orthogonally
                real
                # real[t] - real[t-1] is not a good idea as the
                # dynamics will be crossing zero orthogonally
                * (real >= threshold)
                * (imag_del < 0)
                * (imag >= 0)
                / scale  # scale to get integer precision spike values
            ).to(dtype)
        else:
            spikes = (
                (real >= threshold)
                * (imag_del < 0)
                * (imag >= 0)
            ).to(dtype)

        graded_spike = 1 if graded_spike is True else 0
        if torch.is_tensor(threshold) is False:
            threshold = torch.tensor(threshold, device=device, dtype=dtype)
        ctx.save_for_backward(
            real, imag,
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
    def backward(ctx, grad_output):
        """
        """
        real, imag, threshold, tau_rho, scale_rho, graded_spike \
            = ctx.saved_tensors

        # we need to calculate the derivative of
        # f_s(u + iv) = \mathcal{H}(u - \vartheta) \delta(v)
        # which are
        # \frac{\partial f_s}{\partial u} = \delta(u - \vartheta) \delta(v)
        # and
        # \frac{\partial f_s}{\partial v} = \mathcal{H}(u - \vartheta)
        # \frac{\partial\delta(v)}{\partial v}
        #
        # If we relax the delta function using
        # \delta(x)\approx\frac{1}{2a}\,e^{(-\frac{|x|}{a})}
        # Then
        # \frac{\partial\delta(x)}{\partial x}\approx
        # -\frac{\text{sign}(x)}{2a^2}e^{(-\frac{|x|}{a})} =
        # -\frac{1}{a}\,\text{sign}(x)\,\delta(x)
        #
        # Note: the scaling by 1/2a is necessary to ensure that
        # area under the dirac-delta is always 1.

        # scale_rho *= scale_rho_mult
        # tau_rho *= tau_rho_mult

        if graded_spike > 0:
            delta_real = torch.exp(-(threshold - real).clamp(0) / tau_rho)
        else:
            delta_real = 1 / 2 / tau_rho * torch.exp(
                -torch.abs(real - threshold) / tau_rho
            )

        delta_imag = 1 / 2 / tau_rho * torch.exp(-torch.abs(imag) / tau_rho)
        step_real = 1 / 2 * (
            1
            + torch.sign(real - threshold)
            * (1 - 2 * tau_rho * delta_real)
        )

        grad_real = scale_rho * delta_real * delta_imag
        grad_imag = - scale_rho / tau_rho * torch.sign(imag) * \
            delta_imag * step_real
        # grad_imag = scale_rho / tau_rho * delta_imag * step_real

        # grad_imag = 1
        # grad_real = 1

        return grad_output * grad_real, grad_output * grad_imag, \
            None, None, None, None, None, None
