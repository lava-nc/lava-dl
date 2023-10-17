# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""This module provides some pre-built loss methods to be used with
spike-train. Standard PyTorch loss are also compatible."""

import torch
import torch.nn.functional as F
from .utils.filter.fir import FIR
from .utils.time.replicate import replicate
from .classifier import Rate, MovingWindow


class SpikeTime(torch.nn.Module):
    """Spike-time based loss. It is similar to van Rossum distance between
    output and desired spike train.

    .. math::
        L = \\int_0^T \\left( \\varepsilon * (s - \\hat{s}) \\right)(t)^2\\,
            \\text{d}t

    Parameters
    ----------
    time_constant : int
        time constant of low pass filter. Defaults to 5.
    length : int
        length of low pass filter. Defaults to 100.
    filter_order : int
        order of low pass filter. Defaults to 1.
    reduction : str
        mean square reduction. Options are 'mean'|'sum'. Defaults to 'sum'.
    """
    def __init__(
        self,
        time_constant=5, length=100, filter_order=1,
        reduction='sum'
    ):
        super(SpikeTime, self).__init__()
        self.filter = FIR(time_constant=time_constant, length=length)
        if filter_order > 1:
            new_fir = self.filter.filter.data
            for _ in range(filter_order - 1):
                new_fir = self.filter(new_fir)
            self.filter.filter.data = new_fir
        self.reduction = reduction

    def forward(self, input, desired):
        """Forward computation of loss.
        """
        return F.mse_loss(
            self.filter(input).flatten(),
            self.filter(desired).flatten(),
            reduction=self.reduction
        )


class SpikeRate(torch.nn.Module):
    """Spike rate loss.

    .. math::
        \\hat {\\boldsymbol r} &=
            r_\\text{true}\\,{\\bf 1}[\\text{label}] +
            r_\\text{false}\\,(1 - {\\bf 1}[\\text{label}])\\

        L &= \\begin{cases}
        \\frac{1}{2}\\int_T(
            {\\boldsymbol r}(t) - \\hat{\\boldsymbol r}(t)
        )^\\top {\\bf 1}\\,\\text dt &\\text{ if moving window}\\\\
        \\frac{1}{2}(
            \\boldsymbol r - \\hat{\\boldsymbol r}
        )^\\top 1 &\\text{ otherwise}
        \\end{cases}


    Note: input is always collapsed in spatial dimension.

    Parameters
    ----------
    true_rate : float
        true spiking rate.
    false_rate : float
        false spiking rate.
    moving_window : int
        size of moving window. If not None, assumes label to be specified
        at every time step. Defaults to None.
    reduction : str
        loss reduction method. One of 'sum'|'mean'. Defaults to 'sum'.

    Returns
    -------

    """
    def __init__(
        self, true_rate, false_rate,
        moving_window=None, reduction='sum'
    ):
        super(SpikeRate, self).__init__()
        if not (true_rate >= 0 and true_rate <= 1):
            raise AssertionError(
                f'Expected true rate to be between 0 and 1. Found {true_rate=}'
            )
        if not (false_rate >= 0 and false_rate <= 1):
            raise AssertionError(
                f'Expected false rate to be between 0 and 1. '
                f'Found {false_rate=}'
            )
        self.true_rate = true_rate
        self.false_rate = false_rate
        self.reduction = reduction
        if moving_window is not None:
            self.window = MovingWindow(moving_window)
        else:
            self.window = None

    def forward(self, input, label):
        """Forward computation of loss.
        """
        input = input.reshape(input.shape[0], -1, input.shape[-1])
        if self.window is None:  # one label for each sample in a batch
            one_hot = F.one_hot(label, num_classes=input.shape[1])
            spike_rate = Rate.rate(input)
            target_rate = self.true_rate * one_hot \
                + self.false_rate * (1 - one_hot)
            return F.mse_loss(
                spike_rate.flatten(),
                target_rate.flatten(),
                reduction=self.reduction
            )

        if len(label.shape) == 1:  # assume label is in (batch, time) form
            label = replicate(label, input.shape[-1])
        # transpose the time dimension to the end
        # (batch, time, num_class) -> (batch, num_class, time)
        one_hot = F.one_hot(
            label,
            num_classes=input.shape[1]
        ).transpose(2, 1)  # one hot encoding in time
        spike_rate = self.window.rate(input)
        target_rate = self.true_rate * one_hot \
            + self.false_rate * (1 - one_hot)
        return F.mse_loss(
            spike_rate.flatten(),
            target_rate.flatten(),
            reduction=self.reduction
        )


class SpikeMax(torch.nn.Module):
    """Spike max (NLL) loss.

    .. math::
        L &= \\begin{cases}
            -\\int_T
                {\\bf 1}[\\text{label}]^\\top
                \\log(\\boldsymbol p(t))\\,\\text dt
                &\\text{ if moving window}\\\\
            -{\\bf 1}[\\text{label}]^\\top
            \\log(\\boldsymbol p) &\\text{ otherwise}
        \\end{cases}

    Note: input is always collapsed in spatial dimension.

    Parameters
    ----------
    moving_window : int
        size of moving window. If not None, assumes label to be specified
        at every time step. Defaults to None.
    mode : str
        confidence mode. One of 'probability'|'softmax'.
        Defaults to 'probability'.
    reduction : str
        loss reduction method. One of 'sum'|'mean'. Defaults to 'sum'.
    """
    def __init__(
        self, moving_window=None, mode='probability', reduction='sum'
    ):
        super(SpikeMax, self).__init__()
        if moving_window is not None:
            self.window = MovingWindow(moving_window)
        else:
            self.window = None
        self.mode = mode
        self.reduction = reduction

    def forward(self, input, label):
        """Forward computation of loss.
        """
        input = input.reshape(input.shape[0], -1, input.shape[-1])
        if self.window is None:  # one label for each sample in a batch
            if self.mode == 'probability':
                log_p = torch.log(Rate.confidence(input, mode=self.mode))
            else:
                log_p = Rate.confidence(input, mode='logsoftmax')

            return F.nll_loss(log_p, label, reduction=self.reduction)
        else:
            if len(label.shape) == 1:  # assume label is in (batch, time) form
                float_label = label[..., None].float()
                label = replicate(float_label, input.shape[-1]).to(label.dtype)
            # transpose the time dimension to the end
            # (batch, time, num_class) -> (batch, num_class, time)
            if self.mode == 'probability':
                log_p = torch.log(
                    self.window.confidence(input, mode=self.mode)
                )
            else:
                log_p = self.window.confidence(input, mode='logsoftmax')
            return F.nll_loss(
                log_p.transpose(1, 2).reshape(-1, input.shape[1]),
                label.flatten(),
                reduction=self.reduction,
            )


class SpikeMoid(torch.nn.Module):
    """
    SpikeMoid (BCE) loss.

    .. math::

        \\text{if sliding window:} \\quad
        p(t) = \\sigma\\left(\\frac{r(t) - \\theta}{\\alpha}\\right) \\\\
        \\text{otherwise:} \\quad
        p = \\sigma\\left(\\frac{r - \\theta}{\\alpha}\\right)

    r signifies a spike rate calculated over the time dimension

    .. math::
        \\mathcal{L} = \\begin{cases}
            -\\int_T \\hat{y}(t) \\cdot \\log{p(t)}
            + (1 - \\hat{y}(t)) \\cdot \\log{(1 - p(t))}\\,\\text{d}t
            &\\text{if sliding window} \\\\
            -\\left(\\hat{y} \\cdot \\log{p}
            + (1 - \\hat{y}) \\cdot \\log{(1 - p)}\\right)
            &\\text{otherwise}
        \\end{cases}

    Note: input is always collapsed in the spatial dimension.
    r signifies a spike rate calculated over the time dimension

    Parameters
    ----------
    moving_window : int
        size of moving window. If not None, assumes label to be specified
        at every time step. Defaults to None.
    reduction : str
        loss reduction method. One of 'sum'|'mean'. Defaults to 'sum'.
    alpha : int
        Sigmoid temperature parameter. Defaults to 1.
    theta : int
        Bias term for logits. Defaults to 1.
    """
    def __init__(
        self, moving_window=None, reduction='sum', alpha=1, theta=0
    ):
        super(SpikeMoid, self).__init__()
        if moving_window is not None:
            self.window = MovingWindow(moving_window)
        else:
            self.window = None
        self.reduction = reduction
        self.alpha = alpha
        self.theta = theta

    def forward(self, input, label):
        """Forward computation of loss.
        """
        input = input.reshape(input.shape[0], -1, input.shape[-1])
        if self.window is None:  # one label for each sample in a batch
            scaled_input = (input - self.theta) / self.alpha
            probs = torch.sigmoid(scaled_input.mean(-1)).flatten(0, 1)
            return F.binary_cross_entropy(
                probs,
                label.flatten(),
                reduction=self.reduction
            )
        else:
            # assume label is in (batch, num_classes, time) form
            if len(label.shape) == 2:
                label = replicate(label, input.shape[-1])
                float_label = label[..., None]
            rates = self.window.rate(input)
            probs = torch.sigmoid((rates - self.theta) / self.alpha)
            return F.binary_cross_entropy(
                probs.flatten(),
                label.flatten(),
                reduction=self.reduction
            )


class SparsityEnforcer:
    """Event sparsity enforcement module. Penalizes event rate higher than
    a specific value.

    Parameters
    ----------
    max_rate : float, optional
        Rate above which the events are penalized, by default 0.01.
    lam : float, optional
        Ratio of event rate loss scaling, by default 1.0.
    """
    def __init__(self, max_rate: float = 0.01, lam: float = 1.0) -> None:
        self.max_rate = max_rate
        self.lam = lam
        self.loss_list = []

    def clear(self) -> None:
        """Clear all gathered sparsity loss.
        """
        self.loss_list = []

    @property
    def loss(self) -> torch.tensor:
        """Accumulate sparsity loss.
        """
        return self.lam * sum(self.loss_list)

    def append(self, x: torch.tensor) -> None:
        """Appends loss tickets given the state of input tensors.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.
        """
        mean_event_rate = torch.mean(torch.abs(x))
        self.loss_list.append(F.mse_loss(F.relu(mean_event_rate
                                                - self.max_rate),
                                         torch.zeros_like(mean_event_rate)))
