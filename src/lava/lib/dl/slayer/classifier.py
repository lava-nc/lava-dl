# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Classifier modules."""

import torch
import torch.nn.functional as F
from .utils.filter import FIR


class Rate(torch.nn.Module):
    """Global rate based classifier. It considers the event rate of the spike
    train over the entire duration as the confidence score.

    .. math::
        \\text{rate: } {\\bf r} &= \\frac{1}{T}\\int_T{\\bf s}(t)\\,\\text dt\\

        \\text{confidence: } {\\bf c} &= \\begin{cases}
            \\frac{\\bf r}{\\bf r^\\top 1}
                &\\text{ if mode=probability} \\\\
            \\frac{\\exp({\\bf r})}{\\exp({\\bf r})^\\top \\bf1}
                &\\text{ if mode=softmax} \\\\
            \\log\\left(
                \\frac{\\exp({\\bf r})}{\\exp({\\bf r})^\\top \\bf1}
            \\right)
                &\\text{ if mode=softmax}
        \\end{cases} \\\\

        \\text{prediction: } p &= \\arg\\max(\\bf r)

    Examples
    --------

    >>> classifier = Rate
    >>> prediction = classifier(spike)
    """

    def __init__(self):
        super(Rate, self).__init__()

    def forward(self, spike):
        """
        """
        return Rate.predict(spike)

    @staticmethod
    def rate(spike):
        """Given spike train, returns the output spike rate.

        Parameters
        ----------
        spike : torch tensor
            spike tensor. First dimension is assumed to be batch, and last
            dimension is assumed to be time. Spatial dimensions are collapsed
            by default.

        Returns
        -------
        torch tensor
            spike rate.

        Examples
        --------

        >>> rate = classifier.rate(spike)
        >>> rate = Rate.rate(spike)
        """
        return torch.mean(spike, dim=-1)

    @staticmethod
    def confidence(spike, mode='probability', eps=1e-6):
        """Given spike train, returns the confidence of the output class based
        on spike rate.

        Parameters
        ----------
        spike : torch tensor
            spike tensor. First dimension is assumed to be batch, and last
            dimension is assumed to be time. Spatial dimensions are collapsed
            by default.
        mode : str
            confidence mode. One of 'probability'|'softmax'|'logsoftmax'.
            Defaults to 'probability'.
        eps : float
            infinitesimal value. Defaults to 1e-6.

        Returns
        -------
        torch tensor
            confidence.

        Examples
        --------

        >>> confidence = classifier.confidence(spike)
        >>> confidence = Rate.confidence(spike)
        """
        rate = Rate.rate(spike).reshape(spike.shape[0], -1)
        if mode == 'probability':
            return rate / (torch.sum(rate, dim=1, keepdim=True) + eps)
        elif mode == 'softmax':
            return F.softmax(rate, dim=1)
        elif mode == 'logsoftmax':
            return F.log_softmax(rate, dim=1)

    @staticmethod
    def predict(spike):
        """Given spike train, predicts the output class based on spike rate.

        Parameters
        ----------
        spike : torch tensor
            spike tensor. First dimension is assumed to be batch, and last
            dimension is assumed to be time. Spatial dimensions are collapsed
            by default.

        Returns
        -------
        torch tensor
            indices of max spike activity.

        Examples
        --------

        >>> prediction = classifier.predict(spike)
        >>> prediction = Rate.predict(spike)
        """
        rate = Rate.rate(spike)
        return torch.max(rate.reshape(spike.shape[0], -1), dim=1)[1]


class MovingWindow(torch.nn.Module):
    """Moving window based classifier. It produces a timeseries of
    classification/prediction based on moving window estimate.

    .. math::
        \\text{rate: } {\\bf r}(t)
            &= \\frac{1}{W}\\int_{t-W}^T {\\bf s}(t)\\,\\text dt \\

        \\text{confidence: } {\\bf c}(t) &= \\begin{cases}
            \\frac{{\\bf r}(t)}{{\\bf r}(t)^\\top {\\bf1}}
                &\\text{ if mode=probability} \\\\
            \\frac{\\exp({\\bf r}(t))}{\\exp({\\bf r}(t))^\\top \\bf1}
                &\\text{ if mode=softmax} \\\\
            \\log\\left(
                \\frac{\\exp({\\bf r}(t))}{\\exp({\\bf r}(t))^\\top \\bf1}
            \\right) &\\text{ if mode=softmax}
        \\end{cases} \\

        \\text{prediction: } p(t) &= \\arg\\max({\\bf r}(t))

    Parameters
    ----------
    time_window : int
        size of moving window.
    mode : str
        confidence mode. One of 'probability'|'softmax'|'logsoftmax'.
        Defaults to 'probability'.
    eps : float
        infinitesimal value. Defaults to 1e-6.

    Examples
    --------

    >>> classifier = MovingWindow(20)
    >>> prediction = classifier(spike)
    """
    def __init__(self, time_window, mode='probability', eps=1e-6):
        super(MovingWindow, self).__init__()
        self.filter = FIR(fir_response=torch.ones(time_window) / time_window)
        # moving window average
        self.mode = mode
        self.eps = eps

    def rate(self, spike):
        """Moving window spike rate.

        Parameters
        ----------
        spike : torch tensor
            spike input.

        Returns
        -------
        torch tensor
            spike rate.

        Examples
        --------

        >>> rate = classifier.rate(spike)
        """
        return self.filter(spike)

    def confidence(self, spike, mode=None):
        """Moving window confidence.

        Parameters
        ----------
        spike : torch tensor
            spike input.
        mode : str
            confidence mode. If it is None, the object's mode is used.
            Defaults to None.

        Returns
        -------
        torch tensor
            output confidence.

        Examples
        --------

        >>> confidence = classifier.confidence(spike)
        """
        sliding_rate = self.rate(spike).reshape(
            spike.shape[0], -1, spike.shape[-1])

        if mode is None:
            mode = self.mode

        if mode == 'probability':
            return sliding_rate / (torch.sum(
                sliding_rate, dim=1, keepdim=True
            ) + self.eps)
        if mode == 'softmax':
            return F.softmax(sliding_rate, dim=1)
        if mode == 'logsoftmax':
            return F.log_softmax(sliding_rate, dim=1)

        raise Exception(f'Unsupported mode {self.mode}.')

    def predict(self, spike):
        """Moving window prediction.

        Parameters
        ----------
        spike : torch tensor
            spike input.

        Returns
        -------
        torch tensor
            output prediction.

        Examples
        --------

        >>> prediction = classifier.predict(spike)
        """
        return torch.argmax(self.rate(spike), dim=1)

    def forward(self, spike):
        """
        """
        self.predict(spike)
