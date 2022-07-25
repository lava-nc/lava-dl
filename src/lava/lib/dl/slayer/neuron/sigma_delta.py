# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Sigma Delta neuron."""

import torch

from . import base
from ..dendrite import Sigma
from ..axon import Delta


def neuron_params(device_params, scale=1 << 6):
    """Translates device parameters to neuron parameters.

    Parameters
    ----------
    device_params : dictionary
        dictionary of device parameter specification.

    scale : int
        neuron scale value. Default value = 1  <<  6.

    Returns
    -------
    dictionary
        dictionary of neuron parameters that can be used to initialize neuron
        class.
    """
    # p_scale = 1 << 12
    return {
        'threshold': device_params['vThMant'] / scale,
    }


class Neuron(base.Neuron):
    """This is the implementation of Sigma-Delta wrapper neuron.

    The internal state representations are scaled down compared to
    the actual hardware implementation. This allows for a natural range of
    synaptic weight values as well as the gradient parameters.

    The neuron parameters like threshold, decays are represented as real
    values. They internally get converted to fixed precision representation of
    the hardware. It also provides properties to access the neuron
    parameters in fixed precision states. The parameters are internally clamped
    to the valid range.

    Parameters
    ----------
    threshold : float
        neuron threshold.
    activation: fx-ptr or lambda
        The neuron activation class instance that needs to be wrapped
        by sigma-delta unit. For e.g. ``torch.nn.functional.relu`` would
        give sigma-delta-relu unit.
    tau_grad : float, optional
        time constant of spike function derivative. Defaults to 1.
    scale_grad : float, optional
        scale of spike function derivative. Defaults to 1.
    scale : int, optional
        scale of the internal state. `scale=1` will result in values in the
        range expected from the of Loihi hardware. Defaults to 1 << 6.
    cum_error : bool, optional
        flag to enable/disable residual state of delta unit. Defaults to False.
    norm : fx-ptr or lambda, optional
        normalization function on the dendrite output. None means no
        normalization. Defaults to None.
    dropout : fx-ptr or lambda, optional
        neuron dropout method. None means no normalization. Defaults to None.
    shared_param : bool, optional
        flag to enable/disable shared parameter neuron group. If it is
        False, individual parameters are assigned on a per-channel basis.
        Defaults to True.
    persistent_state : bool, optional
        flag to enable/disable persistent state between iterations.
        Defaults to False.
    requires_grad : bool, optional
        flag to enable/disable learning on neuron parameter. Defaults to False.
    """
    def __init__(
        self, threshold, activation,
        tau_grad=1, scale_grad=1,
        scale=1 << 6, cum_error=False,
        norm=None, dropout=None,
        shared_param=True, persistent_state=False, requires_grad=False
    ):
        """ """
        super(Neuron, self).__init__(
            threshold=threshold,
            w_scale=scale,
            s_scale=scale * (1 << 6),
            norm=norm,
            dropout=dropout,
            persistent_state=persistent_state,
            shared_param=shared_param,
            requires_grad=requires_grad
        )
        self.graded_spike = True
        self.activation = activation

        self.sigma = Sigma(persistent_state=self.persistent_state)
        self.delta = Delta(
            threshold=self._threshold,
            scale=self.s_scale,
            tau_grad=tau_grad,
            scale_grad=scale_grad,
            cum_error=cum_error,
            shared_param=self.shared_param,
            persistent_state=self.persistent_state,
            requires_grad=self.requires_grad
        )
        self.bias_is_set = False
        self.register_parameter(
            'bias',
            torch.nn.Parameter(
                torch.FloatTensor([0]),
                requires_grad=self.requires_grad
            )
        )

    @property
    def device(self):
        """The device memory (cpu/cuda) where the object lives."""
        return self.delta.pre_state.device

    # @property
    # def shape(self):
    #     """The shape of the layer
    #     """
    #     assert self.sigma.shape == self.delta.shape, \
    #         f'The shape of sigma and delta do not match. '\
    #         f'Found {self.sigma.shape=} and {self.delta.shape=}.'
    #     return self.sigma.shape

    @property
    def threshold(self):
        """Neuron threshold"""
        return self.delta.threshold

    @property
    def scale(self):
        """Scale difference between slayer representation and hardware
        representation of the variable states."""
        return self.w_scale

    @property
    def device_params(self):
        """Dictionary of device parameters."""
        return {
            'type': 'SDNN',
            'activation': self.activation.__name__,
            'vThMant': self.v_th_mant,
        }

    def set_bias(self, bias):
        """Sets the bias for sigma-delta unit

        Parameters
        ----------
        bias : torch tensor
            bias corresponding to each neuron.

        """
        if self.shape is None:
            self.bias.data = bias.to(self.bias.device)
        else:
            self.bias.data = bias.reshape(self.shape).to(self.bias.device)
        self.bias_is_set = True

    def forward(self, input):
        """Computes the full response of the neuron instance to an input.
        The input shape must match with the neuron shape. For the first time,
        the neuron shape is determined from the input automatically.

        Parameters
        ----------
        input : torch tensor
            Input tensor.

        Returns
        -------
        torch tensor
            graded spike response of the neuron.

        """

        # dendrite computation
        dend = self.sigma(input)
        if self.norm is not None:
            dend = self.norm(dend)

        # neuron computation
        if self.bias_is_set:
            axon = self.delta(
                self.activation(
                    dend + torch.unsqueeze(
                        torch.unsqueeze(self.bias, dim=0),
                        dim=-1
                    )
                ))
        else:
            axon = self.delta(self.activation(dend))

        # axon = self.quantize_8bit(axon*2)/2

        # axon computation
        if self.drop is not None:
            axon = self.drop(axon)

        if self.shape is None:
            if self.sigma.shape != self.delta.shape:
                raise AssertionError(
                    f'The shape of sigma and delta do not match. '
                    f'Found {self.sigma.shape=} and {self.delta.shape=}.'
                )
            self.shape = self.sigma.shape

        return axon
