# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Abstract neuron base class."""

import numpy as np
import torch

from ..utils import quantize

# Base neuron class
# Provides a base framework for neuron classes


class Neuron(torch.nn.Module):
    """This is an abstract class that governs the minimal basic functionality
    of a neuron object.

    Parameters
    ----------
    threshold : float or torch tensor
        neuron threshold.
    tau_grad : float, optional
        controls the relaxation of spike function gradient. It determines
        the scope of voltage/state around the neuron threshold that
        effectively contributes to the error. Default is 1
    scale_grad : float, optional
        controls the scale of spike function gradient. It controls the
        gradient flow across layers. It should be increased if there is
        vanishing gradient and increased if there is exploding gradient.
        Default is 1
    p_scale : int, optional
        scaling factor of neuron parameter. Default is 1
    w_scale : int, optional
        scaling factor for dendrite input (weighted spikes). It's good to
        compute synaptic operations and its gradients on smaller range.
        w_scale scales down the synaptic weights for quantization.
        The actual synaptic weights must be descaled. Default is 1
    s_scale : int, optional
        scaling factor for neuron states. The fixed percision neuron states
        are scaled down by s_scale so that they are in a reasonable range
        for gradient flow. Default is 1
    norm : fx-ptr or lambda, optional
        normalization function on the dendrite output. None means no
        normalization. Default is None
    dropout : fx-ptr or lambda, optional
        neuron dropout method. None means no dropout. Default is None
    persistent_state : bool, optional
        flag to enable/disable persitent state between iterations.
        Default is False
    shared_param : bool, optional
        flag to enable/disable shared parameter for neuron group. If False,
        idividual parameters are assigned on a per-channel basis.
        Default is True
    requires_grad : bool, optional
        flag to enable/disable learnable neuron decays. Default is True
    complex : bool, optional
        flag to indicate real or complex neuron. Defaul

    Attributes
    ----------
    Neuron Attributes
        threshold


    Attributes
    ----------
    Gradient Attributes
        * tau_grad
        * scale_grad

    Attributes
    ----------
    Scaling Atrributes
        * p_scale
        * w_scale
        * s_scale

    Attributes
    ----------
    Normalization Attributes
        * norm

    Attributes
    ----------
    Dropout Attributes
        * dropout

    Attributes
    ----------
    State Attributes
        * persistent_state

    Attributes
    ----------
    Group Attributes
        * shared_param
        * complex

    Attributes
    ----------
    Debug Attributes
        * debug : bool
            It is False by default. There shall be no constructor access to this
            flag. If desired, it should be explicitly set.
    """
    def __init__(
        self, threshold,
        tau_grad=1, scale_grad=1,
        p_scale=1, w_scale=1, s_scale=1,
        norm=None, dropout=None,
        persistent_state=False, shared_param=True,
        requires_grad=True,
        complex=False
    ):
        """
        """
        super(Neuron, self).__init__()
        # shape and num_neurons should be initialized only when shared_param is
        # true and during the first forward pass
        # They should be trivially immutable
        self.num_neurons = None
        self.shape = None
        self.p_scale = p_scale
        self.w_scale = int(w_scale)
        self.s_scale = int(s_scale)
        # quantize to proper value
        self._threshold = int(threshold * self.w_scale) / self.w_scale
        self.tau_rho = tau_grad * self._threshold
        self.scale_rho = scale_grad
        self.shared_param = shared_param
        self.persistent_state = persistent_state
        self.requires_grad = requires_grad
        self.debug = False
        if self.s_scale < self.w_scale:
            raise AssertionError(
                'State scale is expected to be greater than weight scale. '
                'Found following\n'
                f'state scale = {self.s_scale}\n'
                f'weight scale = {self.w_scale}\n'
            )
        self.complex = complex  # by default the neuron is not complex

        if norm is not None:
            if self.complex is False:
                self.norm = norm()
                if hasattr(self.norm, 'pre_hook_fx'):
                    self.norm.pre_hook_fx = self.quantize_8bit
            else:
                self.real_norm = norm()
                self.imag_norm = norm()
                if hasattr(self.real_norm, 'pre_hook_fx'):
                    self.real_norm.pre_hook_fx = self.quantize_8bit
                if hasattr(self.imag_norm, 'pre_hook_fx'):
                    self.imag_norm.pre_hook_fx = self.quantize_8bit
        else:
            self.norm = None
            if self.complex is True:
                self.real_norm = None
                self.imag_norm = None
        self.drop = dropout if dropout is not None else None

    @property
    def threshold(self):
        """ """
        return self._threshold

    def quantize_8bit(self, weight, descale=False):
        """Quantization method for 8 bit equivalent input when descaled.
        This should be linked with synapse instance.

        Parameters
        ----------
        weight : torch.tensor
            synaptic weight.
        descale : Bool
            flag to scale/descale the weight (Default value = False)

        Returns
        -------
        torch.tensor
            quantized weight.

        Examples
        --------
        It can be used like a normal function. But the intended use is as
        follows

        >>> synapse.pre_hook_fx = neuron.quantize_8bit
        """
        if descale is False:
            return quantize(
                weight, step=2 / self.w_scale
            ).clamp(-256 / self.w_scale, 255 / self.w_scale)
        else:
            return quantize(
                weight, step=2 / self.w_scale
            ).clamp(-256 / self.w_scale, 255 / self.w_scale) * self.w_scale

    @property
    def weight_exponent(self):
        """Get weight exponent."""
        return int(np.log2(self.s_scale // self.w_scale) - 6)

    @property
    def v_th_mant(self):
        """Get voltage-threshold-mantessa parameter."""
        return int(self.threshold * self.w_scale)

    @property
    def device(self):
        """ """
        pass

    def clamp(self):
        """ """
        pass
