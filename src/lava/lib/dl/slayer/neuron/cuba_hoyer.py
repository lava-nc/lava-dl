# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""CUBA neuron model."""

import torch
import torch.nn as nn

from .dynamics import leaky_integrator
from ..spike import HoyerSpike
from .cuba import Neuron

# These are tuned heuristically so that scale_grad=1 and tau_grad=1 serves
# as a good starting point

# SCALE_RHO_MULT = 0.1
# TAU_RHO_MULT = 10
SCALE_RHO_MULT = 0.1
TAU_RHO_MULT = 100
# SCALE_RHO_MULT = 1
# TAU_RHO_MULT = 1

class HoyerNeuron(Neuron):
    """This is the implementation of Loihi CUBA neuron.

    .. math::
        u[t] &= (1 - \\alpha_u)\\,u[t-1] + x[t] \\

        v[t] &= (1 - \\alpha_v)\\,v[t-1] + u[t] + \\text{bias} \\

        s[t] &= v[t] \\geq \\vartheta \\

        v[t] &= v[t]\\,(1-s[t])

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
    current_decay : float or tuple
        the fraction of current decay per time step. If ``shared_param``
        is False, then it can be specified as a tuple (min_decay, max_decay).
    voltage_decay : float or tuple
        the fraction of voltage decay per time step. If ``shared_param`` is
        False, then it can be specified as a tuple (min_decay, max_decay).
    tau_grad : float, optional
        time constant of spike function derivative. Defaults to 1.
    scale_grad : float, optional
        scale of spike function derivative. Defaults to 1.
    scale : int, optional
        scale of the internal state. ``scale=1`` will result in values in the
        range expected from the of Loihi hardware. Defaults to 1<<6.
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
        flag to enable/disable persistent state between iterations. Defaults to
        False.
    requires_grad : bool, optional
        flag to enable/disable learning on neuron parameter. Defaults to False.
    graded_spike : bool, optional
        flag to enable/disable graded spike output. Defaults to False.
    num_features : int, optinal
        if the Hoyer neuron is behind the Conv, it is the number of feature channels; if behind the Dense, it should be 1. Defaults to 1.
    T: int, optinal
        the number of time steps. Defaults to 1.
    hoyer_type: str, optinal
        sum: the Hoyer Ext will be averaged across all feature channels; cw: the Hoyer Ext will be channel-wise. Defaults to sum.
    momentum: float, optinal
        the value used for the running_hoyer_ext computation. 
    """
    def __init__(
        self, threshold, current_decay, voltage_decay,
        tau_grad=1, scale_grad=1, scale=1 << 6,
        norm=None, dropout=None,
        shared_param=True, persistent_state=False, requires_grad=False, graded_spike=False,
        num_features=1, hoyer_mode=True, T=1, hoyer_type='sum', momentum=0.9, delay=False
    ):
        super(HoyerNeuron, self).__init__(
            threshold=threshold,
            current_decay=current_decay,
            voltage_decay=voltage_decay,
            tau_grad=tau_grad,
            scale_grad=scale_grad,
            scale=scale,
            norm=norm,
            dropout=dropout,
            shared_param=shared_param,
            persistent_state=persistent_state,
            requires_grad=requires_grad
        )
        
        # add some attributes for hoyer spiking
        # self.learnable_thr = nn.Parameter(torch.FloatTensor([self.threshold]), requires_grad=True)
        self.register_parameter(
            'learnable_thr',
            torch.nn.Parameter(
                torch.FloatTensor([self.threshold]),
                requires_grad=self.requires_grad
            ),
        )
        self.T = T
        self.hoyer_type = hoyer_type
        self.hoyer_mode = hoyer_mode
        self.num_features = num_features
        self.momentum = 0.9
        if self.num_features > 1:
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1,1), requires_grad=True)
            if self.hoyer_mode:
                self.bn = nn.BatchNorm2d(num_features=self.num_features)
        self.delay = delay

        if self.num_features > 1: 
                # Conv layer  B,C,H,W,T
                # self.register_buffer('running_hoyer_ext', torch.zeros([1, self.num_features, 1, 1, T], **factory_kwargs))
            if self.hoyer_type == 'sum':
                self.register_buffer('running_hoyer_ext', torch.zeros([1, 1, 1, 1, T]))
            else:
                self.register_buffer('running_hoyer_ext', torch.zeros([1, self.num_features, 1, 1, T]))
        else:
            # Linear layer  B,C,T
            self.register_buffer('running_hoyer_ext', torch.zeros([1, 1, T]))

        if norm is not None:
            if self.complex is False:
                self.norm = norm(num_features=num_features)
                if hasattr(self.norm, 'pre_hook_fx'):
                    self.norm.pre_hook_fx = self.quantize_8bit
            else:
                self.real_norm = norm(num_features=num_features)
                self.imag_norm = norm(num_features=num_features)
                if hasattr(self.real_norm, 'pre_hook_fx'):
                    self.real_norm.pre_hook_fx = self.quantize_8bit
                if hasattr(self.imag_norm, 'pre_hook_fx'):
                    self.imag_norm.pre_hook_fx = self.quantize_8bit
        else:
            self.norm = None
            if self.complex is True:
                self.real_norm = None
                self.imag_norm = None

        # self.register_buffer('ref_delay', torch.FloatTensor([ref_delay]))

        self.clamp()
    
    def thr_clamp(self):
        """Clamps the threshold value to
        :math:`[\\verb~1/scale~, \\infty)`."""
        self.learnable_thr.data.clamp_(1 / self.scale)

    def spike(self, voltage, hoyer_ext=1.0):
        """Extracts spike points from the voltage timeseries. It assumes the
        reset dynamics is already applied.

        Parameters
        ----------
        voltage : torch tensor
            neuron voltage dynamics
        hoyer_ext : torch tensor
            extra hoyer ext

        Returns
        -------
        torch tensor
            spike output

        """
        spike = HoyerSpike.apply(
            voltage,
            hoyer_ext,
            self.tau_rho * TAU_RHO_MULT,
            self.scale_rho * SCALE_RHO_MULT,
            self.graded_spike,
            self.voltage_state,
            # self.s_scale,
            1,
        )

        if self.persistent_state is True:
            with torch.no_grad():
                self.voltage_state = leaky_integrator.persistent_state(
                    self.voltage_state, spike[..., -1]
                ).detach().clone()

        if self.drop is not None:
            spike = self.drop(spike)

        return spike

    def cal_hoyer_loss(self, x, thr=None):
        if thr:
            x[x>thr] = thr
        x[x<0.0] = 0.0
        # avoid division by zero
        return (torch.sum(torch.abs(x))**2) / (torch.sum(x**2) + 1e-9) 

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
            spike response of the neuron.

        """
        if not self.hoyer_mode:
            out = super().forward(input)
            return out
        if self.num_features > 1 and hasattr(self, 'bn'):
            B,C,H,W,T = input.shape
            input = self.bn(input.permute(4,0,1,2,3).reshape(T*B,C, H, W).contiguous()).reshape(T,B,C,H,W).permute(1,2,3,4,0).contiguous()
        _, voltage = self.dynamics(input)
        self.hoyer_loss = self.cal_hoyer_loss(torch.clamp(voltage.clone(), min=0.0, max=1.0), 1.0)
        self.clamp()
        self.thr_clamp()
        voltage = voltage / self.learnable_thr
        if self.training:
            clamped_input = torch.clamp(voltage.clone().detach(), min=0.0, max=1.0)
            dim = tuple(range(clamped_input.ndim-1))
            if self.hoyer_type == 'sum':
                hoyer_ext = torch.sum(clamped_input**2, dim=dim) / (torch.sum(torch.abs(clamped_input), dim=dim))
            else:
                hoyer_ext = torch.sum((clamped_input)**2, dim=(0,2,3), keepdim=True) / torch.sum(torch.abs(clamped_input), dim=(0,2,3), keepdim=True)

            hoyer_ext = torch.nan_to_num(hoyer_ext, nan=1.0)
            with torch.no_grad():
                if self.delay:
                    # delay hoyer ext
                    self.running_hoyer_ext[..., 0] = 0
                    self.running_hoyer_ext = torch.roll(self.running_hoyer_ext, shifts=-1, dims=-1)
                    self.running_hoyer_ext = self.momentum * hoyer_ext + (1 - self.momentum) * self.running_hoyer_ext
                else:
                    # do not delay hoyer ext
                    self.running_hoyer_ext = self.momentum * hoyer_ext + (1 - self.momentum) * self.running_hoyer_ext
        else:
            hoyer_ext = self.running_hoyer_ext
        output = self.spike(voltage, hoyer_ext)
        return output
