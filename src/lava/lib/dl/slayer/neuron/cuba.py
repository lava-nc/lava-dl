# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""CUBA neuron model."""

import numpy as np
import torch

from . import base
from .dynamics import leaky_integrator
from ..spike.spike import Spike
from ..utils import quantize


# These are tuned heuristically so that scale_grad=1 and tau_grad=1 serves
# as a good starting point

# SCALE_RHO_MULT = 0.1
# TAU_RHO_MULT = 10
SCALE_RHO_MULT = 0.1
TAU_RHO_MULT = 100
# SCALE_RHO_MULT = 1
# TAU_RHO_MULT = 1


def neuron_params(device_params, scale=1 << 6, p_scale=1 << 12):
    """Translates device parameters to neuron parameters.

    Parameters
    ----------
    device_params : dictionary
        dictionary of device parameter specification.

    scale : int
        neuron scale value. Default value = 1 << 6.
    p_scale : int
        parameter scale value. Default value = 1 << 12

    Returns
    -------
    dictionary
        dictionary of neuron parameters that can be used to initialize neuron
        class.
    """
    return {
        'threshold': device_params['vThMant'] / scale,
        'current_decay': device_params['iDecay'] / p_scale,
        'voltage_decay': device_params['vDecay'] / p_scale,
    }


class Neuron(base.Neuron):
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
    """
    def __init__(
        self, threshold, current_decay, voltage_decay,
        tau_grad=1, scale_grad=1, scale=1 << 6,
        norm=None, dropout=None,
        shared_param=True, persistent_state=False, requires_grad=False,
        graded_spike=False
    ):
        super(Neuron, self).__init__(
            threshold=threshold,
            tau_grad=tau_grad,
            scale_grad=scale_grad,
            p_scale=1 << 12,
            w_scale=scale,
            s_scale=scale * (1 << 6),
            norm=norm,
            dropout=dropout,
            persistent_state=persistent_state,
            shared_param=shared_param,
            requires_grad=requires_grad
        )

        self.graded_spike = graded_spike
        # slayer dynamics and spike mechanism spike if voltage >= vth to have
        # better estimate of surrogate gradient.
        # but Loihi spikes if voltage > vth
        # so threshold_eps is added to get the same effect in hardware.
        self.threshold_eps = 0.01 / self.s_scale

        if self.shared_param is True:
            if np.isscalar(current_decay) is False:
                raise AssertionError(
                    f'Expected current decay to be a scalar when shared_param '
                    f'is True. Found {current_decay=}.'
                )
            if np.isscalar(voltage_decay) is False:
                raise AssertionError(
                    f'Expected voltage decay to be a scalar when shared_param '
                    f'is True. Found {voltage_decay=}.'
                )
            self.register_parameter(
                'current_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * current_decay]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'voltage_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * voltage_decay]),
                    requires_grad=self.requires_grad,
                )
            )
        else:
            if np.isscalar(current_decay) is True:  # 1% jitter for now
                self.current_decay_min = current_decay * 0.99
                self.current_decay_max = current_decay * 1.01
            else:
                if len(current_decay) != 2:
                    raise AssertionError(
                        f'Expected current decay to be of length 2 i.e. '
                        f'[min, max]. Found {current_decay=}.'
                    )
                self.current_decay_min = current_decay[0]
                self.current_decay_max = current_decay[1]

            if np.isscalar(voltage_decay) is True:
                self.voltage_decay_min = voltage_decay * 0.99
                self.voltage_decay_max = voltage_decay * 1.01
            else:
                if len(voltage_decay) != 2:
                    raise AssertionError(
                        f'Expected voltage decay to be of length 2 i.e. '
                        f'[min, max]. Found {voltage_decay=}.'
                    )
                self.voltage_decay_min = voltage_decay[0]
                self.voltage_decay_max = voltage_decay[1]

            self.register_parameter(
                'current_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * self.current_decay_min]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'voltage_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * self.voltage_decay_min]),
                    requires_grad=self.requires_grad,
                )
            )

        # self.register_buffer('ref_delay', torch.FloatTensor([ref_delay]))

        self.register_buffer(
            'current_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )
        self.register_buffer(
            'voltage_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )

        self.clamp()

    def clamp(self):
        """A function to clamp the sin decay and cosine decay parameters to be
        within valid range. The user will generally not need to call this
        function.
        """
        with torch.no_grad():
            self.current_decay.data.clamp_(0, self.p_scale)
            self.voltage_decay.data.clamp_(0, self.p_scale)

    @property
    def device(self):
        """The device memory (cpu/cuda) where the object lives."""
        return self.current_decay.device

    @property
    def cx_current_decay(self):
        """The compartment current decay parameter to be used for configuring
        Loihi hardware."""
        self.clamp()
        val = quantize(self.current_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def cx_voltage_decay(self):
        """The compartment voltage decay parameter to be used for configuring
        Loihi hardware."""
        self.clamp()
        val = quantize(self.voltage_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def ref_delay(self):
        """Refractory delay."""
        # ref_delay of 1 is assumed for now
        return 1

    @property
    def scale(self):
        """Scale difference between slayer representation and hardware
        representation of the variable states."""
        return self.w_scale

    @property
    def device_params(self):
        """Dictionary of device parameters."""
        return {
            'type': 'CUBA',
            'iDecay': self.cx_current_decay,
            'vDecay': self.cx_voltage_decay,
            'vThMant': self.v_th_mant,
            'refDelay': self.ref_delay,
            'gradedSpike': self.graded_spike,
        }

    def dynamics(self, input):
        """Computes the dynamics (without spiking behavior) of the neuron
        instance to an input. The input shape must match with the neuron shape.
        For the first time, the neuron shape is determined from the input
        automatically.

        Parameters
        ----------
        input : torch tensor
            Input tensor.

        Returns
        -------
        torch tensor
            current response of the neuron.
        torch tensor
            voltage response of the neuron.
        """
        if self.shape is None:
            self.shape = input.shape[1:-1]
            if len(self.shape) == 0:
                raise AssertionError(
                    f"Expected input to have at least 3 dimensions: "
                    f"[Batch, Spatial dims ..., Time]. "
                    f"It's shape is {input.shape}."
                )
            self.num_neurons = np.prod(self.shape)
            if self.shared_param is False:
                current_decay = self.current_decay_min \
                    + (self.current_decay_max - self.current_decay_min)\
                    * torch.rand(
                        self.shape[0],
                        dtype=torch.float
                    ).to(self.device)
                voltage_decay = self.voltage_decay_min \
                    + (self.voltage_decay_max - self.voltage_decay_min)\
                    * torch.rand(
                        self.shape[0],
                        dtype=torch.float
                    ).to(self.device)
                self.current_decay.data = self.p_scale * current_decay
                self.voltage_decay.data = self.p_scale * voltage_decay

                del self.current_decay_min
                del self.current_decay_max
                del self.voltage_decay_min
                del self.voltage_decay_max
        else:
            if input.shape[1:-1] != self.shape:
                raise AssertionError(
                    f'Input tensor shape ({input.shape}) does not match with '
                    f'Neuron shape ({self.shape}).'
                )

        dtype = self.current_state.dtype
        device = self.current_state.device
        if self.current_state.shape[0] != input.shape[0]:
            # persistent state cannot proceed due to change in batch dimension.
            # this likely indicates change from training to testing set
            self.current_state = torch.zeros(
                input.shape[:-1]
            ).to(dtype).to(device)
            self.voltage_state = torch.zeros(
                input.shape[:-1]
            ).to(dtype).to(device)

        if self.current_state.shape[1:] != self.shape:
            # this means current_state and voltage_state are not initialized
            # properly
            self.current_state = self.current_state * torch.ones(
                input.shape[:-1]
            ).to(dtype).to(device)
            self.voltage_state = self.voltage_state * torch.ones(
                input.shape[:-1]
            ).to(dtype).to(device)

        # clamp the values only when learning is enabled
        # This means we don't need to clamp the values after gradient update.
        # It is done in runtime now. Might be slow, but overhead is negligible.
        if self.requires_grad is True:
            self.clamp()

        current = leaky_integrator.dynamics(
            input,
            quantize(self.current_decay),
            self.current_state.contiguous(),
            self.s_scale,
            debug=self.debug
        )

        if self.norm is not None:
            current = self.norm(current)

        voltage = leaky_integrator.dynamics(
            current,  # bias can be enabled by adding it here
            quantize(self.voltage_decay),
            self.voltage_state.contiguous(),
            self.s_scale,
            self.threshold + self.threshold_eps,
            debug=self.debug
        )

        if self.persistent_state is True:
            with torch.no_grad():
                # The state at last time step needs to be cloned. Otherwise,
                # the previous result will be affected since
                # it will use same memory space.
                self.current_state = current[..., -1].clone()
                self.voltage_state = voltage[..., -1].clone()

        return current, voltage

    def spike(self, voltage):
        """Extracts spike points from the voltage timeseries. It assumes the
        reset dynamics is already applied.

        Parameters
        ----------
        voltage : torch tensor
            neuron voltage dynamics

        Returns
        -------
        torch tensor
            spike output

        """
        spike = Spike.apply(
            voltage,
            self.threshold + self.threshold_eps,
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
        _, voltage = self.dynamics(input)
        return self.spike(voltage)
