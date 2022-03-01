# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Abstract bootstrap layer blocks."""

import torch
import lava.lib.dl.slayer as slayer

from ..ann_sampler import AnnSampler
from ..routine import Mode


def doc_modifier(doc):
    """
    """
    return doc.replace(
        'delay_shift (bool, optional): flag to simulate spike propagation '
        'delay from one layer to next. Defaults to True.',
        'delay_shift (bool, optional): flag to simulate spike propagation '
        'delay from one layer to next. Defaults to False.'
    )


class AbstractBlock(torch.nn.Module):
    """Abstract bootstrap block"""
    def __init__(self, *args, **kwargs):
        """
        """
        super(AbstractBlock, self).__init__(*args, **kwargs)
        self.f = AnnSampler()
        self.mode = Mode.SAMPLE
        if 'delay_shift' not in kwargs.keys():
            # change the default mode to no-delay shift.
            self.delay_shift = False
        if 'delay' in kwargs.keys():
            # axonal delays are not allowed.
            self.delay = None
        if hasattr(self, 'synapse'):
            if self.synapse.complex is True:
                raise AssertionError(
                    f'Only real synapses are supported. '
                    f'Found {self.synapse.complex=}.'
                )

    def fit(self):
        """Fit the sampling points to estimate piecewise linear model."""
        self.f.fit()
        self.f.soft_clear()

    def _forward_synapse(self, x):
        """Forward computation of synapse

        Parameters
        ----------
        x : torch tensor
            input tensor.

        """
        return self.synapse(x)

    def _forward_ann(self, x):
        """Forward computation in ANN mode.
        """
        # TODO: Add time flattenning logic based on input's time dim
        z = self._forward_synapse(x)

        if self.neuron.norm is not None:
            z = self.neuron.norm(z)

        x = self.f(z)

        if self.neuron.drop is not None:
            x = self.neuron.drop(x)

        return x

    def _forward_snn(self, x, sample=False):
        """Forward computation in SNN mode.

        Parameters
        ----------
        x : torch tensor
            input tensor.
        sample : bool
            flag to enable SNN rate data points. Defaults to False.
        """
        # TODO: Add time expansion logic based on input's time dim
        z = self._forward_synapse(x)
        x = self.neuron(z)
        if sample is True:
            self.f.append(x, z)

        if self.delay_shift is True:
            x = slayer.axon.delay(x, 1)

        return x

    def forward(self, x, mode=Mode.ANN):
        """Forward calculation block

        Parameters
        ----------
        x : torch tensor
            input tensor.
        mode : optional
            forward operation mode. Can be one of
            ``Mode.SNN` | `Mode.ANN` | `Mode.SAMPLE``.
            Defaults to Mode.ANN.

        Returns
        -------
        torch tensor
            output tensor.

        """
        if mode == Mode.ANN:
            x = self._forward_ann(x)
        elif mode == Mode.FIT:
            self.fit()
            x = self._forward_ann(x)
        elif mode == Mode.SNN:
            x = self._forward_snn(x)
        elif mode == Mode.SAMPLE:
            x = self._forward_snn(x, sample=True)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x
