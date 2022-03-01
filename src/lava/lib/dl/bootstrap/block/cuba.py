# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Bootstrap CUBA layer blocks."""

import lava.lib.dl.slayer.block.cuba as cuba_block
from .base import Mode, AbstractBlock, doc_modifier


class Input(cuba_block.Input, AbstractBlock):
    def __init__(self, *args, **kwargs):
        """
        """
        super(Input, self).__init__(*args, **kwargs)
        if self.neuron is None:
            raise AssertionError(
                f'Expected valid neuron in block. Found {self.neuron=}.'
            )

    def _forward_synapse(self, x):
        """Forward computation of synapse

        Args:
            x (torch tensor): input tensor.
        """
        z = self.pre_hook_fx(x)
        if self.weight is not None:
            z = z * self.pre_hook_fx(self.weight)
        if self.bias is not None:
            z = z + self.pre_hook_fx(self.bias)
        return z

    def forward(self, x, mode=Mode.ANN):
        """
        """
        return AbstractBlock.forward(self, x, mode)


Input.__doc__ = doc_modifier(cuba_block.Input.__doc__)


class Flatten(cuba_block.Flatten, AbstractBlock):
    def __init__(self, *args, **kwargs):
        """
        """
        super(Flatten, self).__init__(*args, **kwargs)

    def forward(self, x, mode=Mode.ANN):
        return cuba_block.Flatten.forward(self, x)


Flatten.__doc__ = doc_modifier(cuba_block.Flatten.__doc__)


class Affine(cuba_block.Affine, AbstractBlock):
    def __init__(self, *args, **kwargs):
        """
        """
        super(Affine, self).__init__(*args, **kwargs)

    def _forward_synapse(self, x):
        """Forward computation of synapse

        Args:
            x (torch tensor): input tensor.
        """
        if self.mask is not None:
            self.synapse.weight.data *= self.mask
        z = self.synapse(x)
        if self.neuron.shape is None:
            self.neuron.shape = z.shape[1:-1]
        return z

    def forward(self, x, mode=Mode.ANN):
        return self._forward_synapse(x)

    def fit(self):  # nothing to fit here
        return None


Affine.__doc__ = doc_modifier(cuba_block.Affine.__doc__)


class Dense(cuba_block.Dense, AbstractBlock):
    def __init__(self, *args, **kwargs):
        """
        """
        super(Dense, self).__init__(*args, **kwargs)

    def _forward_synapse(self, x):
        """Forward computation of synapse

        Args:
            x (torch tensor): input tensor.
        """
        if self.mask is not None:
            self.synapse.weight.data *= self.mask
        return self.synapse(x)

    def forward(self, x, mode=Mode.ANN):
        """
        """
        return AbstractBlock.forward(self, x, mode)


Dense.__doc__ = doc_modifier(cuba_block.Dense.__doc__)


class Conv(cuba_block.Conv, AbstractBlock):
    def __init__(self, *args, **kwargs):
        """
        """
        super(Conv, self).__init__(*args, **kwargs)

    def forward(self, x, mode=Mode.ANN):
        """
        """
        return AbstractBlock.forward(self, x, mode)


Conv.__doc__ = doc_modifier(cuba_block.Conv.__doc__)
