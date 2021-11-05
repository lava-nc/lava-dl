# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Sigma Decoder implementation."""

import torch


class Sigma(torch.nn.Module):
    """Sigma decoder implementation.

    Parameters
    ----------
    persistent_state : bool
        flag to enable persistent state. Defaults to False.

    Atrributes
    ----------
    persistent_state
    shape : torch shape
        shape of the sigma unit. It is initialized on first run. The value is
        None before initialization.
    pre_state: torch tensor
        previous state of sigma unit.

    """
    def __init__(self, persistent_state=False):
        super(Sigma, self).__init__()

        self.persistent_state = persistent_state
        self.shape = None
        self.register_buffer(
            'pre_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )

    def forward(self, input):
        """
        """
        if self.shape is None:
            self.shape = input.shape[1:-1]
            assert len(self.shape) > 0, \
                f"Expected input to have at least 3 dimensions: "\
                f"[Batch, Spatial dims ..., Time]. "\
                f"It's shape is {input.shape}."
        else:
            assert input.shape[1:-1] == self.shape, \
                f'Input tensor shape ({input.shape}) '\
                f'does not match with Neuron shape ({self.shape}).'

        if self.pre_state.shape[0] != input.shape[0]:
            # persistent state cannot proceed due to change in batch dimension.
            # this likely indicates change from training to testing set
            self.pre_state = torch.zeros(
                input.shape[:-1]
            ).to(self.pre_state.dtype).to(self.pre_state.device)

        output = torch.cumsum(input, dim=-1)

        if self.persistent_state is True:
            output += torch.unsqueeze(
                self.pre_state,
                dim=len(self.pre_state.shape)
            )
            self.pre_state = output[..., -1].detach().clone()

        return output
