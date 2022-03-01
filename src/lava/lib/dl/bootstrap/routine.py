# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""ANN-SNN mode switching routine helper."""

from enum import Enum
import lava.lib.dl.slayer as slayer


class Mode(Enum):
    """Enum constants for different mode of operation. Valid modes are
    `SNN` | `ANN` | `SAMPLE | FIT`.
    """
    SNN = 0
    ANN = 1
    SAMPLE = 2
    FIT = 3


class LayerMode:
    """Iterator that iterates layer/block's mode of operation

    Parameters
    ----------
    crossover : int
        point below which layer is always in ``mode.SNN``.
    base_mode : enum
        global mode of operation. Options are
        ``mode.{SNN | ANN | SAMPLE | FIT}``.
    """
    def __init__(self, crossover, base_mode):
        """
        """
        self.crossover = crossover
        self.base_mode = base_mode
        self.count = 0

    def __str__(self):
        mode_str = f'{self.base_mode}'.split('.')[-1]
        if self.crossover == 0:
            crossover_str = ''
        else:
            crossover_str = f'Crossover at: {self.crossover}'
        return f'Mode: {mode_str}' + crossover_str

    def __iter__(self):
        """mode iterator
        """
        while True:
            if self.base_mode == Mode.SAMPLE:
                yield Mode.SAMPLE
            elif self.count < self.crossover:
                yield Mode.SNN
            else:
                yield self.base_mode

            self.count += 1


class Scheduler:
    """Hybrid mode iterator

    Parameters
    ----------
    num_sample_iter : int
        number of iteration to sample data. Defaults to 10.
    sample_period : int
        epoch interval to initiate data sampling. Defaults to 10.
    crossover_epochs : list
        list of ints that define crossover landmarks. None means no landmarks.
        Defaults to None.

    Returns
    -------

    """
    def __init__(
        self, num_sample_iter=10, sample_period=10, crossover_epochs=None
    ):
        self.num_sample_iter = num_sample_iter
        self.sample_period = sample_period
        self.crossover_epochs = crossover_epochs
        self.crossover = 0
        self.snn_stat = slayer.utils.LearningStat()

    def mode(self, epoch, iteration, train=True):
        """Block operation mode generator.

        Parameters
        ----------
        epoch : int
            current epoch.
        iteration : int
            current iteration.
        train : bool
            training or evaluation mode flag. Defaults to True.

        Returns
        -------
        iterator
            operation mode iterator

        """
        if train:
            # crossover only changes in training mode
            if self.crossover_epochs is not None:
                if epoch in self.crossover_epochs and iteration == 0:
                    self.crossover += 1
            if (
                iteration < self.num_sample_iter
                and epoch % self.sample_period == 0
            ):
                return LayerMode(self.crossover, Mode.SAMPLE)
            if (
                iteration == self.num_sample_iter
                and epoch % self.sample_period == 0
            ):
                return LayerMode(self.crossover, Mode.FIT)
            return LayerMode(self.crossover, Mode.ANN)
        else:
            if epoch % self.sample_period == 0:
                return LayerMode(self.crossover, Mode.SNN)
            return LayerMode(self.crossover, Mode.ANN)

    def sync_snn_stat(self, stat):
        """Sync SNN stat.

        Parameters
        ----------
        stat : slayer.utils.LearningStat
            learning stat to sync.

        Returns
        -------

        """
        self.snn_stat.num_samples = stat.num_samples
        self.snn_stat.loss_sum = stat.loss_sum
        self.snn_stat.correct_samples = stat.correct_samples

    def update_snn_stat(self):
        """Update snn leraning statistics."""
        self.snn_stat.update()
        self.snn_stat.reset()
