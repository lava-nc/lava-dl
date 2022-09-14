# Copyright : (c) UC Regents, Emre Neftci, 2022 Intel Corporation
# Licence : GPLv3

"""Assistant utility for automatically load network from network
description."""

import torch
from lava.lib.dl.slayer.utils.assistant import Assistant


class DECOLLEAssistant(Assistant):
    """Assistant that bundles training, validation and testing workflow
        for DECOLLE models

    Parameters
    ----------
    net : torch.nn.Module
        network to train.
    error : object or lambda
        an error object or a lambda function that evaluates error.
        It is expected to take ``(output, target)`` | ``(output, label)``
        as it's argument and return a scalar value.
    optimizer : torch optimizer
        the learning optimizer.
    stats : slayer.utils.stats
        learning stats logger. If None, stats will not be logged.
        Defaults to None.
    classifier : slayer.classifier or lambda
        classifier object or lambda function that takes output and
        returns the network prediction. None means regression mode.
        Classification steps are bypassed.
        Defaults to None.
    count_log : bool
        flag to enable count log. Defaults to False.
    lam : float
        lagrangian to merge network layer based loss.
        None means no such additional loss.
        If not None, net is expected to return the accumulated loss as second
        argument. It is intended to be used with layer wise sparsity loss.
        Defaults to None.
   training_mode : str, one of "online" or "batch"
        perform gradient descent at every time-step ("online") as in the
        original paper, or after presentation of a batch of examples.
        Empirically, "online" is expected to perform better, but is slower.

    Attributes
    ----------
    net
    error
    optimizer
    stats
    classifier
    count_log
    lam
    device : torch.device or None
        the main device memory where network is placed. It is None at start and
        gets initialized on the first call.
    """
    def __init__(
            self,
            net, error, optimizer,
            stats=None, classifier=None, count_log=False,
            lam=None, training_mode='online'
    ):

        super(DECOLLEAssistant, self).__init__(net, error, optimizer,
                                               stats, classifier,
                                               count_log, lam)
        if training_mode not in ['online', 'batch']:
            print("training_mode should be one of 'online' or 'batch'")
            raise ValueError
        self.training_mode = training_mode

    def train(self, input, target):
        """Training assistant.

        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.

        Returns
        -------
        output
            network's last readout layer output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.train()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        input = input.to(device)
        target = target.to(device)

        if self.count_log:
            count = [0. for _ in range(len(self.net.blocks))]
        else:
            count = None

        # reset net + burnin
        input = self.net.init_state(input)

        if self.training_mode == 'online':
            readout = torch.Tensor()
            for t in range(input.shape[-1]):
                x = input[..., t].unsqueeze(-1)
                spikes, readouts_t, voltages, count_t = self.net(x)
                loss = self.error(readouts_t, voltages, target)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                readout = torch.cat((readout, readouts_t[-1]), dim=-1)
                if self.count_log:
                    count = [count[i] + count_t[i] / input.shape[-1]
                             for i in range(len(count_t))]
                if self.stats is not None:
                    self.stats.training.loss_sum \
                        += loss.cpu().data.item() * readouts_t[-1].shape[0]
        else:
            spikes, readouts, voltages, count = self.net(input)
            loss = self.error(readouts, voltages, target)
            loss.backward()

            readout = readouts[-1]
            if self.stats is not None:
                self.stats.training.loss_sum\
                    += loss.cpu().data.item() * readout[-1].shape[0]
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.stats is not None:
            self.stats.training.num_samples += input.shape[0]
            if self.classifier is not None:   # classification
                self.stats.training.correct_samples += torch.sum(
                    self.classifier(readouts) == target
                ).cpu().data.item()

        if count is None:
            return readout

        return readout, count

    def test(self, input, target):
        """Testing assistant.

        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.

        Returns
        -------
        output
            network's last readout layer output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.eval()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)

            if self.count_log:
                count = [0. for _ in range(len(self.net.blocks))]
            else:
                count = None

            # reset net + burnin
            input = self.net.init_state(input)

            if self.training_mode == 'online':
                readout = torch.Tensor()
                for t in range(input.shape[-1]):
                    x = input[..., t].unsqueeze(-1)
                    spikes, readouts_t, voltages, count_t = self.net(x)
                    loss = self.error(readouts_t, voltages, target)

                    readout = torch.cat((readout, readouts_t[-1]), dim=-1)
                    if self.count_log:
                        count = [count[i] + count_t[i] / input.shape[-1]
                                 for i in range(len(count_t))]
                    if self.stats is not None:
                        self.stats.testing.loss_sum \
                            += loss.cpu().data.item() * readouts_t[-1].shape[0]
            else:
                spikes, readouts, voltages, count = self.net(input)
                loss = self.error(readouts, voltages, target)

                readout = readouts[-1]
                if self.stats is not None:
                    self.stats.testing.loss_sum \
                        += loss.cpu().data.item() * readout[-1].shape[0]

            if self.stats is not None:
                self.stats.testing.num_samples += input.shape[0]
                self.stats.testing.loss_sum \
                    += loss.cpu().data.item() * readout.shape[0]
                if self.classifier is not None:   # classification
                    self.stats.testing.correct_samples += torch.sum(
                        self.classifier(readout) == target
                    ).cpu().data.item()

            if count is None:
                return readout

            return readout, count

    def valid(self, input, target):
        """Validation assistant.

        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.

        Returns
        -------
        output
            network's last readout layer output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.eval()

        with torch.no_grad():
            device = self.net.device
            input = input.to(device)
            target = target.to(device)

            if self.count_log:
                count = [0. for _ in range(len(self.net.blocks))]
            else:
                count = None

            # reset net + burnin
            input = self.net.init_state(input)

            if self.training_mode == 'online':
                readout = torch.Tensor()
                for t in range(input.shape[-1]):
                    x = input[..., t].unsqueeze(-1)
                    spikes, readouts_t, voltages, count_t = self.net(x)
                    loss = self.error(readouts_t, voltages, target)

                    readout = torch.cat((readout, readouts_t[-1]), dim=-1)
                    if self.count_log:
                        count = [count[i] + count_t[i] / input.shape[-1]
                                 for i in range(len(count_t))]
                    if self.stats is not None:
                        self.stats.validation.loss_sum \
                            += loss.cpu().data.item() * readouts_t[-1].shape[0]
            else:
                spikes, readouts, voltages, count = self.net(input)
                loss = self.error(readouts, voltages, target)

                readout = readouts[-1]
                if self.stats is not None:
                    self.stats.validation.loss_sum \
                        += loss.cpu().data.item() * readout[-1].shape[0]

            if self.stats is not None:
                self.stats.validation.num_samples += input.shape[0]
                if self.lam is None:
                    self.stats.validation.loss_sum \
                        += loss.cpu().data.item() * readout.shape[0]
                else:
                    self.stats.validation.loss_sum \
                        += loss.cpu().data.item() * readout.shape[0]
                if self.classifier is not None:   # classification
                    self.stats.validation.correct_samples += torch.sum(
                        self.classifier(readout) == target
                    ).cpu().data.item()

            if count is None:
                return readout

            return readout, count
