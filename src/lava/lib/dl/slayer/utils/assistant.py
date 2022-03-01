# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Assistant utility for automatically load network from network
description."""

import torch


class Assistant:
    """Assistant that bundles training, validation and testing workflow.

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
        the main device memory where network is placed. It is not at start and
        gets initialized on the first call.
    """
    def __init__(
        self,
        net, error, optimizer,
        stats=None, classifier=None, count_log=False,
        lam=None
    ):
        self.net = net
        self.error = error
        self.optimizer = optimizer
        self.classifier = classifier
        self.stats = stats
        self.count_log = count_log
        self.lam = lam
        self.device = None

    def reduce_lr(self, factor=10 / 3):
        """Reduces the learning rate of the optimizer by ``factor``.

        Parameters
        ----------
        factor : float
            learning rate reduction factor. Defaults to 10/3.

        Returns
        -------

        """
        for param_group in self.optimizer.param_groups:
            print('\nLearning rate reduction from', param_group['lr'])
            param_group['lr'] /= factor

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
            network's output.
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

        count = None
        if self.count_log is True:
            if self.lam is None:
                output, count = self.net(input)
            else:
                output, net_loss, count = self.net(input)
        else:
            if self.lam is None:
                output = self.net(input)
            else:
                output, net_loss = self.net(input)

        loss = self.error(output, target)

        if self.stats is not None:
            self.stats.training.num_samples += input.shape[0]
            self.stats.training.loss_sum += loss.cpu().data.item() \
                * output.shape[0]
            if self.classifier is not None:   # classification
                self.stats.training.correct_samples += torch.sum(
                    self.classifier(output) == target
                ).cpu().data.item()

        if self.lam is not None:  # add net_loss before backward step
            loss += self.lam * net_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if count is None:
            return output

        return output, count

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
            network's output.
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

            count = None
            if self.count_log is True:
                if self.lam is None:
                    output, count = self.net(input)
                else:
                    output, _, count = self.net(input)
            else:
                if self.lam is None:
                    output = self.net(input)
                else:
                    output, _ = self.net(input)

            loss = self.error(output, target)

            if self.stats is not None:
                self.stats.testing.num_samples += input.shape[0]
                self.stats.testing.loss_sum += loss.cpu().data.item() \
                    * output.shape[0]
                if self.classifier is not None:   # classification
                    self.stats.testing.correct_samples += torch.sum(
                        self.classifier(output) == target
                    ).cpu().data.item()

            if count is None:
                return output

            return output, count

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
            network's output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.eval()

        with torch.no_grad():
            device = self.net.device
            input = input.to(device)
            target = target.to(device)

            count = None
            if self.count_log is True:
                if self.lam is None:
                    output, count = self.net(input)
                else:
                    output, _, count = self.net(input)
            else:
                if self.lam is None:
                    output = self.net(input)
                else:
                    output, _ = self.net(input)

            loss = self.error(output, target)

            if self.stats is not None:
                self.stats.validation.num_samples += input.shape[0]
                if self.lam is None:
                    self.stats.validation.loss_sum += loss.cpu().data.item() \
                        * output.shape[0]
                else:
                    self.stats.validation.loss_sum += loss.cpu().data.item() \
                        * output.shape[0]
                if self.classifier is not None:   # classification
                    self.stats.validation.correct_samples += torch.sum(
                        self.classifier(output) == target
                    ).cpu().data.item()

            if count is None:
                return output

            return output, count
