# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""ANN sampler module."""

import matplotlib.pyplot as plt
import numpy as np
import torch


class AnnSampler(torch.nn.Module):
    """ANN data point sampler. It samples the weighted spike input rate
    :math:`z = \\langle z[t]\\rangle` and
    neurons output spike rate (activation)
    :math:`a = \\langle s[t]\\rangle` data points, manages the data points,
    and provides picewise linear ANN activation.

    Parameters
    ----------
    num_centers : int
        number of sampling centers. Defaults to 5.
    sample_range : list
        min and max range of sampling points. Defaults to [0.1, 0.9].
    eps : float
        infinitesimal constant. Defaults to 1e-5.

    """
    def __init__(self, num_centers=5, sample_range=[0.1, 0.9], eps=1e-5):
        """
        """
        # num_centers must be less than half the number of time-steps
        # to properly sample data
        super(AnnSampler, self).__init__()
        self.z = np.array([])
        self.a = np.array([])
        self.slopes = 1
        self.eps = eps
        self.max = 1
        self.centers = None
        self.num_centers = num_centers
        self.sample_range = sample_range
        self.scale = 1
        self.alpha = 0.1

    def append(self, a, z):
        """Appends new data points

        Parameters
        ----------
        a : torch.tensor
            output spike tensor.
        z : torch.tensor
            weighted spike tensor.
        """
        a_rate = torch.mean(a, dim=-1).cpu().data.numpy().flatten()
        z_rate = torch.mean(z, dim=-1).cpu().data.numpy().flatten()
        self.a = np.append(self.a, a_rate)
        self.z = np.append(self.z, z_rate)

    def forward(self, z):
        """Picewise ANN activation

        Parameters
        ----------
        z : torch tensor
            weighted input for ANN.

        Returns
        -------
        torch tensor
            equivalent ANN activation.

        """
        return _pwl.apply(z, self.centers, self.slopes, self.max, self.eps)

    def clear(self):
        """Clears all data points."""
        self.z = np.array([])
        self.a = np.array([])

    def soft_clear(self):
        """Randomly clears half of the data points while retaining some of the
        history."""
        ind = np.random.permutation(len(self.z))[:len(self.z) // 2]
        self.z = np.delete(self.z, ind)
        self.a = np.delete(self.a, ind)

    def plot(self, figsize=None):
        """Plots the piecewise ANN activation.

        Parameters
        ----------
        figsize : tuple of 2
            width and height of figure. Defaults to None.
        """
        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        z = np.linspace(self.z.min(), self.z.max(), 1000)
        plt.plot(self.z, self.a, '.', alpha=0.05, label='data')

        a = self.forward(torch.FloatTensor(z))
        plt.plot(z, a.cpu().data.numpy(), label='slope fit')
        plt.plot(self.centers[:, 0], self.centers[:, 1], 'o')

        plt.legend()

    def fit(self):
        """Fit piecewise linear model from sampled data."""
        self.max = max(
            np.mean(self.a[self.a > self.sample_range[1] * self.a.max()]),
            0.8
        )
        num_centers = self.num_centers
        activation = np.linspace(
            self.sample_range[0] * self.max,
            self.sample_range[1] * self.max,
            num_centers + 1
        )
        centers = []
        for i in range(num_centers):
            ind = np.argwhere(
                (self.a > activation[i]) & (self.a <= activation[i + 1])
            )
            center_ind = None
            if self.centers is not None:
                center_ind = np.argwhere(
                    (self.centers[:, 1] > activation[i])
                    * (self.centers[:, 1] <= activation[i + 1])
                )
                center_ind = center_ind.flatten()
                if len(center_ind) == 0:
                    center_ind = None
            # print(len(ind))
            if len(ind) == 0 and center_ind is not None:
                centers.append([
                    self.centers[center_ind, 0].mean(),
                    self.centers[center_ind, 1].mean()
                ])
            elif len(ind) > 0 and center_ind is not None:
                centers.append([
                    self.alpha * np.median(self.z[ind])
                    + (1 - self.alpha) * self.centers[center_ind, 0].mean(),
                    self.alpha * np.median(self.a[ind])
                    + (1 - self.alpha) * self.centers[center_ind, 1].mean()
                ])
            elif len(ind) > 0 and center_ind is None:
                centers.append(
                    [np.median(self.z[ind]), np.median(self.a[ind])]
                )

        if len(centers) == 0:  # the best we can do here is unit slope
            centers.append([0, 0])
            centers.append([1, 1])
        elif len(centers) == 1:
            centers = [[0, 0], centers[0]]
        centers = np.array(centers)
        # print(centers)
        # print('y1 =', centers[1:, 1])
        # print('y2 =', centers[:-1, 1])

        self.slopes = (
            centers[1:, 1] - centers[:-1, 1]
        ) / (centers[1:, 0] - centers[:-1, 0])
        self.slopes = np.abs(np.concatenate(
            [[self.slopes[0]], self.slopes, [self.slopes[-1]]]
        ))
        self.centers = np.vstack((
            np.array([centers[0][0] - centers[0][1] / self.slopes[0], 0]),
            centers,
            np.array([
                centers[-1][0] + (self.max - centers[-1][1]) / self.slopes[-1],
                self.max
            ]),
        ))

        self.scale = self.centers[-1, 0] / self.centers[-1, 1]

        # print(self.centers)
        # print(self.slopes)
        # print(1 / np.mean(self.slopes[1:-1]))


class _pwl(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, z, centers, slopes, max, eps):
        """
        """
        a = torch.zeros_like(z)
        for i in range(len(centers) - 1):
            a = torch.where(
                (z > centers[i, 0]) & (z <= centers[i + 1, 0]),
                centers[i, 1] + slopes[i] * (z - centers[i, 0]),
                a
            )
        a = torch.where(z > centers[-1, 0], max + a, a)

        centers = torch.autograd.Variable(
            torch.tensor(centers, device=z.device, dtype=z.dtype),
            requires_grad=False
        )
        slopes = torch.autograd.Variable(
            torch.tensor(slopes, device=z.device, dtype=z.dtype),
            requires_grad=False
        )
        eps = torch.autograd.Variable(
            torch.tensor(eps, device=z.device, dtype=z.dtype),
            requires_grad=False
        )
        ctx.save_for_backward(z, centers, slopes, eps)

        return a

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        (z, centers, slopes, eps) = ctx.saved_tensors
        grad = torch.zeros_like(grad_output)
        for i in range(len(centers) - 1):
            grad = torch.where(
                (z > centers[i, 0]) & (z <= centers[i + 1, 0]),
                slopes[i],
                grad
            )
        grad *= centers[-1, 0] / centers[-1, 1]
        grad += eps

        return grad * grad_output, None, None, None, None
