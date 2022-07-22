# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Delta encoder implementation."""

import torch
from ..utils import quantize, QUANTIZE_MODE


class _DeltaUnit(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(
        ctx,
        input, threshold,
        pre_state, residual_state, error_state,
        cum_error,
        tau_grad, scale_grad
    ):
        """
        """
        # ignoring the cumulative error and residual state,
        # the delta forward unit can be formulated as
        # delta_x[t] = x[t] - x[t-1]
        # y[t] = delta_x[t] * H(|delta_x[t]| - threshold)
        output = torch.zeros_like(input)
        delta_input = torch.zeros_like(input)
        error = error_state
        if cum_error is True:
            for t in range(input.shape[-1]):
                delta = input[..., t] - pre_state + residual_state
                delta_input[..., t] = delta
                error += delta
                output[..., t] = torch.where(
                    torch.abs(error) >= threshold,
                    delta,
                    0 * delta
                ).to(input.dtype)
                error *= 1 - (torch.abs(output[..., t]) > 0)
                residual_state = (delta - output[..., t]).detach()
                pre_state = input[..., t].detach()
        else:
            for t in range(input.shape[-1]):
                delta = input[..., t] - pre_state + residual_state
                delta_input[..., t] = delta
                output[..., t] = torch.where(
                    torch.abs(delta) >= threshold,
                    delta,
                    0 * delta
                ).to(input.dtype)
                residual_state = (delta - output[..., t]).detach()
                pre_state = input[..., t].detach()

        ctx.save_for_backward(
            delta_input, threshold,
            torch.autograd.Variable(
                torch.tensor(
                    tau_grad,
                    device=input.device,
                    dtype=input.dtype
                ),
                requires_grad=False
            ),
            torch.autograd.Variable(
                torch.tensor(
                    scale_grad,
                    device=input.device,
                    dtype=input.dtype
                ),
                requires_grad=False
            ),
        )

        return output, residual_state, error

    @staticmethod
    def backward(ctx, grad_output, *_):
        """
        """
        # the backward computation is
        # delta_val = delta(|delta_x[t]| - threshold)
        # grad_threshold[t] = (- delta_x[t] * delta_val) * grad_y[t]
        # grad_delta_x[t] = (|delta_x[t]| * delta_val) * grad_y[t]
        # grad_x[t] = grad_delta_x[t] - grad_delta_x[t+1]
        delta_input, threshold, tau_grad, scale_grad = ctx.saved_tensors

        # different relaxation options
        # nascent_delta = lambda x: scale_grad * torch.sinc(x/tau_grad)
        def nascent_delta(x):
            return scale_grad * torch.exp(
                -torch.abs(x) / tau_grad
            )

        # nascent_delta = lambda x: scale_grad *
        #   (1 - torch.clamp(torch.abs(x), max=tau_grad) / tau_grad)
        delta_sub = torch.abs(delta_input) - threshold
        grad_threshold = torch.where(
            delta_sub > 0,
            -delta_input * nascent_delta(delta_sub) / threshold,
            -torch.sign(delta_input)
        ) * grad_output
        grad_delta_input = grad_output
        grad_input = torch.zeros_like(grad_output)
        grad_input[..., :-1] = grad_delta_input[..., :-1] \
            - grad_delta_input[..., 1:]
        grad_input[..., -1] = grad_delta_input[..., -1]

        if threshold.shape[0] == 1:  # shared_param is true
            grad_threshold = torch.unsqueeze(torch.sum(grad_threshold), dim=0)
        else:
            # shared_param is false. In this case,
            # the threshold needs to be reduced along
            # all dimensions except channel
            grad_threshold = torch.sum(
                grad_threshold.reshape(
                    delta_input.shape[0], delta_input.shape[1], -1
                ),
                dim=[0, -1]
            ).reshape(threshold.shape[:-1])
            # threshold dim was unsqueezed in the last dim

        return grad_input, grad_threshold, None, None, None, None, None, None


class Delta(torch.nn.Module):
    """Implements delta differential encoding followed by thresholding.
    The thresholds are learnable, individually or as a group.

    .. math::
        \\Delta x[t] &= x[t] - x[t-1] + r[t-1] \\\\
        y[t] &= \\begin{cases}
            \\Delta x[t] &\\text{ if } \\Delta x[t] \\geq \\vartheta \\\\
            0 &\\text{ otherwise}
        \\end{cases}\\\\
        r[t] &= \\Delta x[t] - y[t]

    For cumulative error, output evaluation is changed to

    .. math::
        e[t] &= e[t] + \\Delta x[t]\\\\
        y[t] &= \\begin{cases}
            \\Delta x[t] &\\text{ if } e[t] \\geq \\vartheta \\\\
            0 &\\text{ otherwise}\\\\
        e[t] &= e[t] * (1 - \\mathcal{H}(|y[t]|))
        \\end{cases}

    Parameters
    ----------
    threshold : float
        threshold value.
    scale : int
        quantization step size. Defaults to 64.
    tau_grad : float
        threshold gradient relaxation parameter. Defaults to 1.
    scale_grad : float
        threshold gradient scaling parameter. Defaults to 1.
    cum_error : bool
        flag to enable cumulative error before thresholding.
        Defaults to False.
    shared_param : bool
        flag to enable shared threshold. Defaults to True.
    persistent_state : bool
        flag to enable persistent delta states. Defaults to False.
    requires_grad : bool
        flag to enable threshold gradient. Defaults to False.

    Attributes
    ----------
    scale
    tau_grad
    scale_grad
    cum_error
    shared_param
    persistent_state
    requires_grad
    shape: torch shape
        shape of delta block. It is identified on runtime. The value is None
        before that.
    pre_state: torch tensor
        previous state of delta unit.
    residual_state : torch tensor
        residual state of delta unit.
    error_state : torch tensor
        error state of delta unit.

    Examples
    --------
        >> delta = Delta(threshold=1)
        >> y = delta(x) # differential threshold encoding

    """
    def __init__(
        self, threshold, scale=(1 << 6),
        tau_grad=1, scale_grad=1,
        cum_error=False, shared_param=True, persistent_state=False,
        requires_grad=False
    ):
        super(Delta, self).__init__()

        self.scale = scale
        self.tau_grad = tau_grad
        self.scale_grad = scale_grad
        self.cum_error = cum_error
        self.shared_param = shared_param
        self.persistent_state = persistent_state
        self.requires_grad = requires_grad
        self.shape = None
        self.register_buffer(
            'pre_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )
        self.register_buffer(
            'residual_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )
        self.register_buffer(
            'error_state',
            torch.zeros(1, dtype=torch.float),
            persistent=False
        )

        self.register_parameter(
            'threshold',
            torch.nn.Parameter(
                torch.FloatTensor([threshold]),
                requires_grad=self.requires_grad
            ),
        )

    def clamp(self):
        """Clamps the threshold value to
        :math:`[\\verb~1/scale~, \\infty)`."""
        self.threshold.data.clamp_(1 / self.scale)

    @property
    def device(self):
        """Device property of object

        Parameters
        ----------

        Returns
        -------
        torch.device
            returns the device memory where the object lives.

        """
        # return self.inv_threshold.device
        return self.threshold.device

    def forward(self, input):
        """
        """
        if self.shape is None:
            self.shape = input.shape[1:-1]
            if len(self.shape) == 0:
                raise AssertionError(
                    f"Expected input to have at least 3 dimensions: "
                    f"[Batch, Spatial dims ..., Time]. "
                    f"It's shape is {input.shape}."
                )
            if self.shared_param is False:
                self.threshold.data = self.threshold.data \
                    * torch.ones(self.shape[0]).to(self.device)
                self.threshold.data = self.threshold.data.reshape(
                    [self.shape[0]] + [1 for _ in self.shape[1:]]
                )
        else:
            if input.shape[1:-1] != self.shape:
                raise AssertionError(
                    f'Input tensor shape ({input.shape}) does not match with'
                    f'Neuron shape ({self.shape}).'
                )

        if self.pre_state.shape[0] != input.shape[0]:
            # persistent state cannot proceed due to change in batch dimension.
            # this likely indicates change from training to testing set
            self.pre_state = torch.zeros(input.shape[:-1]).to(
                self.pre_state.dtype
            ).to(self.pre_state.device)
            self.error_state = torch.zeros(input.shape[:-1]).to(
                self.error_state.dtype
            ).to(self.error_state.device)
            self.residual_state = torch.zeros(input.shape[:-1]).to(
                self.residual_state.dtype
            ).to(self.residual_state.device)

        self.clamp()

        if self.persistent_state is True:
            _pre_state = self.pre_state
            _residual_state = self.residual_state
        else:
            _pre_state = torch.zeros_like(input[..., 0])
            _residual_state = torch.zeros_like(input[..., 0])

        if self.cum_error is True:
            _error_state = self.error_state
        else:
            _error_state = torch.zeros_like(input[..., 0])

        output, residual_state, error_state = _DeltaUnit.apply(
            input, self.threshold,
            _pre_state,
            _residual_state,
            _error_state,
            self.cum_error, self.tau_grad, self.scale_grad
        )

        if self.persistent_state is True:
            self.pre_state = input[..., -1].clone().detach()
            self.residual_state = residual_state.clone().detach()
            if self.cum_error is True:
                self.error_state = error_state.clone().detach()

        return quantize(output, step=1 / self.scale, mode=QUANTIZE_MODE.FLOOR)
