# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Base block class"""

import numpy as np
import torch
import torch.nn.functional as F

from ..axon import delay

from lava.lib.dl.slayer.utils import recurrent


class AbstractInput(torch.nn.Module):
    """Abstract input block class. This should never be instantiated on its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    weight : float, optional
        weight for affine transform of input. None means no weight scaling.
        Defaults to None.
    bias : float, optional
        bias for affine transform of input. None means no bias shift.
        Defaults to None.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of
        average event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params=None,
        weight=None, bias=None, delay_shift=True, count_log=False
    ):
        super(AbstractInput, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = None

        self.count_log = count_log

        # These variables must be initialized by another abstract function
        self.neuron = None
        # self.synapse = None
        self.delay = None

        if weight is not None:
            self.register_parameter(
                'weight',
                torch.nn.Parameter(
                    torch.FloatTensor([weight]),
                    requires_grad=True
                )
            )  # learnable weight scaling factor
        else:
            self.weight = None

        if bias is not None:
            self.register_parameter(
                'bias',
                torch.nn.Parameter(
                    torch.FloatTensor([bias]),
                    requires_grad=True
                )
            )  # learnable weight scaling factor
        else:
            self.bias = None

        self.input_shape = None
        self.pre_hook_fx = lambda x: x
        self.delay_shift = delay_shift

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        if self.neuron is not None:
            z = self.pre_hook_fx(x)
            if self.weight is not None:
                z = z * self.pre_hook_fx(self.weight)
            if self.bias is not None:
                z = z + self.pre_hook_fx(self.bias)
            x = self.neuron(z)

        if self.delay_shift is True:
            x = delay(x, 1)

        if self.input_shape is None:
            if self.neuron is not None:
                self.input_shape = self.neuron.shape
            else:
                self.input_shape = x.shape[1:-1]

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        if self.neuron is not None:
            return self.neuron.shape
        else:
            return self.input_shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(w):
            return self.pre_hook_fx(
                w, descale=True
            ).reshape(w.shape[:2]).cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['input'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.shape))

        if self.neuron is not None:
            if self.weight is not None:
                handle.create_dataset('weight', data=weight(self.weight))
            if self.bias is not None:
                handle.create_dataset('bias', data=weight(self.bias))

            for key, value in self.neuron.device_params.items():
                handle.create_dataset(f'neuron/{key}', data=value)


class AbstractFlatten(torch.nn.Module):
    """Abstract flatten block class. This should never be instantiated on its
    own.

    Parameters
    ----------
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(self, count_log=False):
        super(AbstractFlatten, self).__init__()

        self.count_log = count_log
        self.shape = None

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        if self.shape is None:
            self.shape = [x.shape[1] * x.shape[2] * x.shape[3], 1, 1]

        if self.count_log is True:
            return x.reshape((x.shape[0], -1, x.shape[-1])), None
        else:
            return x.reshape((x.shape[0], -1, x.shape[-1]))

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        handle.create_dataset(
            'type', (1, ), 'S10', ['flatten'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.shape))


class AbstractAverage(torch.nn.Module):
    """Abstract average block class. This should never be instantiated on its
    own.

    Parameters
    ----------
    num_outputs : int
        number of output population groups.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of
        average event rate is returned. Defaults to False.
    """
    def __init__(self, num_outputs, count_log=False):
        super(AbstractAverage, self).__init__()

        self.count_log = count_log
        self.num_outputs = num_outputs

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        # N, _, _, _, T = x.shape
        N = x.shape[0]
        T = x.shape[-1]
        if self.count_log is True:
            return torch.mean(x.reshape((N, self.num_outputs, -1, T)), dim=2),\
                None
        else:
            return torch.mean(x.reshape((N, self.num_outputs, -1, T)), dim=2)

    @property
    def shape(self):
        """Shape of the block.
        """
        return torch.Size([self.num_outputs, 1, 1])

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        handle.create_dataset(
            'type', (1, ), 'S10', ['average'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))


class AbstractAffine(torch.nn.Module):
    """Abstract affine transform class. This should never be instantiated on
    its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights
        before synaptic operation. None means no transformation.
        Defaults to None.
    dynamics : bool, optional
        flag to enable neuron dynamics. If False, only the dendrite current
        is returned. Defaults to True.
    mask : bool array, optional
        boolean synapse mask that only enables relevant synapses. None
        means no masking is applied. Defaults to None.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of
        average event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_neurons, out_neurons,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        dynamics=True, mask=None, count_log=False
    ):
        super(AbstractAffine, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = {
            'in_neurons': in_neurons,
            'out_neurons': out_neurons,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.count_log = count_log
        self.dynamics = dynamics

        if mask is None:
            self.mask = None
        else:
            self.register_buffer(
                'mask',
                mask.reshape(mask.shape[0], mask.shape[1], 1, 1, 1)
            )

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        if self.mask is not None:
            if self.synapse.complex is True:
                self.synapse.real.weight.data *= self.mask
                self.synapse.imag.weight.data *= self.mask
            else:
                self.synapse.weight.data *= self.mask

        z = self.synapse(x)
        # x = self.neuron.dynamics(z)

        if self.dynamics is True:
            x = self.neuron.dynamics(z)
            x = x[1]  # voltage or imag state
        else:
            x = z[1] if self.synapse.complex is True else z

        if self.count_log is True:
            return x, None
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['dense'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset('inFeatures', data=self.synapse.in_channels)
        handle.create_dataset('outFeatures', data=self.synapse.out_channels)

        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        # for key, value in self.neuron.device_params.items():
        #     handle.create_dataset(f'neuron/{key}', data=value)


class AbstractTimeDecimation(torch.nn.Module):
    """Abstract time decimation block class. This should never be instantiated
    on its own.

    Parameters
    ----------
    factor : int, optional
        number of time units to decimate in a single bin. Must be in
        powers of 2.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(self, factor, count_log=False):
        super(AbstractTimeDecimation, self).__init__()

        self.count_log = count_log
        if factor == 0:
            raise AssertionError(
                f'Expected factor to be a positive integer. Found {factor=}.'
            )
        if (factor & (factor - 1)) != 0:
            raise AssertionError(
                f'Expected factor to be a power of 2. Found {factor=}.'
            )
        self.factor = int(factor)

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        pad_len = int(
            np.ceil(x.shape[-1] / self.factor) * self.factor - x.shape[-1]
        )
        x = F.pad(x, (0, pad_len))
        out_shape = list(x.shape)
        out_shape[-1] = int(x.shape[-1] / self.factor)
        # x = torch.sum(
        #         x.reshape(-1, x.shape[-1] // self.factor, self.factor),
        #         dim=-1
        #     ).reshape(out_shape)
        x = torch.mean(
            x.reshape(-1, x.shape[-1] // self.factor, self.factor),
            dim=-1
        ).reshape(out_shape)

        if self.count_log is True:
            return x, None
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return torch.Size([self.num_outputs, 1, 1])

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        handle.create_dataset(
            'type', (1, ), 'S10', ['average'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))


class AbstractDense(torch.nn.Module):
    """Abstract dense block class. This should never be instantiated on its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    delay : bool, optional
        flag to enable axonal delay. Defaults to False.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    mask : bool array, optional
        boolean synapse mask that only enables relevant synapses. None means no
        masking is applied. Defaults to None.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_neurons, out_neurons,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        delay=False, delay_shift=True, mask=None, count_log=False,
    ):
        super(AbstractDense, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = {
            'in_neurons': in_neurons,
            'out_neurons': out_neurons,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.count_log = count_log

        if mask is None:
            self.mask = None
        else:
            self.register_buffer(
                'mask',
                mask.reshape(mask.shape[0], mask.shape[1], 1, 1, 1)
            )

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None
        self.delay_shift = delay_shift

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        if self.mask is not None:
            if self.synapse.complex is True:
                self.synapse.real.weight.data *= self.mask
                self.synapse.imag.weight.data *= self.mask
            else:
                self.synapse.weight.data *= self.mask

        z = self.synapse(x)
        x = self.neuron(z)
        if self.delay_shift is True:
            x = delay(x, 1)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['dense'.encode('ascii', 'ignore')]
        )

        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset('inFeatures', data=self.synapse.in_channels)
        handle.create_dataset('outFeatures', data=self.synapse.out_channels)

        if self.synapse.weight_norm_enabled:
            self.synapse.disable_weight_norm()

        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )


class AbstractConv(torch.nn.Module):
    """Abstract convolution block class. This should never be instantiated on
    its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    in_features : int
        number of input features.
    out_features : int
        number of output features.
    kernel_size : int
        kernel size.
    stride : int or tuple of two ints, optional
        convolution stride. Defaults to 1.
    padding : int or tuple of two ints, optional
        convolution padding. Defaults to 0.
    dilation : int or tuple of two ints, optional
        convolution dilation. Defaults to 1.
    groups : int, optional
        number of blocked connections. Defaults to 1.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    delay : bool, optional
        flag to enable axonal delay. Defaults to False.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_features, out_features, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        delay=False, delay_shift=True, count_log=False
    ):
        super(AbstractConv, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = {
            'in_features': in_features,
            'out_features': out_features,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.count_log = count_log

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None
        self.delay_shift = delay_shift

    def forward(self, x):
        """Forward computation method. The input must be in ``NCHWT`` format.
        """
        z = self.synapse(x)
        x = self.neuron(z)
        if self.delay_shift is True:
            x = delay(x, 1)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:-1]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        # descriptors
        handle.create_dataset(
            'type', (1, ), 'S10', ['conv'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset('inChannels', data=self.synapse.in_channels)
        handle.create_dataset('outChannels', data=self.synapse.out_channels)
        handle.create_dataset('kernelSize', data=self.synapse.kernel_size[:-1])
        handle.create_dataset('stride', data=self.synapse.stride[:-1])
        handle.create_dataset('padding', data=self.synapse.padding[:-1])
        handle.create_dataset('dilation', data=self.synapse.dilation[:-1])
        handle.create_dataset('groups', data=self.synapse.groups)

        # weights
        if self.synapse.weight_norm_enabled:
            self.synapse.disable_weight_norm()
        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )


class AbstractPool(torch.nn.Module):
    """Abstract input block class. This should never be instantiated on its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    kernel_size : int or tuple of two ints
        size of pooling kernel.
    stride : int or tuple of two ints, optional
        stride of pooling operation. Defaults to None.
    padding : int or tuple of two ints, optional
        padding of pooling operation. Defaults to 0.
    dilation : int or tuple of two ints, optional
        dilation of pooling kernel. Defaults to 1.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    delay : bool, optional
        flag to enable axonal delay. Defaults to False.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, kernel_size,
        stride=None, padding=0, dilation=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        delay=False, delay_shift=True, count_log=False
    ):
        super(AbstractPool, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        if 'norm' in self.neuron_params.keys():
            self.neuron_params['norm'] = None
        # synapse parameters
        self.synapse_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.count_log = count_log

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None
        self.delay_shift = delay_shift

    def forward(self, x):
        """Forward computation method. The input must be in ``NCHWT`` format.
        """
        z = self.synapse(x)
        x = self.neuron(z)
        if self.delay_shift is True:
            x = delay(x, 1)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:-1]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        # descriptors
        handle.create_dataset(
            'type', (1, ), 'S10', ['pool'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset('kernelSize', data=self.synapse.kernel_size[:-1])
        handle.create_dataset('stride', data=self.synapse.stride[:-1])
        handle.create_dataset('padding', data=self.synapse.padding[:-1])
        handle.create_dataset('dilation', data=self.synapse.dilation[:-1])

        # weight
        if self.synapse.weight_norm_enabled:
            self.synapse.disable_weight_norm()
        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        # delay
        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)


class AbstractConvT(torch.nn.Module):
    """Abstract convolution Traspose block class. This should never be
    instantiated on its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    in_features : int
        number of input features.
    out_features : int
        number of output features.
    kernel_size : int
        kernel size.
    stride : int or tuple of two ints, optional
        convolutionT stride. Defaults to 1.
    padding : int or tuple of two ints, optional
        convolutionT padding. Defaults to 0.
    dilation : int or tuple of two ints, optional
        convolutionT dilation. Defaults to 1.
    groups : int, optional
        number of blocked connections. Defaults to 1.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    delay : bool, optional
        flag to enable axonal delay. Defaults to False.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_features, out_features, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        delay=False, delay_shift=True, count_log=False
    ):
        super(AbstractConvT, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = {
            'in_features': in_features,
            'out_features': out_features,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.count_log = count_log

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None
        self.delay_shift = delay_shift

    def forward(self, x):
        """
        """
        z = self.synapse(x)
        x = self.neuron(z)
        if self.delay_shift is True:
            x = delay(x, 1)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:-1]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        # descriptors
        handle.create_dataset(
            'type', (1, ), 'S10', ['convT'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset('inChannels', data=self.synapse.in_channels)
        handle.create_dataset('outChannels', data=self.synapse.out_channels)
        handle.create_dataset('kernelSize', data=self.synapse.kernel_size[:-1])
        handle.create_dataset('stride', data=self.synapse.stride[:-1])
        handle.create_dataset('padding', data=self.synapse.padding[:-1])
        handle.create_dataset('dilation', data=self.synapse.dilation[:-1])
        handle.create_dataset('groups', data=self.synapse.groups)

        # weights
        if self.synapse.weight_norm_enabled:
            self.synapse.disable_weight_norm()
        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )


class AbstractUnpool(torch.nn.Module):
    """Abstract Unpool block class. This should never be instantiated on its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    kernel_size : int or tuple of two ints
        size of unpooling kernel.
    stride : int or tuple of two ints, optional
        stride of unpooling operation. Defaults to None.
    padding : int or tuple of two ints, optional
        padding of unpooling operation. Defaults to 0.
    dilation : int or tuple of two ints, optional
        dilation of unpooling kernel. Defaults to 1.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    delay : bool, optional
        flag to enable axonal delay. Defaults to False.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, kernel_size,
        stride=None, padding=0, dilation=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        delay=False, delay_shift=True, count_log=False
    ):
        super(AbstractUnpool, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        if 'norm' in self.neuron_params.keys():
            self.neuron_params['norm'] = None
        # synapse parameters
        self.synapse_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.count_log = count_log

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None
        self.delay_shift = delay_shift

    def forward(self, x):
        """
        """
        z = self.synapse(x)
        x = self.neuron(z)
        if self.delay_shift is True:
            x = delay(x, 1)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:-1]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        # descriptors
        handle.create_dataset(
            'type', (1, ), 'S10', ['unpool'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset('kernelSize', data=self.synapse.kernel_size[:-1])
        handle.create_dataset('stride', data=self.synapse.stride[:-1])
        handle.create_dataset('padding', data=self.synapse.padding[:-1])
        handle.create_dataset('dilation', data=self.synapse.dilation[:-1])

        # weight
        if self.synapse.weight_norm_enabled:
            self.synapse.disable_weight_norm()
        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        # delay
        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)


class AbstractResidual(torch.nn.Module):
    pass


class AbstractKWTA(torch.nn.Module):
    """Abstract K-Winner-Takes-All block class. This should never be
    instantiated on its own. The formulation is described as below:

    .. math::
        s_\\text{out}[t] = f_s\\left(\\mathbf{W}\\,s_\\text{in}[t]
                           + \\mathbf{R}\\,s_{out}[t-1]
                           + \\alpha\\,(N-2K)\\right)\\\\
        \\mathbf{R} = \\begin{bmatrix}
        a &-1 &\\cdots &-1\\\\
        -1 & a &\\cdots &-1\\\\
        \\vdots &\\vdots &\\ddots &\\vdots\\\\
        -1 &-1 &\\cdots & a
        \\end{bmatrix},\\qquad |a| < 1

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    num_winners : [type]
        number of winners.
    self_excitation : float, optional
        self excitation factor. Defaults to 0.5.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    requires_grad : bool, optional
        flag for learnable recurrent synapse. Defaults to True.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_neurons, out_neurons,
        num_winners, self_excitation=0.5,
        weight_scale=1, weight_norm=False,
        pre_hook_fx=None, delay_shift=True, requires_grad=True,
        count_log=False
    ):
        super(AbstractKWTA, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = {
            'in_neurons': in_neurons,
            'out_neurons': out_neurons,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.num_neurons = out_neurons  # N
        self.num_winners = num_winners  # K
        self.requires_grad = requires_grad

        self.register_parameter(  # parameter: a
            'self_excitation',
            torch.nn.Parameter(
                torch.FloatTensor([self_excitation]),
                requires_grad=self.requires_grad
            )
        )

        self.count_log = count_log

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None
        self.bias = None
        self.register_buffer('spike_state', torch.zeros(1, dtype=torch.float))
        self.persistent_state = True
        self.neuron_params['persistent_state'] = True
        # make sure persistent state is always enabled
        self.delay_shift = delay_shift

    def clamp(self):
        self.self_excitation.data.clamp_(0, 1)

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        # s_\text{out}[t] = f_s\left(\mathbf{W}\,s_\text{in}[t]
        #               + \mathbf{R}\,s_{out}[t-1] + \alpha\,(N-2K)\right)\\
        # \mathbf{R} = \begin{bmatrix}
        # a &-1 &\cdots &-1\\
        # -1 & a &\cdots &-1\\
        # \vdots &\vdots &\ddots &\vdots\\
        # -1 &-1 &\cdots & a
        # \end{bmatrix},\qquad |a| < 1
        if self.bias is None:
            z = self.synapse(x)
        else:
            z = self.synapse(x) + self.bias
        x = torch.zeros_like(z).to(x.device)

        self.clamp()
        recurrent_weight = (
            (1 + self.self_excitation)
            * torch.eye(self.num_neurons).to(x.device) - 1
        )
        recurrent_bias = self.neuron.threshold * (
            self.num_neurons - 2 * self.num_winners
        ) / self.num_neurons / self.neuron.w_scale
        recurrent_bias = torch.FloatTensor([recurrent_bias]).to(x.device)
        if self.neuron.quantize_8bit is not None:
            recurrent_weight = self.neuron.quantize_8bit(recurrent_weight)
            recurrent_bias = self.neuron.quantize_8bit(recurrent_bias)

        spike = torch.zeros(z.shape[:-1]).to(x.device)
        if z.shape[0] == self.spike_state.shape[0]:
            spike = spike + self.spike_state

        for time in range(z.shape[-1]):
            dendrite = z[..., time:time + 1]
            feedback = F.linear(
                spike.reshape(x.shape[0], self.num_neurons),
                recurrent_weight
            ).reshape(dendrite.shape) + recurrent_bias  # or max psp
            spike = self.neuron(dendrite + feedback)
            x[..., time:time + 1] = spike

        # self.spike_state = spike.clone().detach().reshape(z.shape[:-1])

        if self.delay_shift is True:
            x = delay(x, 1)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        pass


class AbstractRecurrent(torch.nn.Module):
    """Abstract recurrent block class. This should never be instantiated on its
    own. The recurrent formulation is described below:

    .. math::
        s_\\text{out}[t] = f_s\\left(\\mathbf{W}\\,s_\\text{in}[t]
                        + \\mathbf{R}\\,s_{out}[t-1]\\right)

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter.Defaults to None.
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    requires_grad : bool, optional
        flag for learnable recurrent synapse. Defaults to True.
    delay : bool, optional
        flag to enable axonal delay. Defaults to False.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_neurons, out_neurons,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        requires_grad=True,
        delay=False, delay_shift=True, count_log=False
    ):
        super(AbstractRecurrent, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = {
            'in_neurons': in_neurons,
            'out_neurons': out_neurons,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.recurrent_params = {
            'in_neurons': out_neurons,
            'out_neurons': out_neurons,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.num_neurons = out_neurons
        self.requires_grad = requires_grad

        self.count_log = count_log

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.input_synapse = None
        self.recurrent_synapse = None
        self.delay = None
        self.bias = None
        self.register_buffer('spike_state', torch.zeros(1, dtype=torch.float))
        self.persistent_state = True
        self.neuron_params['persistent_state'] = True
        # make sure persistent state is always enabled
        self.delay_shift = delay_shift

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """
        # s_\text{out}[t] = f_s\left(\mathbf{W}\,s_\text{in}[t]
        #                   + \mathbf{R}\,s_{out}[t-1]\right)
        if self.bias is None:
            z = self.input_synapse(x)
        else:
            z = self.input_synapse(x) + self.bias

        spike = torch.zeros(z.shape[:-1]).to(x.device)

        if z.shape[0] == self.spike_state.shape[0]:
            spike = spike + self.spike_state

        x = recurrent.custom_recurrent(z, self.neuron, self.recurrent_synapse)

        self.spike_state = spike.clone().detach().reshape(z.shape[:-1])

        if self.delay_shift is True:
            x = delay(x, 1)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        pass
