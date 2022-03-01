# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Spike/event Input Output and visualization module."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class Event():
    """This class provides a way to store, read, write and visualize spike
    event.

    Members:
        * x (numpy int array): x index of spike event.
        * y (numpy int array): y index of spike event
          (not used if the spatial dimension is 1).
        * c (numpy int array): channel index of spike event.
        * t (numpy double array): timestamp of spike event.
          Time is assumed to be in ms.
        * p (numpy int or double array): payload of spike event.
          None for binary spike.
        * graded (bool): flag to indicate graded or binary spike.

    Parameters
    ----------
    x_event : int array
        x location of event
    y_event : int array or None
        y location of event
    c_event : int array
        c location of event
    t_event : int array or float array
        time of event
    payload : int array or float array or None
        payload of event. None for binary event. Defaults to None.

    Examples
    --------

    >>> td_event = Event(x_event, y_event, c_event, t_event)
    """
    def __init__(self, x_event, y_event, c_event, t_event, payload=None):
        if y_event is None:
            self.dim = 1
        else:
            self.dim = 2
        self.graded = payload is not None
        if type(x_event) is np.ndarray:  # x spatial dimension
            self.x = x_event
        else:
            self.x = np.asarray(x_event)

        if type(y_event) is np.ndarray:  # y spatial dimension
            self.y = y_event
        else:
            self.y = np.asarray(y_event)

        if type(c_event) is np.ndarray:  # channel dimension
            self.c = c_event
        else:
            self.c = np.asarray(c_event)

        if type(t_event) is np.ndarray:  # time stamp in ms
            self.t = t_event
        else:
            self.t = np.asarray(t_event)

        if self.graded:
            if type(payload) is np.ndarray:
                self.p = payload
            else:
                self.p = np.asarray(payload)
        else:
            self.p = None

        if not issubclass(self.x.dtype.type, np.integer):
            self.x = self.x.astype('int')
        if not issubclass(self.c.dtype.type, np.integer):
            self.c = self.c.astype('int')

        if self.dim == 2:
            if not issubclass(self.y.dtype.type, np.integer):
                self.y = self.y.astype('int')

        self.c -= self.c.min()

    def to_tensor(self, sampling_time=1, dim=None):	 # Sampling time in ms
        """Returns a numpy tensor that contains the spike events sampled in
        bins of ``sampling_time``. The array is of dimension
        (channels, height, time) or``CHT`` for 1D data. The array is of
        dimension (channels, height, width, time) or``CHWT`` for 2D data.

        Parameters
        ----------
        sampling_time : int
            event data sampling time. Defaults to 1.
        dim : int or None
            desired dimension. It is inferred if None. Defaults to None.

        Returns
        -------
        np array
            spike tensor.

        Examples
        --------

        >>> spike = td_event.to_tensor()
        """
        if self.dim == 1:
            if dim is None:
                dim = (
                    np.round(max(self.c) + 1).astype(int),
                    np.round(max(self.x) + 1).astype(int),
                    np.round(max(self.t) / sampling_time + 1).astype(int),
                )
            frame = np.zeros((dim[0], 1, dim[1], dim[2]))
        elif self.dim == 2:
            if dim is None:
                dim = (
                    np.round(max(self.c) + 1).astype(int),
                    np.round(max(self.y) + 1).astype(int),
                    np.round(max(self.x) + 1).astype(int),
                    np.round(max(self.t) / sampling_time + 1).astype(int),
                )
            frame = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        return self.fill_tensor(frame, sampling_time).reshape(dim)

    def fill_tensor(
        self, empty_tensor,
        sampling_time=1, random_shift=False, binning_mode='OR'
    ):  # Sampling time in ms
        """Returns a numpy tensor that contains the spike events sampled in
        bins of ``sampling_time``. The tensor is of dimension
        (channels, height, width, time) or``CHWT``.

        Parameters
        ----------
        empty_tensor : numpy or torch tensor
            an empty tensor to hold spike data .
        sampling_time : float
            the width of time bin. Defaults to 1.
        random_shift : bool
            flag to randomly shift the sample in time. Defaults to False.
        binning_mode : str
            the way spikes are binned. Options are 'OR'|'SUM'. If the event is
            graded binning mode is overwritten to 'SUM'. Defaults to 'OR'.

        Returns
        -------
        numpy or torch tensor
            spike tensor.

        Examples
        --------

        >>> spike = td_event.fill_tensor( torch.zeros((2, 240, 180, 5000)) )
        """
        if random_shift is True:
            t_start = np.random.randint(
                max(
                    int(self.t.min() / sampling_time),
                    int(self.t.max() / sampling_time) - empty_tensor.shape[3],
                    empty_tensor.shape[3] - int(self.t.max() / sampling_time),
                    1,
                )
            )
        else:
            t_start = 0

        x_event = np.round(self.x).astype(int)
        c_event = np.round(self.c).astype(int)
        t_event = np.round(self.t / sampling_time).astype(int) - t_start
        if self.graded:
            payload = self.p
            binning_mode = 'SUM'

        # print('shifted sequence by', t_start)

        if self.dim == 1:
            valid_ind = np.argwhere(
                (x_event < empty_tensor.shape[2])
                & (c_event < empty_tensor.shape[0])
                & (t_event < empty_tensor.shape[3])
                & (x_event >= 0)
                & (c_event >= 0)
                & (t_event >= 0)
            )
            if binning_mode.upper() == 'OR':
                empty_tensor[
                    c_event[valid_ind],
                    0,
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] = payload if self.graded is True else 1 / sampling_time
            elif binning_mode.upper() == 'SUM':
                empty_tensor[
                    c_event[valid_ind],
                    0,
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] += payload if self.graded is True else 1 / sampling_time
            else:
                raise Exception(
                    f'Unsupported binning_mode. It was {binning_mode}'
                )

        elif self.dim == 2:
            y_event = np.round(self.y).astype(int)
            valid_ind = np.argwhere(
                (x_event < empty_tensor.shape[2])
                & (y_event < empty_tensor.shape[1])
                & (c_event < empty_tensor.shape[0])
                & (t_event < empty_tensor.shape[3])
                & (x_event >= 0)
                & (y_event >= 0)
                & (c_event >= 0)
                & (t_event >= 0)
            )

            if binning_mode.upper() == 'OR':
                empty_tensor[
                    c_event[valid_ind],
                    y_event[valid_ind],
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] = payload if self.graded is True else 1 / sampling_time
            elif binning_mode.upper() == 'SUM':
                empty_tensor[
                    c_event[valid_ind],
                    y_event[valid_ind],
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] += payload if self.graded is True else 1 / sampling_time
            else:
                raise Exception(
                    'Unsupported binning_mode. It was {binning_mode}'
                )

        return empty_tensor

    def _show_event_1d(
        self, fig=None, frame_rate=24,
        pre_compute_frames=True, repeat=False, plot=True
    ):
        """
        """
        _ = pre_compute_frames  # just for consistency
        if self.dim != 1:
            raise Exception(
                'Expected self dimension to be 1. It was: {}'.format(self.dim)
            )
        if fig is None:
            fig = plt.figure()

        interval = 1e3 / frame_rate					# in ms
        x_dim = self.x.max() + 1
        t_max = self.t.max()
        t_min = self.t.min()
        # c_max = self.c.max() + 1
        min_frame = int(np.floor(t_min / interval))
        max_frame = int(np.ceil(t_max / interval)) + 1

        # ignore pre_compute_frames

        raster, = plt.plot([], [], '.')
        scan_line, = plt.plot([], [])
        plt.axis((t_min - 0.1 * t_max, 1.1 * t_max, -0.1 * x_dim, 1.1 * x_dim))

        def animate(i):
            """
            """
            t_end = (i + min_frame + 1) * interval
            ind = (self.t < t_end)
            # update raster
            raster.set_data(self.t[ind], self.x[ind])
            # update raster scan line
            scan_line.set_data(
                [t_end + interval, t_end + interval], [0, x_dim]
            )

        anim = animation.FuncAnimation(
            fig, animate, frames=max_frame,
            interval=interval, repeat=repeat
        )

        if plot is True:
            plt.show()
        return anim

    def _show_event_1d_graded(
        self, fig=None, frame_rate=24,
        pre_compute_frames=True, repeat=False, plot=True
    ):
        """
        """
        _ = pre_compute_frames  # just for consistency
        if self.dim != 1:
            raise Exception(
                'Expected self dimension to be 1. It was: {}'.format(self.dim)
            )
        if self.graded is not True:
            raise Exception(
                'Expected graded events. It was: {}'.format(self.graded)
            )
        if fig is None:
            fig = plt.figure()

        interval = 1e3 / frame_rate					# in ms
        x_dim = self.x.max() + 1
        t_max = self.t.max()
        t_min = self.t.min()
        # c_max = self.c.max() + 1
        p_min = self.p.min()
        p_max = self.p.max()
        min_frame = int(np.floor(t_min / interval))
        max_frame = int(np.ceil(t_max / interval)) + 1

        # ignore pre_compute_frames
        cmap = plt.get_cmap("tab10")
        scatter = plt.scatter([], [], [])
        scan_line, = plt.plot([], [], color=cmap(1))
        plt.axis((t_min - 0.1 * t_max, 1.1 * t_max, -0.1 * x_dim, 1.1 * x_dim))

        def animate(i):
            """
            """
            t_end = (i + min_frame + 1) * interval
            ind = (self.t < t_end)
            # update raster
            alpha = (self.p[ind] - p_min) / (p_max - p_min)
            # scatter.set_data(self.t[ind], self.x[ind], alpha*50, alpha=alpha)
            scatter.set_offsets(np.vstack([self.t[ind], self.x[ind]]).T)
            scatter.set_sizes(alpha * 50)
            scatter.set_alpha(alpha)
            # update raster scan line
            scan_line.set_data(
                [t_end + interval, t_end + interval], [0, x_dim]
            )

        anim = animation.FuncAnimation(
            fig, animate, frames=max_frame,
            interval=interval, repeat=repeat
        )

        if plot is True:
            plt.show()

        return anim

    def _show_event_2d(
        self, fig=None, frame_rate=24,
        pre_compute_frames=True, repeat=False, plot=True
    ):
        """
        """
        if self.dim != 2:
            raise Exception(
                'Expected self dimension to be 2. It was: {}'.format(self.dim)
            )
        if fig is None:
            fig = plt.figure()

        interval = 1e3 / frame_rate					# in ms
        x_dim = self.x.max() + 1
        y_dim = self.y.max() + 1

        if pre_compute_frames is True:
            min_frame = int(np.floor(self.t.min() / interval))
            max_frame = int(np.ceil(self.t.max() / interval))
            image = plt.imshow(np.zeros((y_dim, x_dim, 3)))
            frames = np.zeros((max_frame - min_frame, y_dim, x_dim, 3))

            # precompute frames
            for i in range(len(frames)):
                t_start = (i + min_frame) * interval
                t_end = (i + min_frame + 1) * interval
                time_mask = (self.t >= t_start) & (self.t < t_end)
                r_ind = (time_mask & (self.c == 1))
                g_ind = (time_mask & (self.c == 2))
                b_ind = (time_mask & (self.c == 0))
                frames[i, self.y[r_ind], self.x[r_ind], 0] = 1
                frames[i, self.y[g_ind], self.x[g_ind], 1] = 1
                frames[i, self.y[b_ind], self.x[b_ind], 2] = 1

            def animate(frame):
                """
                """
                image.set_data(frame)
                return image

            anim = animation.FuncAnimation(
                fig, animate,
                frames=frames, interval=interval, repeat=repeat
            )

        else:
            min_frame = int(np.floor(self.t.min() / interval))
            max_frame = int(np.ceil(self.t.max() / interval))
            image = plt.imshow(np.zeros((y_dim, x_dim, 3)))

            def animate(i):
                """
                """
                t_start = (i + min_frame) * interval
                t_end = (i + min_frame + 1) * interval
                frame = np.zeros((y_dim, x_dim, 3))
                time_mask = (self.t >= t_start) & (self.t < t_end)
                r_ind = (time_mask & (self.c == 1))
                g_ind = (time_mask & (self.c == 2))
                b_ind = (time_mask & (self.c == 0))
                frame[self.y[r_ind], self.x[r_ind], 0] = 1
                frame[self.y[g_ind], self.x[g_ind], 1] = 1
                frame[self.y[b_ind], self.x[b_ind], 2] = 1
                image.set_data(frame)
                return image

            anim = animation.FuncAnimation(
                fig, animate,
                frames=max_frame - min_frame,
                interval=interval, repeat=repeat
            )

        # save the animation as an mp4. This requires ffmpeg or mencoder to be
        # installed. The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5. You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        # if saveAnimation: anim.save('show_self_animation.mp4', fps=30)

        if plot is True:
            plt.show()

        return anim

    def show(
        self, fig=None, frame_rate=24,
        pre_compute_frames=True, repeat=False
    ):
        """Visualizes spike event.

        Parameters
        ----------
        fig : int
            plot figure ID. Defaults to None.
        frame_rate : int
            frame rate of visualization. Defaults to 24.
        pre_compute_frames : bool
            flag to enable precomputation of frames for faster visualization.
            Defaults to True.
        repeat : bool
            flag to enable repeat of animation. Defaults to False.

        Examples
        --------

        >>> self.show()
        """
        if fig is None:
            fig = plt.figure()
        if self.dim == 1:
            if self.graded is True:
                self._show_event_1d_graded(
                    fig, frame_rate=frame_rate,
                    pre_compute_frames=pre_compute_frames,
                    repeat=repeat
                )
            else:
                self._show_event_1d(
                    fig, frame_rate=frame_rate,
                    pre_compute_frames=pre_compute_frames,
                    repeat=repeat
                )
        else:
            self._show_event_2d(
                fig, frame_rate=frame_rate,
                pre_compute_frames=pre_compute_frames,
                repeat=repeat
            )

    def anim(
        self, fig=None, frame_rate=24,
        pre_compute_frames=True, repeat=True
    ):
        """Get animation object for spike event.

        Parameters
        ----------
        fig : int
            plot figure ID. Defaults to None.
        frame_rate : int
            frame rate of visualization. Defaults to 24.
        pre_compute_frames : bool
            flag to enable precomputation of frames for faster visualization.
            Defaults to True.
        repeat : bool
            flag to enable repeat of animation. Defaults to False.

        Returns
        -------
        anim
            matplotlib anim object.

        Examples
        --------

        >>> anim = self.anim()
        """
        if fig is None:
            fig = plt.figure()
        if self.dim == 1:
            if self.graded is True:
                anim = self._show_event_1d_graded(
                    fig,
                    frame_rate=frame_rate,
                    pre_compute_frames=pre_compute_frames,
                    repeat=repeat,
                    plot=False
                )
            else:
                anim = self._show_event_1d(
                    fig,
                    frame_rate=frame_rate,
                    pre_compute_frames=pre_compute_frames,
                    repeat=repeat,
                    plot=False
                )
        else:
            anim = self._show_event_2d(
                fig,
                frame_rate=frame_rate,
                pre_compute_frames=pre_compute_frames,
                repeat=repeat,
                plot=False
            )

        plt.close(anim._fig)
        return anim


def tensor_to_event(spike_tensor, sampling_time=1):
    """Returns td_event event from a numpy array (of dimension 3 or 4).
    The numpy array must be of dimension (channels, height, time) or ``CHT``
    for 1D data.
    The numpy array must be of dimension (channels, height, width, time) or
    ``CHWT`` for 2D data.

    Parameters
    ----------
    spike_tensor : numpy or torch tensor
        spike tensor.
    sampling_time : float
        the width of time bin. Defaults to 1.

    Returns
    -------
    Event
        spike event

    Examples
    --------

    >>> td_event = tensor_to_Event(spike)
    """
    if spike_tensor.ndim == 3:
        spike_event = np.argwhere(spike_tensor != 0)
        x_event = spike_event[:, 1]
        y_event = None
        c_event = spike_event[:, 0]
        t_event = spike_event[:, 2]
        payload = spike_tensor[
            spike_event[:, 0],
            spike_event[:, 1],
            spike_event[:, 2]
        ] * sampling_time
    elif spike_tensor.ndim == 4:
        spike_event = np.argwhere(spike_tensor != 0)
        x_event = spike_event[:, 2]
        y_event = spike_event[:, 1]
        c_event = spike_event[:, 0]
        t_event = spike_event[:, 3]
        payload = spike_tensor[
            spike_event[:, 0],
            spike_event[:, 1],
            spike_event[:, 2],
            spike_event[:, 3]
        ] * sampling_time
    else:
        raise Exception(
            f'Expected numpy array of 3 or 4 dimension. '
            f'It was {spike_tensor.ndim}'
        )

    if np.abs(payload - np.ones_like(payload)).sum() < 1e-6:  # binary spikes
        return Event(x_event, y_event, c_event, t_event * sampling_time)
    else:
        return Event(
            x_event, y_event, c_event, t_event * sampling_time, payload
        )


def read_1d_spikes(filename):
    """Reads one dimensional binary spike file and returns a td_event event.

    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 16 bits (bits 39-24) represent the neuronID.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * the last 23 bits (bits 22-0) represent the spike event timestamp in
            microseconds.

    Parameters
    ----------
    filename : str
        name of spike file.

    Returns
    -------
    Event
        spike event.

    Examples
    --------

    >>> td_event = read_1d_spikes(file_path)
    """
    with open(filename, 'rb') as input_file:
        input_byte_array = input_file.read()
    input_as_int = np.asarray([x for x in input_byte_array])
    x_event = (input_as_int[0::5] << 8) | input_as_int[1::5]
    c_event = input_as_int[2::5] >> 7
    t_event = (
        (input_as_int[2::5] << 16)
        | (input_as_int[3::5] << 8)
        | (input_as_int[4::5])
    ) & 0x7FFFFF

    # convert spike times to ms
    return Event(x_event, None, c_event, t_event / 1000)


def encode_1d_spikes(filename, td_event):
    """Writes one dimensional binary spike file from a td_event event.

    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 16 bits (bits 39-24) represent the neuronID.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * the last 23 bits (bits 22-0) represent the spike event timestamp in
          microseconds.

    Parameters
    ----------
    filename : str
        name of spike file.
    td_event : event
        spike event object

    Examples
    --------

    >>> encode_1d_spikes(file_path, td_event)
    """
    if td_event.dim != 1:
        raise Exception(
            'Expected td_event dimension to be 1. '
            'It was: {}'.format(td_event.dim)
        )
    x_event = np.round(td_event.x).astype(int)
    c_event = np.round(td_event.c).astype(int)
    # encode spike time in us
    t_event = np.round(td_event.t * 1000).astype(int)
    output_byte_array = bytearray(len(t_event) * 5)
    output_byte_array[0::5] = np.uint8((x_event >> 8) & 0xFF00).tobytes()
    output_byte_array[1::5] = np.uint8((x_event & 0xFF)).tobytes()
    output_byte_array[2::5] = np.uint8(
        ((t_event >> 16) & 0x7F)
        | (c_event.astype(int) << 7)
    ).tobytes()
    output_byte_array[3::5] = np.uint8((t_event >> 8) & 0xFF).tobytes()
    output_byte_array[4::5] = np.uint8(t_event & 0xFF).tobytes()
    with open(filename, 'wb') as output_file:
        output_file.write(output_byte_array)


def read_2d_spikes(filename):
    """Reads two dimensional binary spike file and returns a td_event event.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.

    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * The last 23 bits (bits 22-0) represent the spike event timestamp in
          microseconds.

    Parameters
    ----------
    filename : str
        name of spike file.

    Returns
    -------
    Event
        spike event.

    Examples
    --------

    >>> td_event = read_2d_spikes(file_path)
    """
    with open(filename, 'rb') as input_file:
        input_byte_array = input_file.read()
    input_as_int = np.asarray([x for x in input_byte_array])
    x_event = input_as_int[0::5]
    y_event = input_as_int[1::5]
    c_event = input_as_int[2::5] >> 7
    t_event = (
        (input_as_int[2::5] << 16)
        | (input_as_int[3::5] << 8)
        | (input_as_int[4::5])
    ) & 0x7FFFFF

    # convert spike times to ms
    return Event(x_event, y_event, c_event, t_event / 1000)


def encode_2d_spikes(filename, td_event):
    """Writes two dimensional binary spike file from a td_event event.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.

    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * The last 23 bits (bits 22-0) represent the spike event timestamp in
          microseconds.

    Parameters
    ----------
    filename : str
        name of spike file.
    td_event : event
        spike event object

    Examples
    --------

    >>> encode_2d_spikes(file_path, td_event)
    """
    if td_event.dim != 2:
        raise Exception(
            'Expected td_event dimension to be 2. '
            'It was: {}'.format(td_event.dim)
        )
    x_event = np.round(td_event.x).astype(int)
    y_event = np.round(td_event.y).astype(int)
    c_event = np.round(td_event.c).astype(int)
    # encode spike time in us
    t_event = np.round(td_event.t * 1000).astype(int)
    output_byte_array = bytearray(len(t_event) * 5)
    output_byte_array[0::5] = np.uint8(x_event).tobytes()
    output_byte_array[1::5] = np.uint8(y_event).tobytes()
    output_byte_array[2::5] = np.uint8(
        ((t_event >> 16) & 0x7F)
        | (c_event.astype(int) << 7)
    ).tobytes()
    output_byte_array[3::5] = np.uint8((t_event >> 8) & 0xFF).tobytes()
    output_byte_array[4::5] = np.uint8(t_event & 0xFF).tobytes()
    with open(filename, 'wb') as output_file:
        output_file.write(output_byte_array)


def read_np_spikes(filename, fmt='xypt', time_unit=1e-3):
    """Reads numpy spike event and returns a td_event event.
    The numpy array is assumed to be of nEvent x event dimension.

    Parameters
    ----------
    filename : str
        name of spike file.
    fmt : str
        format of numpy event ordering. Options are 'xypt'. Defaults to 'xypt'.
    time_unit : float
        scale factor that converts the data to seconds. Defaults to 1e-3.

    Returns
    -------
    Event
        spike object.

    Examples
    --------

    >>> td_event = read_np_spikes(file_path)
    >>> td_event = read_np_spikes(file_path, fmt='xypt')
    >>> td_event = read_np_spikes(file_path, time_unit=1e-6)
    """
    np_event = np.load(filename)
    if fmt == 'xypt':
        if np_event.shape[1] == 3:
            return Event(
                np_event[:, 0].astype('int'),
                None,
                np_event[:, 1],
                np_event[:, 2] * time_unit * 1e3
            )
        elif np_event.shape[1] == 4:
            return Event(
                np_event[:, 0],
                np_event[:, 1],
                np_event[:, 2],
                np_event[:, 3] * time_unit * 1e3
            )
        else:
            raise Exception(
                'Numpy array format did not match. '
                'Expected it to be nEvents x eventd_eventim.'
            )
    else:
        raise Exception(f"{fmt=} not implemented.")
    # TODO: modify for graded spikes


def encode_np_spikes(filename, td_event, fmt='xypt', time_unit=1e-3):
    """Writes td_event event into numpy file.

    Parameters
    ----------
    filename : str
        name of spike file.
    td_event : event
        spike event.
    fmt : str
        format of numpy event ordering. Options are 'xypt'. Defaults to 'xypt'.
    time_unit : float
        scale factor that converts the data to seconds. Defaults to 1e-3.

    Examples
    --------

    >>> encode_np_spikes(file_path, td_event)
    >>> encode_np_spikes(file_path, td_event, fmt='xypt')
    """
    if fmt == 'xypt':
        if td_event.dim == 1:
            np_event = np.zeros((len(td_event.x), 3))
            np_event[:, 0] = td_event.x
            np_event[:, 1] = td_event.c
            np_event[:, 2] = td_event.t
        elif td_event.dim == 2:
            np_event = np.zeros((len(td_event.x), 4))
            np_event[:, 0] = td_event.x
            np_event[:, 1] = td_event.y
            np_event[:, 2] = td_event.c
            np_event[:, 3] = td_event.t
        else:
            raise Exception(
                'Numpy array format did not match. '
                'Expected it to be nEvents x eventd_eventim.'
            )
    else:
        raise Exception(f"{fmt=} not implemented.")
    np.save(filename, np_event)
    # TODO: modify for graded spikes
