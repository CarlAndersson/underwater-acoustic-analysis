import numpy as np
import scipy.signal
from . import positional, _core
import itertools
import abc


class Data(_core.Leaf):
    dims = tuple()

    def __init__(self, data, dims=None, **kwargs):
        super().__init__(**kwargs)
        self._data = np.asarray(data)

        if dims is not None:
            self.dims = dims
        if len(self.dims) != self.data.ndim:
            raise ValueError('The number of dimensions in the data does not match the number of expected axes')

    def copy(self, deep=False, **kwargs):
        obj = super().copy(**kwargs)
        if deep:
            obj._data = self.data.copy()
        else:
            obj._data = self.data
        obj.dims = self.dims
        return obj

    def reduce(self, function, dim, *args, _new_class=None, **kwargs):
        if isinstance(dim, (int, str)):
            dim = [dim]
        reduce_axes = []
        for ax in dim:
            if isinstance(ax, str):
                ax = self.dims.index(ax)
            reduce_axes.append(ax)
        reduce_axes = tuple(reduce_axes)
        new_dims = tuple(dim for idx, dim in enumerate(self.dims) if idx not in reduce_axes)

        if isinstance(function, _core.Reduction):
            function = function.function

        out = function(self.data, axis=reduce_axes, *args, **kwargs)
        out = _core.NodeOperation.wrap_output(out, self, _new_class=_new_class)
        out.dims = new_dims
        return out


class Time(_core.TimeLeafin, Data):
    dims = ('time',)

    def __init__(self, data, samplerate, start_time, downsampling=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.samplerate = samplerate
        self.downsampling = downsampling
        # TODO: parse the start time here as well, if it's a string.
        self._start_time = start_time

    def copy(self, **kwargs):
        obj = super().copy(**kwargs)
        obj.samplerate = self.samplerate
        obj.downsampling = self.downsampling
        obj._start_time = self._start_time
        return obj

    @property
    def num_samples(self):
        return self.data.shape[-1]

    @property
    def datarate(self):
        return self.samplerate if self.downsampling is None else self.samplerate / self.downsampling

    @property
    def relative_time(self):
        return np.arange(self.num_samples) / self.datarate

    @property
    def timestamps(self):
        return [self._start_time + positional.datetime.timedelta(seconds=t) for t in self.relative_time]

    @property
    def time_period(self):
        return _core.TimePeriod(start=self._start_time, duration=self.data.shape[-1] / self.datarate)

    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        window = self.time_period.subperiod(time, start=start, stop=stop, center=center, duration=duration)
        start = (window.start - self._start_time).total_seconds()
        stop = (window.stop - self._start_time).total_seconds()
        # Indices assumed to be seconds from start
        start = np.math.floor(start * self.datarate)
        stop = np.math.ceil(stop * self.datarate)
        obj = self.copy()
        obj._data = self.data[..., start:stop]
        obj._start_time = self._start_time + positional.datetime.timedelta(seconds=start / self.datarate)
        return obj

    def reduce(self, function, dim, *args, **kwargs):
        if dim == 'time':
            kwargs.setdefault('_new_class', Data)
        return super().reduce(function=function, dim=dim, *args, **kwargs)


class Pressure(Time):
    @classmethod
    def from_raw_and_calibration(cls, data, calibration, **kwargs):
        """Create a pressure signal from raw values and known calibration.

        Parameters
        ----------
        data : ndarray
            The raw unscaled input data.
            Should be shape `(n_ch, n_samp)` or 1d with just `n_samp`
        calibration : numeric
            The calibration given as dB re. U/μPa,
            where U is the units of the `data`.
            If `data` is in volts, give the calibration in
            dB re. 1V/μPa.
            If `data` is in "file units", e.g. scaled to [-1, 1] fullscale,
            the calibration value must scale from "file units" to μPa.
        """
        data = np.asarray(data)
        calibration = np.asarray(calibration).astype('float32')
        c = 10**(calibration / 20) / 1e-6  # Calibration values are given as dB re. 1μPa
        c = c.reshape((-1,) + (1,) * (data.ndim - 1))
        if data.dtype in (np.int8, np.int16, np.int32, np.float32):
            c = c.astype(np.float32)

        calibrated = data / c
        obj = cls(data=calibrated, **kwargs)
        return obj


class Frequency(Data):
    dims = ('frequency',)

    def __init__(self, data, frequency, bandwidth, **kwargs):
        super().__init__(data=data, **kwargs)
        self.frequency = frequency
        self.bandwidth = bandwidth

    def copy(self, **kwargs):
        obj = super().copy(**kwargs)
        obj.frequency = self.frequency
        obj.bandwidth = self.bandwidth
        return obj

    def reduce(self, function, dim, *args, **kwargs):
        if dim == 'frequency':
            kwargs.setdefault('_new_class', Data)
        return super().reduce(function=function, dim=dim, *args, **kwargs)


class TimeFrequency(Time, Frequency):
    dims = ('frequency', 'time')

    def reduce(self, function, dim, *args, **kwargs):
        if dim == 'time':
            kwargs.setdefault('_new_class', Frequency)
        elif dim == 'frequency':
            kwargs.setdefault('_new_class', Time)
        return super().reduce(function=function, dim=dim, *args, **kwargs)


class DataStack(_core.Branch):
    @property
    def data(self):
        return np.stack([child.data for child in self._children], axis=0)
