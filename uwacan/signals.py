import numpy as np
import scipy.signal
from . import positional, _core
import itertools
import abc



class DataTree(_core.Branch):
    def __new__(cls, *args, children=None, **kwargs):
        if children is not None:
            first = next(iter(children.values()))
            obj = super().__new__(first._leaf_type._tree_class)
        else:
            obj = super().__new__(cls)
        if not isinstance(obj, cls):
            obj.__init__(*args, children=children, **kwargs)
        return obj

    @property
    def data(self):
        return np.stack([child.data for child in self.values()], axis=0)


class TimeDataTree(_core.TimeBranchin, DataTree):
    ...


class FrequencyDataTree(DataTree):
    @property
    def frequency(self):
        leaves = self._traverse(leaves=True, branches=False, root=False)
        frequency = next(leaves).frequency
        for leaf in leaves:
            leaf_frequency = leaf.frequency
            if not (frequency is leaf_frequency or (
                np.shape(frequency) == np.shape(leaf_frequency)
                and np.allclose(frequency, leaf_frequency)
            )):
                raise ValueError('Cannot access frequency of DataTree where the frequencies of the leaves is not consistent.')
        return frequency


class TimeFrequencyDataTree(TimeDataTree, FrequencyDataTree):
    ...


class Data(_core.Leaf):
    dims = tuple()
    _tree_class = DataTree

    def __new__(cls, data, dims=None, **kwargs):
        if dims is not None:
            if 'time' in dims and 'frequency' in dims:
                obj = super().__new__(TimeFrequency)
            elif 'time' in dims:
                obj = super().__new__(Time)
            elif 'frequency' in dims:
                obj = super().__new__(Frequency)
            else:
                obj = super().__new__(Data)
        else:
            obj = super().__new__(cls)
        if not isinstance(obj, cls):
            obj.__init__(data, dims=dims, **kwargs)
        return obj

    def __init__(self, data, dims=None, **kwargs):
        super().__init__(**kwargs)
        self._data = np.asarray(data)

        if dims is not None:
            self.dims = dims
        if len(self.dims) != self.data.ndim:
            raise ValueError('The number of dimensions in the data does not match the number of expected axes')

    def clone(self, data, dims=None):
        return type(self)(data, dims=dims or self.dims, metadata=self.metadata)

    def reduce(self, function, dim, *args, **kwargs):
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
        if not isinstance(out, Data):
            out = self.clone(out, dims=new_dims)
        return out


class Time(_core.TimeLeafin, Data):
    dims = ('time',)
    _tree_class = TimeDataTree

    def __init__(self, data, samplerate, start_time, **kwargs):
        super().__init__(data=data, **kwargs)
        self.samplerate = samplerate
        # TODO: parse the start time here as well, if it's a string.
        self._start_time = start_time

    def clone(self, data, dims=None):
        dims = self.dims if dims is None else dims
        kwargs = {'dims': dims, 'metadata': self.metadata}
        if 'time' in dims:
            kwargs.update(samplerate=self.samplerate, start_time=self._start_time)
        return type(self)(data, **kwargs)

    @property
    def num_samples(self):
        return self.data.shape[-1]

    @property
    def relative_time(self):
        return np.arange(self.num_samples) / self.datarate

    @property
    def timestamps(self):
        return [self._start_time + positional.datetime.timedelta(seconds=t) for t in self.relative_time]

    @property
    def time_period(self):
        return _core.TimePeriod(start=self._start_time, duration=self.data.shape[-1] / self.samplerate)

    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        window = self.time_period.subperiod(time, start=start, stop=stop, center=center, duration=duration)
        start = (window.start - self._start_time).total_seconds()
        stop = (window.stop - self._start_time).total_seconds()
        # Indices assumed to be seconds from start
        start = np.math.floor(start * self.datarate)
        stop = np.math.ceil(stop * self.datarate)
        return type(self)(
            data=self.data[..., start:stop],
            start_time=self._start_time + positional.datetime.timedelta(seconds=start / self.samplerate),
            samplerate=self.samplerate,
            metadata=self.metadata,
        )


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
    _tree_class = FrequencyDataTree

    def __init__(self, data, frequency, bandwidth, **kwargs):
        super().__init__(data=data, **kwargs)
        self.frequency = frequency
        self.bandwidth = bandwidth

    def clone(self, data, dims=None):
        dims = dims or self.dims
        kwargs = {'dims': dims, 'metadata': self.metadata}
        if 'frequency' in dims:
            kwargs.update(frequency=self.frequency, bandwidth=self.bandwidth)
        return type(self)(data, **kwargs)


class TimeFrequency(Time, Frequency):
    dims = ('frequency', 'time')
    _tree_class = TimeFrequencyDataTree

    def clone(self, data, dims=None):
        dims = self.dims if dims is None else dims
        kwargs = {'dims': dims, 'metadata': self.metadata}
        if 'time' in dims:
            kwargs.update(samplerate=self.samplerate, start_time=self._start_time)
        if 'frequency' in dims:
            kwargs.update(frequency=self.frequency, bandwidth=self.bandwidth)
        return type(self)(data, **kwargs)


class DataStack(_core.Branch):
    @property
    def data(self):
        return np.stack([child.data for child in self._children], axis=0)
