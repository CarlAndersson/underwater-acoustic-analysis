import numpy as np
import scipy.signal
from . import positional, _core
import itertools
import abc





# class Container(_Node):
#     def __init__(self, items, layer, **kwargs):
#         super().__init__(**kwargs)
#         self.layer = layer
#         if isinstance(items, (Container, Data)):
#             items = [items]
#         self.items = items
#         for item in self.items:
#             item.container = self

#     def copy(self, new_items=None):
#         if new_items is None:
#             new_items = [item.copy() for item in self.items]
#         new = type(self).__new__(type(self))
#         new.key = self.key
#         new.layer = self.layer
#         new.items = new_items
#         return new

#     def __getitem__(self, key):
#         # if key is a time window, we should make a new container by restricting to the time window at the data layer
#         # if key is one of the keys for one of the items, return it
#         # else, make a new container with asking each item for the key
#         for item in self.items:
#             if item.key == key:
#                 return item
#         return self.apply(lambda item: item[key])

#     def apply(self, func, *args, axis=None, **kwargs):
#         if axis is None:
#             items = [item.apply(func, *args, **kwargs) for item in self.items]
#             return self.copy(items)
#             return type(self)(items, layer=self.layer, key=self.key)

#         if axis != self.layer:
#             items = [item.apply(func, *args, axis=axis, **kwargs) for item in self.items]
#             return self.copy(items)
#             return type(self)(items, layer=self.layer, key=self.key)

#         new = self.items[0].copy()
#         for newdata, *olddata in zip(new._leaves, *[item._leaves for item in self.items]):
#             stacked = [item.data for item in olddata]
#             newdata.data = func(stacked, *args, **kwargs)
#         return new

    # @property
    # def _leaves(self):
    #     yield from itertools.chain(*[item._leaves for item in self.items])




    #     if isinstance(axis, (int, str)):
    #         axis = [axis]
    #     reduce_axes = []
    #     for ax in axis:
    #         if isinstance(ax, str):
    #             ax = self.axes.index(ax)
    #         reduce_axes.append(ax)
    #     reduce_axes = tuple(reduce_axes)
    #     new_axes = tuple(ax for idx, ax in enumerate(self.axes) if idx not in reduce_axes)

    #     obj = self.copy()
    #     obj.data = func(obj.data, *args, axis=reduce_axes, **kwargs)
    #     obj.axes = new_axes
    #     return obj

    # def reduce(self, layer, func):
    #     if layer != self.items[0].layer:
    #         new = self.copy()
    #         new.items = [item.reduce(layer, func) for item in self.items]
    #         return new

    #     new = self.items[0].copy()
    #     # new.items = [item.copy() for item in self.items[0]]
    #     for newdata, *olddata in zip(new.datanodes, *[item.datanodes for item in self.items]):
    #         # print('new:', newdata, 'old:', *olddata)
    #         stacked = np.stack([item.data for item in olddata], axis=-1)
    #         stacked = [item.data for item in olddata]
    #         newdata.data = func(stacked)
    #    return new

class Data(_core.Leaf):
    axes = tuple()

    def __init__(self, data, axes=None, **kwargs):
        super().__init__(**kwargs)
        self._data = np.asarray(data)

        if axes is not None:
            self.axes = axes
        if len(self.axes) != self.data.ndim:
            raise ValueError('The number of dimensions in the data does not match the number of expected axes')
            # if len(self.axes) < self.data.ndim:
            #     self.axes = self.axes[-self.data.ndim]
            # else:
            #     raise ValueError('The number of dimensions in the data does not match the number of expected axes')
        # if self.data.ndim == len(self.axes) + 1:
            # Override the class default with a `channels` axis?
            # self.axes = ('channels', ) + self.axes

    def copy(self, deep=False, **kwargs):
        obj = super().copy(**kwargs)
        if deep:
            obj._data = self.data.copy()
        else:
            obj._data = self.data
        obj.axes = self.axes
        return obj




    # def sum(self, axis=None):
        # return self.apply(np.sum, axis=axis)

    # def mean(self, axis=None):
        # return self.apply(np.nanmean, axis=axis)

    # @classmethod
    # def stack(cls, items, axis):
    #     obj = items[0].copy(_new_class=cls)
    #     obj.data = np.stack([item.data for item in items], axis=0)
    #     obj.axes = (axis,) + obj.axes
    #     return obj

    # @property
    # def _leaves(self):
    #     yield self

    def reduce(self, function, axis, *args, _new_class=None, **kwargs):
        if isinstance(axis, (int, str)):
            axis = [axis]
        reduce_axes = []
        for ax in axis:
            if isinstance(ax, str):
                ax = self.axes.index(ax)
            reduce_axes.append(ax)
        reduce_axes = tuple(reduce_axes)
        new_axes = tuple(ax for idx, ax in enumerate(self.axes) if idx not in reduce_axes)

        obj = self.copy(_new_class=_new_class)
        obj._data = function(obj.data, axis=reduce_axes, *args, **kwargs)
        obj.axes = new_axes
        return obj


class Time(Data):
    axes = ('time',)

    def __init__(self, data, samplerate, start_time, downsampling=None, **kwargs):
        super().__init__(data=data, **kwargs)
        # self.data = np.asarray(data)
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
    def time_window(self):
        return positional.TimeWindow(
            start=self._start_time,
            duration=self.data.shape[-1] / self.datarate
        )

    def __getitem__(self, key):
        if isinstance(key, positional.TimeWindow):
            key = slice(key.start, key.stop)

        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if isinstance(start, positional.datetime.datetime):
                start = (start - self._start_time).total_seconds()
            if isinstance(stop, positional.datetime.datetime):
                stop = (stop - self._start_time).total_seconds()

            # Indices assumed to be seconds from start
            start = np.math.floor(start * self.datarate)
            stop = np.math.ceil(stop * self.datarate)

            obj = self.copy()
            obj._data = self.data[..., start:stop]
            obj._start_time = self._start_time + positional.datetime.timedelta(seconds=start / self.datarate)
            return obj

        raise IndexError('only TimeWindows or slices of integers/datetimes are valid indices to Signal containers')

    def reduce(self, function, axis, *args, **kwargs):
        if axis == 'time':
            kwargs.setdefault('_new_class', Data)
        return super().reduce(function=function, axis=axis, *args, **kwargs)


    # def spectrogram(self, window_duration=None, overlap=None, **kwargs):
    #     if window_duration is not None:
    #         window_samples = round(window_duration * self.samplerate)
    #         if 'nperseg' in kwargs and kwargs['nperseg'] != window_samples:
    #             raise ValueError("Conflicting values for window size between 'nperseg' and 'window_duration'")
    #         kwargs['nperseg'] = window_samples
    #         if 'window' in kwargs and kwargs['window'].size != window_samples:
    #             raise ValueError("Specified length of window differs to length of specified window")

    #     if overlap is not None:
    #         if 'window' in kwargs:
    #             kwargs['nperseg'] = kwargs['window'].size
    #         if 'nperseg' not in kwargs:
    #             kwargs['nperseg'] = 256  # scipy default. we need this to compute overlap
    #         overlap_samples = round(kwargs['nperseg'] * overlap)
    #         if 'noverlap' in kwargs and kwargs['noverlap'] != overlap_samples:
    #             raise ValueError("Conflicting values for overlap size between 'noverlap' and 'overlap'")
    #         kwargs['noverlap'] = overlap_samples

    #     f, t, Sxx = scipy.signal.spectrogram(
    #         x=self.data,
    #         fs=self.samplerate,
    #         **kwargs
    #     )
    #     obj = self.copy(_new_class=TimeFrequency)
    #     obj.data = Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
    #     # The array returned from the spectrogram function is the real part of the original stft, reshaped.
    #     # This means that the array takes twice the memory (the imaginary part is still around),
    #     # and it's not contiguous which slows down filtering a lot.
    #     obj.start_time = self.time_window.start + positional.datetime.timedelta(seconds=t[0])
    #     obj.downsampling = window_samples - overlap_samples
    #     obj.frequency = f
    #     obj.bandwidth = self.samplerate / window_samples,
    #     return obj

        # return TimeFrequency(
        #     data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
        #     # The array returned from the spectrogram function is the real part of the original stft, reshaped.
        #     # This means that the array takes twice the memory (the imaginary part is still around),
        #     # and it's not contiguous which slows down filtering a lot.
        #     samplerate=time_signal.samplerate,
        #     start_time=time_signal.time_window.start + positional.datetime.timedelta(seconds=t[0]),
        #     downsampling=window_samples - overlap_samples,
        #     frequency=f,
        #     bandwidth=time_signal.samplerate / window_samples,
        #     _name=self._name
        # )
        # super().__init__(
        #     data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
        #     # The array returned from the spectrogram function is the real part of the original stft, reshaped.
        #     # This means that the array takes twice the memory (the imaginary part is still around),
        #     # and it's not contiguous which slows down filtering a lot.
        #     samplerate=time_signal.samplerate,
        #     start_time=time_signal.time_window.start + positional.datetime.timedelta(seconds=t[0]),
        #     downsampling=window_samples - overlap_samples,
        #     frequency=f,
        #     bandwidth=time_signal.samplerate / window_samples,
        # )


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
        # data = np.atleast_2d(data)
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
    axes = ('frequency',)

    def __init__(self, data, frequency, bandwidth, **kwargs):
        super().__init__(data=data, **kwargs)
        self.frequency = frequency
        self.bandwidth = bandwidth

    def copy(self, **kwargs):
        obj = super().copy(**kwargs)
        obj.frequency = self.frequency
        obj.bandwidth = self.bandwidth
        return obj

    def reduce(self, function, axis=None, *args, **kwargs):
        if axis == 'frequency':
            kwargs.setdefault('_new_class', Data)
        return super().reduce(function=function, axis=axis, *args, **kwargs)


# class Spectrogram(Time, Frequency):
#     axes = ('frequency', 'time')
#     def __init__(self, time_signal, window_duration=None, window='hann', overlap=0.5, *args, **kwargs):
#         window_samples = round(window_duration * time_signal.samplerate)
#         overlap_samples = round(window_duration * overlap * time_signal.samplerate)
#         f, t, Sxx = scipy.signal.spectrogram(
#             x=time_signal.data,
#             fs=time_signal.samplerate,
#             window=window,
#             nperseg=window_samples,
#             noverlap=overlap_samples,
#         )
#         super().__init__(
#             data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
#             # The array returned from the spectrogram function is the real part of the original stft, reshaped.
#             # This means that the array takes twice the memory (the imaginary part is still around),
#             # and it's not contiguous which slows down filtering a lot.
#             samplerate=time_signal.samplerate,
#             start_time=time_signal.time_window.start + positional.datetime.timedelta(seconds=t[0]),
#             downsampling=window_samples - overlap_samples,
#             frequency=f,
#             bandwidth=time_signal.samplerate / window_samples,
#         )

class TimeFrequency(Time, Frequency):
    axes = ('frequency', 'time')

    def reduce(self, function, axis=None, *args, **kwargs):
        if axis == 'time':
            kwargs.setdefault('_new_class', Frequency)
        elif axis == 'frequency':
            kwargs.setdefault('_new_class', Time)
        return super().reduce(function=function, axis=axis, *args, **kwargs)


class DataStack(_core.Branch):
    @property
    def data(self):
        return np.stack([child.data for child in self._children], axis=0)


# class FrequencyDataArray(DataArray):
#     ...


# class TimeDataArray(DataArray):
#     ...


# class TimeFrequencyDataArray(DataArray):
#     ...


# class PowerBands(Frequency):
    # pass


# class SourceSpectrum(Frequency):
    # axes = ('runs', 'segments', 'channels', 'frequency')


# class PowerBandSignal(Time, Frequency):
#     axes = ('channels', 'frequency', 'time')

#     def mean(self, *args, **kwargs):
#         obj = super().mean(*args, **kwargs)
#         if 'time' not in obj.axes:
#             obj = obj.copy(_new_class=PowerBands)
#         return obj
