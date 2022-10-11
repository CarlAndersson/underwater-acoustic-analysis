import numpy as np
import scipy.signal
from . import positional


class Signal:
    axes = tuple()
    def __init__(self, data, axes=None):
        self.data = np.asarray(data)
        if axes is not None:
            self.axes = axes
        if len(self.axes) != self.data.ndim:
            if len(self.axes) < self.data.ndim:
                self.axes = self.axes[-self.data.ndim]
            else:
                raise ValueError('The number of dimensions in the data does not match the number of expected axes')
        # if self.data.ndim == len(self.axes) + 1:
            # Override the class default with a `channels` axis?
            # self.axes = ('channels', ) + self.axes

    def copy(self, deep=False, _new_class=None):
        if _new_class is None:
            _new_class = type(self)
        obj = _new_class.__new__(_new_class)
        if deep:
            obj.data = self.data.copy()
        else:
            obj.data = self.data
        obj.axes = self.axes
        return obj

    def apply(self, func, *args, axis=None, **kwargs):
        if axis is None:
            obj = self.copy()
            obj.data = func(obj.data, *args, **kwargs)
            obj.axes = self.axes[len(self.axes) - np.ndim(obj.data):]
            return obj

        if isinstance(axis, (int, str)):
            axis = [axis]
        reduce_axes = []
        for ax in axis:
            if isinstance(ax, str):
                ax = self.axes.index(ax)
            reduce_axes.append(ax)
        reduce_axes = tuple(reduce_axes)
        new_axes = tuple(ax for idx, ax in enumerate(self.axes) if idx not in reduce_axes)

        obj = self.copy()
        obj.data = func(obj.data, *args, axis=reduce_axes, **kwargs)
        obj.axes = new_axes
        return obj

    def sum(self, axis=None):
        return self.apply(np.sum, axis=axis)

    def mean(self, axis=None):
        return self.apply(np.nanmean, axis=axis)

    @classmethod
    def stack(cls, items, axis):
        obj = items[0].copy(_new_class=cls)
        obj.data = np.stack([item.data for item in items], axis=0)
        obj.axes = (axis,) + obj.axes
        return obj


class Time(Signal):
    axes = ('channels', 'time')

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
            obj.data = self.data[..., start:stop]
            obj._start_time = self._start_time + positional.datetime.timedelta(seconds=start / self.datarate)
            return obj

        raise IndexError('only TimeWindows or slices of integers/datetimes are valid indices to Signal containers')


class Pressure(Time):
    @classmethod
    def from_raw_and_calibration(cls, raw_signal, calibration, **kwargs):
        """Create a pressure signal from raw values and known calibration.

        Parameters
        ----------
        raw_signal : ndarray
            The raw unscaled input data.
            Should be shape `(n_ch, n_samp)` or 1d with just `n_samp`
        calibration : numeric
            The calibration given as dB re. U/μPa,
            where U is the units of the `raw_signal`.
            If `raw_signal` is in volts, give the calibration in
            dB re. 1V/μPa.
            If `raw_signal` is in "file units", e.g. scaled to [-1, 1] fullscale,
            the calibration value must scale from "file units" to μPa.
        """
        # raw_signal = np.atleast_2d(raw_signal)
        raw_signal = np.asarray(raw_signal)
        calibration = np.asarray(calibration).astype('float32')
        c = 10**(calibration / 20) / 1e-6  # Calibration values are given as dB re. 1μPa
        c = c.reshape((-1,) + (1,) * (raw_signal.ndim - 1))
        if raw_signal.dtype in (np.int8, np.int16, np.int32, np.float32):
            c = c.astype(np.float32)

        calibrated = raw_signal / c
        obj = cls(data=calibrated, **kwargs)
        return obj


class Frequency(Signal):
    axes = ('channels', 'frequency')

    def __init__(self, data, frequency, bandwidth, **kwargs):
        super().__init__(data=data, **kwargs)
        self.frequency = frequency
        self.bandwidth = bandwidth

    def copy(self, **kwargs):
        obj = super().copy(**kwargs)
        obj.frequency = self.frequency
        obj.bandwidth = self.bandwidth
        return obj


class Spectrogram(Time, Frequency):
    axes = ('channels', 'frequency', 'time')
    def __init__(self, time_signal, window_duration=None, window='hann', overlap=0.5):
        window_samples = round(window_duration * time_signal.samplerate)
        overlap_samples = round(window_duration * overlap * time_signal.samplerate)
        f, t, Sxx = scipy.signal.spectrogram(
            x=time_signal.data,
            fs=time_signal.samplerate,
            window=window,
            nperseg=window_samples,
            noverlap=overlap_samples,
        )
        super().__init__(
            data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
            # The array returned from the spectrogram function is the real part of the original stft, reshaped.
            # This means that the array takes twice the memory (the imaginary part is still around),
            # and it's not contiguous which slows down filtering a lot.
            samplerate=time_signal.samplerate,
            start_time=time_signal.time_window.start + positional.datetime.timedelta(seconds=t[0]),
            downsampling=window_samples - overlap_samples,
            frequency=f,
            bandwidth=time_signal.samplerate / window_samples,
        )


class PowerBands(Frequency):
    pass


class SourceSpectrum(Frequency):
    axes = ('runs', 'segments', 'channels', 'frequency')


class PowerBandSignal(Time, Frequency):
    axes = ('channels', 'frequency', 'time')

    def mean(self, *args, **kwargs):
        obj = super().mean(*args, **kwargs)
        if 'time' not in obj.axes:
            obj = obj.copy(_new_class=PowerBands)
        return obj
