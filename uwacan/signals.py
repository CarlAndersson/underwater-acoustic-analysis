import numpy as np
import scipy.signal
from . import positional


class Signal:
    def __init__(self, data, samplerate, start_time, downsampling=None):
        self.data = np.asarray(data)
        self.samplerate = samplerate
        self.downsampling = downsampling
        self._start_time = start_time


    def copy(self):
        obj = type(self).__new__(type(self))
        obj.samplerate = self.samplerate
        obj.downsampling = self.downsampling
        obj.time_window = self.time_window
        obj.data = self.data
        return obj

    @property
    def num_samples(self):
        return self.data.shape[-1]

    @property
    def datarate(self):
        return self.samplerate if self.downsampling is None else self.samplerate / self.downsampling

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
                start = (start - self._start).total_seconds()
                start = np.math.ceil(start * self.datarate)
            if isinstance(stop, positional.datetime.datetime):
                stop = (stop - self._start).total_seconds()
                stop = np.math.floor(stop * self.datarate)

            obj = self.copy()
            obj.data = self.data[..., start:stop]
            obj._start_time = self._start_time + positional.datetime.timedelta(seconds=start / self.datarate)
            return obj

        raise IndexError('only TimeWindows or slices of integers/datetimes are valid indices to Signal containers')


class Pressure(Signal):
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
        raw_signal = np.asarray(raw_signal)
        calibration = np.asarray(calibration)
        c = 10**(calibration / 20) / 1e-6  # Calibration values are given as dB re.
        calibrated = raw_signal / c.reshape((-1,) + (1,) * (raw_signal.ndim - 1))
        obj = cls(data=calibrated, **kwargs)
        return obj


class Spectrogram(Signal):
    # TODO: Shapes! This gets data.shape (n_ch, n_freq, n_time)
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
            data=Sxx,
            samplerate=time_signal.samplerate,
            start_time=time_signal.time_window.start,
            downsampling=window_samples - overlap_samples,
        )
        self.frequencies = f
        self.time = t

    def copy(self):
        obj = super().copy()
        obj.frequencies = self.frequencies
        obj.time = self.time


class NthOctavebandFilterBank:
    # TODO: Shapes! This outputs signals with data.shape (n_ch, n_bands, n_time)
    # TODO: Make a simple class to hold "power signals".
    # It's probably a good idea to have some way to detect that we're working with bandpassed signals, and some way to detect that we're working with power signals?
    def __init__(self, frequency_range, bands_per_octave=3, filter_order=8):
        self.frequency_range = frequency_range
        self.bands_per_octave = bands_per_octave
        self.filter_order = filter_order

    def __call__(self, signal):  # TODO: Call signature!
        if isinstance(signal, Spectrogram):
            powers = self.power_filters(signal.frequencies).dot(signal.data) / signal.frequencies[1]
            return Signal(
                data=powers,
                samplerate=signal.samplerate,
                start_time=signal.time_window.start,
                downsampling=signal.downsampling,
            )
        else:
            raise TypeError(f'Cannot filter data of input type {type(signal)}')

    @property
    def center_frequencies(self):
        lowest_band, highest_band = self.frequency_range
        lowest_band_index = np.round(self.bands_per_octave * np.log2(lowest_band / 1e3))
        highest_band_index = np.round(self.bands_per_octave * np.log2(highest_band / 1e3))
        octaves = np.arange(lowest_band_index, highest_band_index + 1) / self.bands_per_octave
        return 1e3 * 2 ** octaves

    def power_filters(self, frequencies):
        centers = self.center_frequencies[:, None]
        bandwidths = centers * (2**(0.5 / self.bands_per_octave) - 2**(-0.5 / self.bands_per_octave))
        filters = 1 / (
            1
            + ((frequencies**2 - centers**2) / (frequencies * bandwidths))
            ** (2 * self.filter_order)
        )
        return filters
