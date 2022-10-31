"""Various analysis protocols and standards for recorded underwater noise from ships."""

import numpy as np
from . import positional, signals
import scipy.signal


class BureauVeritasSourceSpectrum:
    def __init__(
        self,
        recording,
        track,
        passby_timestamps,
        transmission_model,
        background_noise,
        filterbanks=None,
        aspect_angles=tuple(range(-45, 46, 5)),
        aspect_window_length=100,
        aspect_window_angle=None,
        passby_search_duration=600,
    ):
        # TODO: naming of things. It makes sense to use "segment" for the aspect windows, and BV uses "runs" for each of the passbys.
        self.recording = recording
        self.track = track
        self.transmission_model = transmission_model
        self.background_noise = background_noise

        self.filterbanks = filterbanks or {'thirds': NthOctavebandFilterBank(frequency_range=[10, 50000], bands_per_octave=3, filter_order=8)}
        self.aspect_angles = aspect_angles
        self.aspect_window_length = aspect_window_length
        self.aspect_window_angle = aspect_window_angle
        self.passby_search_duration = passby_search_duration

        if passby_timestamps:
            passby_spectra = [self.process_passby(time) for time in passby_timestamps]
            self.source_spectra = {band: signals.SourceSpectrum.stack([pwr[band] for pwr in passby_spectra], axis='runs') for band in self.filterbanks}

    def process_passby(self, time):
        if not isinstance(time, positional.TimeWindow):
            time = positional.TimeWindow(center=time, duration=self.passby_search_duration)
        recording = self.recording[time]
        track = self.track[time]
        cpa = track.closest_point(recording.position)

        # TODO: some kind of check that the coarse selection of time window is good enough to cover the needed range for the aspect windows.
        # If we fail this check, recursively call the processing function with an extended coarse window.
        # Get aspect angles between ship position and hydrophone position
        time_windows = track.aspect_windows(
            reference_point=recording.position,
            angles=self.aspect_angles,
            window_min_length=self.aspect_window_length,
            window_min_angle=self.aspect_window_angle,
        )

        # TODO: make the spectrogram time window configurable. Make sure to modify the start time and stop time to match!
        time_start = time_windows[0].start - positional.datetime.timedelta(seconds=1)
        time_stop = time_windows[-1].stop + positional.datetime.timedelta(seconds=1)
        time_signal = recording[time_start:time_stop].data
        # spectrogram = signals.Spectrogram(time_signal, window_duration=1)
        spec = spectrogram(time_signal, window_duration=1)

        source_powers = {}
        for band, filterbank in self.filterbanks.items():
            received_power = filterbank(time_signal=time_signal, spectrogram=spec)
            source_power = signals.SourceSpectrum(
                data=np.zeros((len(self.aspect_angles), len(filterbank.frequency))),
                frequency=filterbank.frequency,
                bandwidth=filterbank.bandwidth,
                axes=('segments', 'frequency'),
            )
            for idx, window in enumerate(time_windows):
                window_power = received_power[window].mean(axis='time')
                window_power = self.background_noise.compensate(window_power)
                window_power = self.transmission_model.compensate(
                    window_power,
                    receiver=recording,
                    source_track=track[window].mean,  # The Bureau Veritas method evaluates the TL at window center only.
                    time=window.center
                )
                source_power.data[idx] = window_power.mean(axis='channels').data
            source_powers[band] = source_power
        return source_powers


class NthOctavebandFilterBank:
    def __init__(self, frequency_range, bands_per_octave=3, filter_order=8):
        self.frequency_range = frequency_range
        self.bands_per_octave = bands_per_octave
        self.filter_order = filter_order

    def __call__(self, spectrogram, **kwargs):  # TODO: Call signature!
        if isinstance(spectrogram, signals.DataStack):
            return spectrogram.apply(self, apply_to_data=False)
        if isinstance(spectrogram, signals.TimeFrequency):
            # powers = self.power_filters(signal.frequencies).dot(signal.data) / signal.frequencies[1]
            powers = np.matmul(self.power_filters(spectrogram.frequency), spectrogram.data) / spectrogram.bandwidth
            return signals.TimeFrequency(
                data=powers,
                samplerate=spectrogram.samplerate,
                start_time=spectrogram.time_window.start,
                downsampling=spectrogram.downsampling,
                frequency=self.frequency,
                bandwidth=self.bandwidth,
                _name=spectrogram._name,
                _metadata=spectrogram._metadata,
            )
        else:
            raise TypeError(f'Cannot filter data of input type {type(spectrogram)}')

    @property
    def frequency(self):
        lowest_band, highest_band = self.frequency_range
        lowest_band_index = np.round(self.bands_per_octave * np.log2(lowest_band / 1e3))
        highest_band_index = np.round(self.bands_per_octave * np.log2(highest_band / 1e3))
        octaves = np.arange(lowest_band_index, highest_band_index + 1) / self.bands_per_octave
        return 1e3 * 2 ** octaves

    @property
    def bandwidth(self):
        return self.frequency * (2**(0.5 / self.bands_per_octave) - 2**(-0.5 / self.bands_per_octave))

    def power_filters(self, frequencies):
        centers = self.frequency[:, None]
        bandwidths = self.bandwidth[:, None]
        with np.errstate(divide='ignore'):
            filters = 1 / (
                1
                + ((frequencies**2 - centers**2) / (frequencies * bandwidths))
                ** (2 * self.filter_order)
            )
        return filters


def spectrogram(time_signal, window_duration=None, window='hann', overlap=0.5, *args, **kwargs):
    if isinstance(time_signal, signals.DataStack):
        return time_signal.apply(
            spectrogram,
            window_duration=window_duration,
            window=window, overlap=overlap, *args, apply_to_data=False, **kwargs)
    if not isinstance(time_signal, signals.Time):
        raise TypeError(f"Cannot calculate the spectrogram of object of type '{time_signal.__class__.__name__}'")
    window_samples = round(window_duration * time_signal.samplerate)
    overlap_samples = round(window_duration * overlap * time_signal.samplerate)
    f, t, Sxx = scipy.signal.spectrogram(
        x=time_signal.data,
        fs=time_signal.samplerate,
        window=window,
        nperseg=window_samples,
        noverlap=overlap_samples,
    )
    return signals.TimeFrequency(
        data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
        # The array returned from the spectrogram function is the real part of the original stft, reshaped.
        # This means that the array takes twice the memory (the imaginary part is still around),
        # and it's not contiguous which slows down filtering a lot.
        samplerate=time_signal.samplerate,
        start_time=time_signal.time_window.start + positional.datetime.timedelta(seconds=t[0]),
        downsampling=window_samples - overlap_samples,
        frequency=f,
        bandwidth=time_signal.samplerate / window_samples,
        _name=time_signal._name,
        _metadata=time_signal._metadata,
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
