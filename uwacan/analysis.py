"""Various analysis protocols and standards for recorded underwater noise from ships."""

import numpy as np
from . import positional, signals, _core
import scipy.signal


def bureau_veritas_source_spectrum(
    recording,
    ship_track,
    run_timestamps,
    transmission_model=None,
    background_noise=None,
    filterbank=None,
    aspect_angles=tuple(range(-45, 46, 5)),
    aspect_window_length=100,
    aspect_window_angle=None,
    aspect_window_duration=None,
    search_duration=300,
    spectrogram_resolution=1,
):
    # Argument handling
    if filterbank is None:
        filterbank = NthOctavebandFilterBank(frequency_range=[10, 50000], bands_per_octave=3, filter_order=8)

    # Helper functions
    def process_run(timestamp):
        search = ship_track.time_subperiod(center=timestamp, duration=search_duration)
        cpa = search.closest_point(recording.metadata['hydrophone position'])
        track = ship_track.time_subperiod(center=cpa.timestamp, duration=search_duration)
        time_segments = track.aspect_windows(
            reference_point=recording.metadata['hydrophone position'],
            angles=aspect_angles,
            window_min_length=aspect_window_length,
            window_min_angle=aspect_window_angle,
            window_min_duration=aspect_window_duration,
        )

        time_start = time_segments[0].start - positional.datetime.timedelta(seconds=spectrogram_resolution)
        time_stop = time_segments[-1].stop + positional.datetime.timedelta(seconds=spectrogram_resolution)
        time_signal = recording.time_subperiod(start=time_start, stop=time_stop).data
        spec = spectrogram(time_signal, window_duration=spectrogram_resolution)
        received_power = filterbank(spec)

        power_segments = {}
        for time_segment in time_segments:
            power_segment = received_power.time_subperiod(time_segment).reduce(np.mean, dim='time')
            power_segment.metadata['segment'] = time_segment.angle
            power_segment.metadata['ship position'] = track.time_subperiod(time_segment).mean
            power_segments[time_segment.angle] = power_segment
        power_segments = signals.DataStack(dim='segment', children=power_segments)
        power_segments.metadata['cpa'] = cpa.distance
        power_segments.metadata['run'] = cpa.timestamp

        compensated_power = background_noise(power_segments)
        source_power = transmission_model(compensated_power)
        return source_power

    return signals.DataStack(dim='run', children={timestamp: process_run(timestamp) for timestamp in run_timestamps})


class NthOctavebandFilterBank(_core.LeafFunction):
    def __init__(self, frequency_range, bands_per_octave=3, filter_order=8):
        self.frequency_range = frequency_range
        self.bands_per_octave = bands_per_octave
        self.filter_order = filter_order

    def function(self, spectrogram, **kwargs):  # TODO: Call signature!
        if isinstance(spectrogram, signals.TimeFrequency):
            # powers = self.power_filters(signal.frequencies).dot(signal.data) / signal.frequencies[1]
            powers = np.matmul(self.power_filters(spectrogram.frequency), spectrogram.data) / spectrogram.bandwidth
            return signals.TimeFrequency(
                data=powers,
                samplerate=spectrogram.samplerate,
                start_time=spectrogram.time_period.start,
                downsampling=spectrogram.downsampling,
                frequency=self.frequency,
                bandwidth=self.bandwidth,
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


@_core.LeafFunction
def spectrogram(time_signal, window_duration=None, window='hann', overlap=0.5, *args, **kwargs):
    # if isinstance(time_signal, signals.DataStack):
    #     return time_signal.apply(
    #         spectrogram,
    #         window_duration=window_duration,
    #         window=window, overlap=overlap, *args, apply_to_data=False, **kwargs)
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
        start_time=time_signal.time_period.start + positional.datetime.timedelta(seconds=t[0]),
        downsampling=window_samples - overlap_samples,
        frequency=f,
        bandwidth=time_signal.samplerate / window_samples,
    )
