"""Various analysis protocols and standards for recorded underwater noise from ships."""

import numpy as np
from . import recordings
import scipy.signal
import xarray as xr


class Passage:
    def __init__(self, hydrophone, ship_track):
        start = max(hydrophone.sampling.window.start, ship_track.time_period.start)
        stop = min(hydrophone.sampling.window.stop, ship_track.time_period.stop)

        self.hydrophone = hydrophone.sampling.subwindow(start=start, stop=stop)
        self.ship_track = ship_track.time_subperiod(start=start, stop=stop)


def bureau_veritas_source_spectrum(
    passages,
    transmission_model=None,
    background_noise=None,
    filterbank=None,
    aspect_angles=tuple(range(-45, 46, 5)),
    aspect_window_length=100,
    aspect_window_angle=None,
    aspect_window_duration=None,
    spectrogram_resolution=1,
):
    if filterbank is None:
        filterbank = DecidecadeFilterbank(frequency_range=[10, 50000])
    if transmission_model is None:
        from .transmission_loss import MlogR
        transmission_model = MlogR(m=20)
    if background_noise is None:
        def background_noise(received_power, **kwargs):
            return received_power

    n_passages = len(passages)
    n_segments = len(aspect_angles)
    n_frequencies = len(filterbank.frequency)
    n_channels = passages[0].hydrophone.num_channels

    passage_powers = []
    for passage_idx, passage in enumerate(passages):
        cpa = passage.ship_track.closest_point(passage.hydrophone.position)
        time_segments = passage.ship_track.aspect_windows(
            reference_point=passage.hydrophone.position,
            angles=aspect_angles,
            window_min_length=aspect_window_length,
            window_min_angle=aspect_window_angle,
            window_min_duration=aspect_window_duration,
        )
        time_start = time_segments[0].start.subtract(seconds=spectrogram_resolution)
        time_stop = time_segments[-1].stop.add(seconds=spectrogram_resolution)
        time_data = passage.hydrophone.sampling.subwindow(start=time_start, stop=time_stop).time_data
        spec = spectrogram(time_data, spectrogram_resolution)
        received_power = filterbank(spec)

        segment_powers = []
        for segment_idx, segment in enumerate(time_segments):
            received_segment = received_power.sampling.subwindow(segment).reduce(np.mean, dim='time')

            compensated_segment = background_noise(
                received_segment,
                receiver=passage.hydrophone,
                time=segment.center,
            )

            source_segment = transmission_model(
                compensated_segment,
                receiver=passage.hydrophone,
                source=segment,
                time=segment.center,
            )
            source_segment = source_segment.assign_coords(segment=segment.angle, latitude=segment.position.latitude, longitude=segment.position.longitude)
            segment_powers.append(source_segment)

        segment_powers = xr.concat(segment_powers, dim='segment')
        passage_powers.append(segment_powers.assign_coords(cpa=cpa.distance))
    source_powers = xr.concat(passage_powers, dim='passage')
    return source_powers


def spectrogram(time_signal, window_duration=None, window='hann', overlap=0.5, *args, **kwargs):
    fs = time_signal.sampling.rate
    window_samples = round(window_duration * fs)
    overlap_samples = round(window_duration * overlap * fs)
    f, t, Sxx = scipy.signal.spectrogram(
        x=time_signal.data,
        fs=fs,
        window=window,
        nperseg=window_samples,
        noverlap=overlap_samples,
    )
    dims = list(time_signal.dims)
    dims.insert(dims.index('time'), 'frequency')
    return recordings.time_frequency_data(
        data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
        # The array returned from the spectrogram function is the real part of the original stft, reshaped.
        # This means that the array takes twice the memory (the imaginary part is still around),
        # and it's not contiguous which slows down filtering a lot.
        samplerate=fs / (window_samples - overlap_samples),
        start_time=time_signal.sampling.window.start.add(seconds=t[0]),
        frequency=f,
        bandwidth=fs / window_samples,
        dims=tuple(dims),
    )


class NthDecadeFilterbank:
    def __init__(self, frequency_range, bands_per_decade, filter_order=8, scaling='density'):
        self.frequency_range = frequency_range
        self.bands_per_decade = bands_per_decade
        self.filter_order = filter_order
        self.scaling = scaling

    def __call__(self, spectrogram):
        spectrogram = spectrogram.rename({'frequency': 'input_frequency', 'bandwidth': 'input_bandwidth'})
        filters = self.power_filters(spectrogram.input_frequency)
        with xr.set_options(keep_attrs=True):
            filtered = filters.dot(spectrogram * spectrogram.input_bandwidth)
        return filtered.rename({'output_frequency': 'frequency', 'output_bandwidth': 'bandwidth'})

    @property
    def frequency(self):
        lowest_band, highest_band = self.frequency_range
        lowest_band_index = np.round(self.bands_per_decade * np.log10(lowest_band / 1e3))
        highest_band_index = np.round(self.bands_per_decade * np.log10(highest_band / 1e3))
        decades = np.arange(lowest_band_index, highest_band_index + 1) / self.bands_per_decade
        return 1e3 * 10 ** decades

    @property
    def bandwidth(self):
        return self.frequency * (10**(0.5 / self.bands_per_decade) - 10**(-0.5 / self.bands_per_decade))

    def power_filters(self, frequencies):
        try:
            cashed = self._filter_cashe
        except AttributeError:
            pass
        else:
            if cashed.input_frequency.equals(frequencies):
                return cashed

        centers = xr.DataArray(self.frequency, dims='output_frequency')
        bandwidths = xr.DataArray(self.bandwidth, dims='output_frequency')
        filters = 1 / (
            1
            + ((frequencies**2 - centers**2) / (frequencies * bandwidths))
            ** (2 * self.filter_order)
        )
        filters.coords['output_frequency'] = centers
        filters.coords['output_bandwidth'] = bandwidths
        if self.scaling == 'density':
            filters = filters / bandwidths
        self._filter_cashe = filters
        return filters


class DecidecadeFilterbank(NthDecadeFilterbank):
    def __init__(self, frequency_range, filter_order=8, scaling='density'):
        super().__init__(frequency_range=frequency_range, bands_per_decade=10, filter_order=filter_order, scaling=scaling)
