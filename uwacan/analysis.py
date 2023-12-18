"""Various analysis protocols and standards for recorded underwater noise from ships."""

import numpy as np
from . import recordings, positional, _tools
import scipy.signal
import xarray as xr


def dB(x, power=True, safe_zeros=True, ref=1):
    '''Calculate the decibel of an input value

    Parameters
    ----------
    x : numeric
        The value to take the decibel of
    power : boolean, default True
        Specifies if the input is a power-scale quantity or a root-power quantity.
        For power-scale quantities, the output is 10 log(x), for root-power quantities the output is 20 log(x).
        If there are negative values in a power-scale input, the handling can be controlled as follows:
        - `power='imag'`: return imaginary values
        - `power='nan'`: return nan where power < 0
        - `power=True`: as `nan`, but raises a warning.
    safe_zeros : boolean, default True
        If this option is on, all zero values in the input will be replaced with the smallest non-zero value.
    ref : numeric
        The reference unit for the decibel. Note that this should be in the same unit as the `x` input,
        e.g., if `x` is a power, the `ref` value might need squaring.
    '''
    if safe_zeros and np.size(x) > 1:
        nonzero = x != 0
        min_value = np.nanmin(abs(xr.where(nonzero, x, np.nan)))
        x = xr.where(nonzero, x, min_value)
    if power:
        if np.any(x < 0):
            if power == 'imag':
                return 10 * np.log10(x + 0j)
            if power == 'nan':
                return 10 * np.log10(xr.where(x > 0, x, np.nan))
        return 10 * np.log10(x / ref)
    else:
        return 20 * np.log10(np.abs(x) / ref)


class Passage:
    def __init__(self, recording, track):
        start = max(recording.sampling.window.start, track.sampling.window.start)
        stop = min(recording.sampling.window.stop, track.sampling.window.stop)

        self.recording = recording.sampling.subwindow(start=start, stop=stop)
        self.track = track.sampling.subwindow(start=start, stop=stop)


def bureau_veritas_source_spectrum(
    passages,
    transmission_model=None,
    background_noise=None,
    filterbank=None,
    aspect_angles=tuple(range(-45, 46, 5)),
    aspect_segment_length=100,
    aspect_segment_angle=None,
    aspect_segment_duration=None,
    passage_time_padding=10,
):
    if filterbank is None:
        filterbank = decidecade_filter(lower_bound=10, upper_bound=50_000, window_duration=1, overlap=0.5)
    if transmission_model is None:
        from .transmission_loss import MlogR
        transmission_model = MlogR(m=20)
    if background_noise is None:
        def background_noise(received_power, **kwargs):
            return received_power

    passage_powers = []
    for passage_idx, passage in enumerate(passages):
        cpa = positional.closest_point(passage.recording.sensor, passage.track)
        time_segments = positional.aspect_segments(
            reference=passage.recording.sensor,
            track=passage.track,
            angles=aspect_angles,
            segment_min_length=aspect_segment_length,
            segment_min_angle=aspect_segment_angle,
            segment_min_duration=aspect_segment_duration,
        )
        time_start = positional.time_to_datetime(time_segments.time.min()).subtract(seconds=passage_time_padding)
        time_stop = positional.time_to_datetime(time_segments.time.max()).add(seconds=passage_time_padding)
        time_data = passage.recording.sampling.subwindow(start=time_start, stop=time_stop).time_data()
        received_power = filterbank(time_data)

        segment_powers = []
        for segment_idx, segment in time_segments.groupby('segment'):
            received_segment = received_power.sampling.subwindow(segment).reduce(np.mean, dim='time')
            source = passage.track.sampling.subwindow(segment.time.sel(edge='center'))

            compensated_segment = background_noise(
                received_segment,
                receiver=passage.recording.sensor,
                time=source.time,
            )

            source_segment = transmission_model(
                compensated_segment,
                receiver=passage.recording.sensor,
                source=source,
            )
            source_segment = source_segment.assign_coords(
                segment=segment.segment,
                latitude=source.latitude,
                longitude=source.longitude,
                time=source.time,
            )
            segment_powers.append(source_segment)

        segment_powers = xr.concat(segment_powers, dim='segment')
        passage_powers.append(segment_powers.assign_coords(cpa=cpa.distance))
    source_powers = xr.concat(passage_powers, dim='passage')
    return source_powers


@_tools.prebind
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
        axis=time_signal.dims.index('time'),
    )
    dims = list(time_signal.dims)
    dims[dims.index('time')] = 'frequency'
    dims.append('time')
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


@_tools.prebind
def nth_decade_filter(
    time_signal,
    bands_per_decade,
    time_step=None,
    window_duration=None,
    overlap=None,
    lower_bound=None,
    upper_bound=None,
    hybrid_resolution=False,
    scaling='density',
):
    if None not in (time_step, window_duration):
        overlap = 1 - time_step / window_duration
    elif time_step is not None:
        if overlap is None:
            if hybrid_resolution:
                # Set overlap to achieve hybrid resolution
                overlap = 1 - time_step * hybrid_resolution
            else:
                overlap = 0.5
        window_duration = time_step / (1 - overlap)
    elif window_duration is not None:
        # Cannot set overlap from hybrid resolution, the window duration is already set.
        if overlap is None:
            overlap = 0.5
        time_step = window_duration * (1 - overlap)
    elif hybrid_resolution:
        window_duration = 1 / hybrid_resolution
        if overlap is None:
            overlap = 0.5
        time_step = window_duration * (1 - overlap)
    else:
        # TODO: We could possibly use the entire time signal?
        raise ValueError('Must give at least one of `time_step` and `window_duration`.')

    if not (lower_bound or hybrid_resolution):
        raise ValueError(
            'Cannot have a log-spaced filterbank without lower frequency bound. Specify either `lower_bound` or `hybrid_resolution`.'
        )

    # We could relax these if we want to interpolate. This needs to be implemented in the calculations below.
    if hybrid_resolution:
        if hybrid_resolution * window_duration < 1:
            raise ValueError(
                f'Hybrid filterbank with resolution of {hybrid_resolution:.2f} Hz '
                f'cannot be calculated from temporal windows of {window_duration:.2f} s.'
            )
    else:
        lowest_bandwidth = lower_bound * (10 ** (0.5 / bands_per_decade) - 10 ** (-0.5 / bands_per_decade))
        if lowest_bandwidth * window_duration < 1:
            raise ValueError(
                f'{bands_per_decade}th-decade filter band at {lower_bound:.2f} Hz with bandwidth of {lowest_bandwidth:.2f} Hz '
                f'cannot be calculated from temporal windows of {window_duration:.2f} s.'
            )

    spec = spectrogram(
        time_signal=time_signal,
        window_duration=window_duration,
        overlap=overlap,
        window=('tukey', 2 * overlap),
    ).transpose('frequency', ...)  # Put the frequency axis first for ease of indexing later

    log_band_scaling = 10 ** (0.5 / bands_per_decade)
    upper_bound = upper_bound or spec.frequency.data[-1] / log_band_scaling
    # Get frequency vectors
    if hybrid_resolution:
        minimum_bandwidth_frequency = hybrid_resolution / (log_band_scaling - 1 / log_band_scaling)
        first_log_idx = np.math.ceil(bands_per_decade * np.log10(minimum_bandwidth_frequency / 1e3))
        last_linear_idx = np.math.floor(minimum_bandwidth_frequency / hybrid_resolution)

        while (last_linear_idx + 0.5) * hybrid_resolution > 1e3 * 10 ** ((first_log_idx - 0.5) / bands_per_decade):
            # Condition is "upper edge of last linear band is higher than lower edge of first logarithmic band"
            last_linear_idx += 1
            first_log_idx += 1

        if last_linear_idx * hybrid_resolution > upper_bound:
            last_linear_idx = np.math.floor(upper_bound / hybrid_resolution)
    else:
        last_linear_idx = 0
        first_log_idx = np.round(bands_per_decade * np.log10(lower_bound / 1e3))

    last_log_idx = round(bands_per_decade * np.log10(upper_bound / 1e3))

    lin_centers = np.arange(last_linear_idx) * hybrid_resolution
    lin_lowers = lin_centers - 0.5 * hybrid_resolution
    lin_uppers = lin_centers + 0.5 * hybrid_resolution

    log_centers = 1e3 * 10 ** (np.arange(first_log_idx, last_log_idx + 1) / bands_per_decade)
    log_lowers = log_centers / log_band_scaling
    log_uppers = log_centers * log_band_scaling

    centers = np.concatenate([lin_centers, log_centers])
    lowers = np.concatenate([lin_lowers, log_lowers])
    uppers = np.concatenate([lin_uppers, log_uppers])

    spec_data = spec.data
    banded_data = np.full(centers.shape + spec_data.shape[1:], np.nan)
    spectral_resolution = 1 / window_duration

    for idx, (l, u) in enumerate(zip(lowers, uppers)):
        l_idx = np.math.floor(l / spectral_resolution + 0.5)  # + 0.5 to consider fft bin lower edge
        u_idx = np.math.ceil(u / spectral_resolution - 0.5)  # - 0.5 to consider fft bin upper edge
        l_idx = max(l_idx, 0)
        u_idx = min(u_idx, spec_data.shape[0] - 1)

        if l_idx == u_idx:
            # This can only happen if both frequencies l and u are within the same fft bin.
            # Since we don't allow the fft bins to be larger than the output bins, we thus have the exact same band.
            banded_data[idx] = spec_data[l_idx]
        else:
            first_weight = l_idx + 0.5 - l / spectral_resolution
            last_weight = u / spectral_resolution - u_idx + 0.5
            # Sum the components fully within the output bin `[l_idx + 1:u_idx]`, and weighted components partially in the band.
            this_band = (
                spec_data[l_idx + 1 : u_idx].sum(axis=0)
                + spec_data[l_idx] * first_weight
                + spec_data[u_idx] * last_weight
            )
            banded_data[idx] = this_band * (spectral_resolution / (u - l))  # Rescale the power density.
    banded = recordings.time_frequency_data(
        data=banded_data,
        start_time=spec.sampling.window.start,
        samplerate=spec.sampling.rate,
        frequency=centers,
        bandwidth=uppers - lowers,
        dims=('frequency',) + spec.dims[1:],
    )
    if not scaling == 'density':
        banded *= banded.bandwidth
    return banded


decidecade_filter = nth_decade_filter(bands_per_decade=10, hybrid_resolution=False)
hybrid_millidecade_filter = nth_decade_filter(bands_per_decade=1000, hybrid_resolution=1)
