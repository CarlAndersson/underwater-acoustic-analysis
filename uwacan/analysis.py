"""Various analysis protocols and standards for recorded underwater noise from ships."""

import numpy as np
from . import positional, propagation, _tools
import scipy.signal
import xarray as xr


class _DataWrapper:
    @classmethod
    def _wrap_output(cls, data):
        if not isinstance(data, xr.DataArray):
            return data
        if 'time' in data.dims and 'frequency' in data.dims:
            return TimeFrequencyData(data)
        if 'time' in data.dims:
            return TimeData(data)
        if 'frequency' in data.dims:
            return FrequencyData(data)
        return data

    def __add__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(self.data + other)

    def __radd__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(other + self.data)

    def __sub__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(other - self.data)

    def __mul__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(self.data * other)

    def __rmul__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(other * self.data)

    def __truediv__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(self.data / other)

    def __rtruediv__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(other / self.data)

    def __floordiv__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(self.data // other)

    def __rfloordiv__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(other // self.data)

    def __pow__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(self.data ** other)

    def __rpow__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(other ** self.data)

    def __mod__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(self.data % other)

    def __rmod__(self, other):
        if isinstance(other, _DataWrapper):
            other = other.data
        return self._wrap_output(other % self.data)

    def __neg__(self):
        return self._wrap_output(-self.data)

    def __abs__(self):
        return self._wrap_output(abs(self.data))


class TimeData(_DataWrapper):
    @staticmethod
    def _with_time_vector(data, start_time, samplerate):
        if samplerate is None:
            return data
        if start_time is None:
            if 'time' in data.coords:
                start_time = data.time[0].item()
            start_time = 'now'
        n_samples = data.sizes['time']
        start_time = positional.time_to_np(start_time)
        offsets = np.arange(n_samples) * 1e9 / samplerate
        time = start_time + offsets.astype('timedelta64[ns]')
        return data.assign_coords(time=('time', time, {'rate': samplerate}))

    def __init__(self, data, start_time=None, samplerate=None, dims=None, coords=None):
        if not isinstance(data, xr.DataArray):
            if dims is None:
                if data.ndim == 1:
                    dims = 'time'
                else:
                    raise ValueError(f'Cannot guess dimensions for time data with {data.ndim} dimensions')
            data = xr.DataArray(data, dims=dims)

        data = data.assign_coords(coords)
        data = self._with_time_vector(data, samplerate=samplerate, start_time=start_time)
        self.data = data

    @property
    def samplerate(self):
        return self.data.time.rate

    @property
    def time_window(self):
        # Calculating duration from number and rate means the stop points to the sample after the last,
        # which is more intuitive when considering signal durations etc.
        return positional.TimeWindow(
            start=self.data.time.data[0],
            duration=self.data.sizes['time'] / self.samplerate,
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        original_window = self.time_window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        if isinstance(new_window, positional.TimeWindow):
            start = (new_window.start - original_window.start).total_seconds()
            stop = (new_window.stop - original_window.start).total_seconds()
            # Indices assumed to be seconds from start
            start = np.math.floor(start * self.samplerate)
            stop = np.math.ceil(stop * self.samplerate)
            idx = slice(start, stop)
        else:
            idx = (new_window - original_window.start).total_seconds()
            idx = round(idx * self.samplerate)

        selected_data = self.data.isel(time=idx)
        return type(self)(selected_data)


class FrequencyData(_DataWrapper):
    @staticmethod
    def _with_frequency_bandwidth_vectors(data, frequency, bandwidth):
        if frequency is None:
            return data
        coords = {'frequency': frequency}
        if bandwidth is not None:
            coords['bandwidth'] = ('frequency', np.broadcast_to(bandwidth, np.shape(frequency)))
        return data.assign_coords(coords)

    def __init__(self, data, frequency=None, bandwidth=None, dims=None, coords=None):
        if not isinstance(data, xr.DataArray):
            if dims is None:
                if data.ndim == 1:
                    dims = 'frequency'
                else:
                    raise ValueError(f'Cannot guess dimensions for frequency data with {data.ndim} dimensions')
            data = xr.DataArray(data, dims=dims)
        data = data.assign_coords(coords)
        data = self._with_frequency_bandwidth_vectors(data, frequency=frequency, bandwidth=bandwidth)
        self.data = data

    def estimate_bandwidth(self):
        frequency = np.asarray(self.data.frequency)
        # Check if the frequency array seems linearly or logarithmically spaced
        if frequency[0] == 0:
            diff = frequency[2:] - frequency[1:-1]
            frac = frequency[2:] / frequency[1:-1]
        else:
            diff = frequency[1:] - frequency[:-1]
            frac = frequency[1:] / frequency[:-1]
        diff_err = np.std(diff) / np.mean(diff)
        frac_err = np.std(frac) / np.mean(frac)
        # Note: if there are three values and the first is zero, the std is 0 for both.
        # The equals option makes us end up in the linear frequency case.
        if diff_err <= frac_err:
            # Central differences, with forwards and backwards at the ends
            central = (frequency[2:] - frequency[:-2]) / 2
            first = frequency[1] - frequency[0]
            last = frequency[-1] - frequency[-2]
        else:
            # upper edge is at sqrt(f_{l+1} * f_l), lower edge is at sqrt(f_{l-1} * f_l)
            # the difference simplifies as below.
            central = (frequency[2:]**0.5 - frequency[:-2]**0.5) * frequency[1:-1]**0.5
            # extrapolating to one bin below lowest and one above highest using constant ratio
            # the expression above then simplifies to the expressions below
            first = (frequency[1] - frequency[0]) * (frequency[0] / frequency[1])**0.5
            last = (frequency[-1] - frequency[-2]) * (frequency[-1] / frequency[-2])**0.5
        bandwidth = np.concatenate([[first], central, [last]])
        return xr.DataArray(bandwidth, coords={'frequency': self.data.frequency})


class TimeFrequencyData(TimeData, FrequencyData):
    def __init__(self, data, start_time=None, samplerate=None, frequency=None, bandwidth=None, dims=None, coords=None):
        if not isinstance(data, xr.DataArray):
            if dims is None:
                raise ValueError('Cannot guess dimensions for time-frequency data')
            data = xr.DataArray(data, dims=dims)
        data = data.assign_coords(coords)
        data = self._with_time_vector(data, samplerate=samplerate, start_time=start_time)
        data = self._with_frequency_bandwidth_vectors(data, frequency=frequency, bandwidth=bandwidth)
        self.data = data


class Transit:
    def __init__(self, recording, track):
        start = max(recording.time_window.start, track.time_window.start)
        stop = min(recording.time_window.stop, track.time_window.stop)

        self.recording = recording.subwindow(start=start, stop=stop)
        self.track = track.subwindow(start=start, stop=stop)

    @property
    def time_window(self):
        rec_window = self.recording.time_window
        track_window = self.track.time_window
        return positional.TimeWindow(start=max(rec_window.start, track_window.start), stop=min(rec_window.stop, track_window.stop))

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        subwindow = self.window.subwindow(time, start=start, stop=stop, center=center, duration=duration)
        rec = self.recording.subwindow(subwindow)
        track = self.track.subwindow(subwindow)
        return type(self)(recording=rec, track=track)


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


def bureau_veritas_source_spectrum(
    passages,
    propagation_model=None,
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
    if propagation_model is None:
        propagation_model = propagation.MlogR(m=20)

    if isinstance(propagation_model, propagation.PropagationModel):
        propagation_model = propagation_model.compensate_propagation

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
            source_segment = propagation_model(
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


def time_window_settings(
    duration=None,
    step=None,
    overlap=None,
    num_windows=None,
    signal_length=None,
):
    """Calculates time window overlap settings from various input parameters.

    Some shorthand definitions for the parameters above:
    - D = Duration, how long each window is.
    - S = Step, the time between window starts
    - O = Overlap, which is a fraction of the duration
    - N = The number of windows
    - L = The total signal length

    Each window index=[0, ..., N-1] has
    - start = idx * S
    - stop = idx * S + D

    The last window thus ends at (N-1) S + D.
    Finally, the overlap relations are
    - D = S / (1 - O)
    - S = D (1 - O)
    - O = 1 - S / D

    From these, a number of modes can be used:
    1) The signal length and number of windows are given. At most one of (D, S) can be given.
        The overlap can be given, but will only be used if both D and S are not given.
    2) S and D known, calculate O.
    3) S known, calculate D (assuming O=0 if not given).
    4) D known, calculate S (assuming O=0 if not given).
    For modes 2,3,4, N is calculated if L is given.
    """
    if None not in (num_windows, signal_length):
        if (duration, step, overlap) == (None, None, None):
            duration = step = signal_length / num_windows
            overlap = 0
        # elif (duration, overlap) == (None, None):
        elif None not in (duration, step):
            raise ValueError('Overdetermined time windows')
        elif step is not None:
            # We have the step, calculate the duration
            duration = signal_length - (num_windows - 1) * step
            overlap = 1 - step / duration
        # elif (step, overlap) == (None, None):
        elif duration is not None:
            # We have the duration, calculate the step
            step = (signal_length - duration) / (num_windows - 1)
            overlap = 1 - step / duration
        elif (duration, step) == (None, None):
            # We have the overlap, calculate duration and step
            duration = signal_length / (num_windows + overlap - num_windows * overlap)
            step = duration * (1 - overlap)
        else:
            raise ValueError('Must give at least one of `step` and `duration` or the pair of `signal_length` and `num_windows`.')
    elif None not in (step, duration):
        overlap = 1 - step / duration
    elif step is not None:
        overlap = overlap or 0
        duration = step / (1 - overlap)
    elif duration is not None:
        overlap = overlap or 0
        step = duration * (1 - overlap)
    else:
        raise ValueError('Must give at least one of `step` and `duration` or the pair of `signal_length` and `num_windows`.')

    settings = {
        'duration': duration,
        'step': step,
        'overlap': overlap,
    }
    if signal_length is not None:
        num_windows = num_windows or np.math.floor((signal_length - duration) / step + 1)
        settings['num_windows'] = num_windows
    return settings


@_tools.prebind
def fft(time_signal, nfft=None):
    nfft = nfft or time_signal.sizes['time']
    return xr.apply_ufunc(
        np.fft.rfft,
        time_signal,
        input_core_dims=[['time']],
        output_core_dims=[['frequency']],
        kwargs={'n': nfft},
    ).assign_coords(frequency=np.fft.rfftfreq(nfft, 1 / time_signal.time.rate), time=time_signal.time[0]).rename(time='start_time')


@_tools.prebind
def ifft(spectrum, nfft=None):
    if nfft is None:
        is_odd = np.any(spectrum.isel(frequency=-1).data.imag)
        nfft = (spectrum.sizes['frequency'] - 1) * 2 + (1 if is_odd else 0)

    time_data = xr.apply_ufunc(
        np.fft.irfft,
        spectrum,
        input_core_dims=[['frequency']],
        output_core_dims=[['time']],
        kwargs={'n': nfft},
    ).drop_vars('start_time')
    samplerate = spectrum.frequency[1].item() * nfft
    if hasattr(spectrum, 'start_time'):
        return recordings.time_data(time_data, start_time=spectrum.start_time, samplerate=samplerate)
    time_data.coords['time'] = np.arange(nfft) / samplerate
    time_data.coords['time'].attrs['rate'] = samplerate
    return time_data


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
        coords=time_signal.coords,
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
        if hybrid_resolution is True:
            hybrid_resolution = 1 / window_duration
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
        if lower_bound is not None:
            first_linear_idx = np.math.ceil(lower_bound / hybrid_resolution)
        else:
            first_linear_idx = 0
    else:
        first_linear_idx = last_linear_idx = 0
        first_log_idx = np.round(bands_per_decade * np.log10(lower_bound / 1e3))

    last_log_idx = round(bands_per_decade * np.log10(upper_bound / 1e3))

    lin_centers = np.arange(first_linear_idx, last_linear_idx) * hybrid_resolution
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
        coords=spec.coords,
    )
    if not scaling == 'density':
        banded *= banded.bandwidth
    return banded


decidecade_filter = nth_decade_filter(bands_per_decade=10, hybrid_resolution=False)
hybrid_millidecade_filter = nth_decade_filter(bands_per_decade=1000, hybrid_resolution=1)


def convert_to_radiated_noise(source_power, source_depth, mode=None, power=True):
    if mode is None or not mode:
        return source_power
    kd = 2 * np.pi * source_power.frequency / 1500 * source_depth
    mode = mode.lower()
    if mode == 'iso':
        compensation = (14 * kd**2 + 2 * kd**4) / (14 + 2 * kd**2 + kd**4)
    elif mode == 'average farfield':
        compensation = 1 / (1 / 2 + 1 / (2 * kd**2))
    elif mode == 'isomatch':
        truncation_angle = np.radians(54.3)
        lf_comp = 2 * kd**2 * (truncation_angle - np.sin(truncation_angle) * np.cos(truncation_angle)) / truncation_angle
        compensation = 1 / (1 / 2 + 1 / lf_comp)
    elif mode == 'none':
        compensation = 1
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    if power:
        return source_power * compensation
    else:
        return source_power + 10 * np.log10(compensation)
