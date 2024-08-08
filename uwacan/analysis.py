"""Various analysis protocols and standards for recorded underwater noise from ships."""
import numpy as np
from . import _core, propagation
import scipy.signal
import xarray as xr


class Spectrogram(_core.TimeFrequencyData):
    """Calculates spectrograms

    The processing is done in stft frames determined by `frame_duration`, `frame_step`
    `frame_overlap`, and `frequency_resolution`. At least one of "duration", "step",
    or "resolution" has to be given, see `time_frame_settings` for further details.

    Parameters
    ----------
    data : TimeData, optional
        The time data from which to calculate the spectrogram.
        Omit this to create a callable object for later evaluation.
    frame_duration : float
        The duration of each stft frame, in seconds.
    frame_step : float
        The time step between stft frames, in seconds.
    frame_overlap : float, default 0.5
        The overlap factor between stft frames. A negative value leaves
        gaps between frames.
    frequency_resolution : float
        A frequency resolution to aim for. Only used if `frame_duration` is not given.
    fft_window : str, default="hann"
        The shape of the window used for the stft.
    """
    def __init__(
            self,
            data=None,
            frame_duration=None,
            frame_step=None,
            frame_overlap=0.5,
            frequency_resolution=None,
            fft_window="hann",
            **kwargs
        ):
        super().__init__(data, **kwargs)
        try:
            self.frame_settings = time_frame_settings(
                duration=frame_duration,
                step=frame_step,
                resolution=frequency_resolution,
                overlap=frame_overlap,
            ) | {"window": fft_window}
        except ValueError:
            self.frame_settings = None

        if isinstance(data, _core.TimeData):
            # __call__ will create an object that we only use the ._data from
            self._data = self(data).data

    def __call__(self, time_data):
        if isinstance(time_data, type(self)):
            return time_data
        xr_data = time_data.data
        frame_samples = round(self.frame_settings["duration"] * time_data.samplerate)
        overlap_samples = round(self.frame_settings["duration"] * self.frame_settings["overlap"] * time_data.samplerate)
        f, t, Sxx = scipy.signal.spectrogram(
            x=xr_data.data,  # Gets the numpy array
            fs=time_data.samplerate,
            window=self.frame_settings["window"],
            nperseg=frame_samples,
            noverlap=overlap_samples,
            axis=xr_data.dims.index('time'),
        )
        dims = list(xr_data.dims)
        dims[dims.index('time')] = 'frequency'
        dims.append('time')
        new = type(self)(
            data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
            # The array returned from the spectrogram function is the real part of the original stft, reshaped.
            # This means that the array takes twice the memory (the imaginary part is still around),
            # and it's not contiguous which slows down filtering a lot.
            samplerate=time_data.samplerate / (frame_samples - overlap_samples),
            start_time=time_data.time_window.start.add(seconds=t[0]),
            frequency=f,
            bandwidth=time_data.samplerate / frame_samples,
            dims=tuple(dims),
            coords=xr_data.coords,
        )
        self._transfer_attributes(new)
        return new

    def _transfer_attributes(self, obj):
        super()._transfer_attributes(obj)
        obj.frame_settings = self.frame_settings

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            raise AttributeError(
                f"'{type(self.__name__)}' object has no stored data. "
                "This instance is likely a callable processing object, not a data container."
            )


class NthDecadeSpectrogram(_core.TimeFrequencyData):
    """Calculates Nth-decade spectrograms.

    The processing is done in stft frames determined by `frame_duration`, `frame_step`
    `frame_overlap`, and `hybrid_resolution`. At least one of "duration", "step",
    or "resolution" has to be given, see `time_frame_settings` for further details.
    At least one of `lower_bound` and `hybrid_resolution` has to be given.
    Note that the `frame_duration` and `frame_step` can be auto-chosen from the overlap
    and required frequency resolution, either from `hybrid_resolution` or `lower_bound`.

    Parameters
    ----------
    data : TimeData or Spectrogram, optional
        The time data from which to calculate the spectrogram.
        Omit this to create a callable object for later evaluation.
    frame_duration : float
        The duration of each stft frame, in seconds.
    frame_step : float
        The time step between stft frames, in seconds.
    frame_overlap : float, default 0.5
        The overlap factor between stft frames. A negative value leaves
        gaps between frames.
    lower_bound : float
        The lowest frequency to include in the processing.
    upper_bound : float
        The highest frequency to include in the processing.
    hybrid_resolution : float
        A frequency resolution to aim for. Only used if `frame_duration` is not given.
    scaling : str, default="power spectral density"
        The scaling to use for the output.
        - "power spectral density" scales the output as a power spectral density.
        - "power spectrum" scales the output as the total power in each band.
    fft_window : str, default="hann"
        The shape of the window used for the stft.

    Raises
    ------
    ValueError
        If the processing settings are not compatible, e.g.,
        - frequency bands with bandwidth smaller than the frame duration allows,
        - no lower bound and no hybrid resolution.
    """
    def __init__(
        self,
        data=None,
        bands_per_decade=None,
        frame_step=None,
        frame_duration=None,
        frame_overlap=0.5,
        lower_bound=None,
        upper_bound=None,
        hybrid_resolution=None,
        fft_window="hann",
        **kwargs
    ):
        super().__init__(data, **kwargs)
        try:
            # We cannot use any form of `hybrid_resolution in (True, False, None)`
            # since they use `==`` and not `is` and `1 == True` >> True
            if not (
                hybrid_resolution is True
                or hybrid_resolution is False
                or hybrid_resolution is None
            ):
                resolution = hybrid_resolution
            elif lower_bound is not None and bands_per_decade is not None:
                resolution = lower_bound * (10 ** (0.5 / bands_per_decade) - 10 ** (-0.5 / bands_per_decade))
            else:
                resolution = None
            self.frame_settings = time_frame_settings(
                duration=frame_duration,
                step=frame_step,
                resolution=resolution,
                overlap=frame_overlap,
            ) | {"window": fft_window}
        except ValueError:
            self.frame_settings = None

        self.bands_per_decade = bands_per_decade
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.hybrid_resolution = hybrid_resolution

        if isinstance(data, _core.TimeData):
            # __call__ will create an object that we only use the .data from
            self._data = self(data).data

    def __call__(self, data):
        if isinstance(data, type(self)):
            return data
        if not isinstance(data, Spectrogram):
            data = Spectrogram(
                data,
                frame_step=self.frame_settings["step"],
                frame_duration=self.frame_settings["duration"],
                frame_overlap=self.frame_settings["overlap"],
                frequency_resolution=self.frame_settings["resolution"],
            )

        if self.hybrid_resolution:
            if self.hybrid_resolution is True:
                hybrid_resolution = 1 / data.frame_settings["duration"]
            else:
                hybrid_resolution = self.hybrid_resolution
            if hybrid_resolution * data.frame_settings["duration"] < 1:
                raise ValueError(
                    f'Hybrid filterbank with resolution of {hybrid_resolution:.2f} Hz '
                    f'cannot be calculated from temporal windows of {data.frame_settings["duration"]:.2f} s.'
                )
        else:
            hybrid_resolution = False
            lowest_bandwidth = self.lower_bound * (10 ** (0.5 / self.bands_per_decade) - 10 ** (-0.5 / self.bands_per_decade))
            if lowest_bandwidth * data.frame_settings["duration"] < 1:
                raise ValueError(
                    f'{self.bands_per_decade}th-decade filter band at {self.lower_bound:.2f} Hz with bandwidth of {lowest_bandwidth:.2f} Hz '
                    f'cannot be calculated from temporal windows of {data.frame_settings["duration"]:.2f} s.'
                )

        spec_xr = data.data.transpose("frequency", ...)
        # Get frequency centers for the new bands
        bands_per_decade = self.bands_per_decade
        log_band_scaling = 10 ** (0.5 / bands_per_decade)
        lower_bound = self.lower_bound or 0  # Prevent None or False.
        upper_bound = self.upper_bound or spec_xr.frequency.data[-1] / log_band_scaling
        if hybrid_resolution:
            # The frequency at which the logspaced bands cover at least one linspaced band
            minimum_bandwidth_frequency = hybrid_resolution / (log_band_scaling - 1 / log_band_scaling)
            first_log_idx = np.math.ceil(bands_per_decade * np.log10(minimum_bandwidth_frequency / 1e3))
            last_linear_idx = np.math.floor(minimum_bandwidth_frequency / hybrid_resolution)

            # Since the logspaced bands have pre-determined centers, we can't just start them after the linspaced bands.
            # Often, the bands will overlap at the minimum bandwidth frequency, so we look for the first band
            # that does not overlap, i.e., the upper edge of the last linspaced band is below the lower edge of the first
            # logspaced band
            while (last_linear_idx + 0.5) * hybrid_resolution > 1e3 * 10 ** ((first_log_idx - 0.5) / bands_per_decade):
                # Condition is "upper edge of last linear band is higher than lower edge of first logarithmic band"
                last_linear_idx += 1
                first_log_idx += 1

            if last_linear_idx * hybrid_resolution > upper_bound:
                last_linear_idx = np.math.floor(upper_bound / hybrid_resolution)
            first_linear_idx = np.math.ceil(lower_bound / hybrid_resolution)
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

        # Aggregate input spectrum into bands
        spec_np = spec_xr.data
        banded_data = np.full(centers.shape + spec_np.shape[1:], np.nan)
        spectral_resolution = 1 / data.frame_settings["duration"]
        for idx, (l, u) in enumerate(zip(lowers, uppers)):
            l_idx = int(np.floor(l / spectral_resolution + 0.5))  # (l_idx - 0.5) * Δf = l
            u_idx = int(np.ceil(u / spectral_resolution - 0.5))  # (u_idx + 0.5) * Δf = u
            l_idx = max(l_idx, 0)
            u_idx = min(u_idx, spec_np.shape[0] - 1)

            if l_idx == u_idx:
                # This can only happen if both frequencies l and u are within the same fft bin.
                # Since we don't allow the fft bins to be larger than the output bins, we thus have the exact same band.
                banded_data[idx] = spec_np[l_idx]
            else:
                # weight edge bins by "(whole bin - what is not in the new band) / whole bin"
                # lower fft bin edge l_e = (l_idx - 0.5) * Δf
                # w_l = (Δf - (l - l_e)) / Δf = l_idx + 0.5 - l / Δf
                first_weight = l_idx + 0.5 - l / spectral_resolution
                # upper fft bin edge u_e = (u_idx + 0.5) * Δf
                # w_u = (Δf - (u_e - u)) / Δf = 0.5 - u_idx + u / Δf
                last_weight = u / spectral_resolution - u_idx + 0.5
                # Sum the components fully within the output bin `[l_idx + 1:u_idx]`, and weighted components partially in the band.
                this_band = (
                    spec_np[l_idx + 1 : u_idx].sum(axis=0)
                    + spec_np[l_idx] * first_weight
                    + spec_np[u_idx] * last_weight
                )
                banded_data[idx] = this_band * (spectral_resolution / (u - l))  # Rescale the power density.
        banded = type(self)(
            data=banded_data,
            start_time=spec_xr.time[0],
            samplerate=spec_xr.time.rate,
            frequency=centers,
            bandwidth=uppers - lowers,
            dims=spec_xr.dims,
            coords=spec_xr.coords,
        )
        self._transfer_attributes(banded)
        return banded

    def _transfer_attributes(self, obj):
        super()._transfer_attributes(obj)
        obj.frame_settings = self.frame_settings

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no stored data. "
                "This instance is likely a callable processing object, not a data container."
            ) from None


class ShipLevel(_core.DatasetWrap):
    """Calculates and stores measured ship levels."""
    @classmethod
    def analyze_transits(
        cls,
        *transits,
        filterbank=None,
        propagation_model=None,
        background_noise=None,
        transit_min_angle=None,
        transit_min_duration=None,
        transit_min_lengh=None,
    ):
        if filterbank is None:
            filterbank = NthDecadeSpectrogram(
                bands_per_decade=10,
                lower_bound=20,
                upper_bound=20_000,
                frame_step=1
            )

        if background_noise is None:
            def background_noise(received_power, **kwargs):
                return received_power

        if propagation_model is None:
            propagation_model = propagation.MlogR(m=20)

        if isinstance(propagation_model, propagation.PropagationModel):
            propagation_model = propagation_model.compensate_propagation

        results = []
        for transit in transits:
            if (transit_min_angle, transit_min_duration, transit_min_lengh) == (None, None, None):
                cpa_time = transit.track.closest_point(transit.recording.sensor)["time"].data
            else:
                segment = transit.track.aspect_segments(
                    reference=transit.recording.sensor,
                    angles=0,
                    segment_min_duration=transit_min_duration,
                    segment_min_angle=transit_min_angle,
                    segment_min_length=transit_min_lengh,
                )
                cpa_time = segment.time.sel(edge="center").data
                transit = transit.subwindow(segment)

            direction = transit.track.average_course('eight')
            time_data = transit.recording.time_data()
            received_power = filterbank(time_data)

            received_power = background_noise(received_power)
            track = transit.track.resample(received_power.time)
            source_power = propagation_model(received_power=received_power, receiver=transit.recording.sensor, source=track)
            transit_time = (received_power.data["time"] - cpa_time) / np.timedelta64(1, "s")
            closest_to_cpa = np.abs(transit_time).argmin("time").item()
            segment = xr.DataArray(np.arange(transit_time.time.size) - closest_to_cpa, coords={"time": received_power.time})
            transit_results = xr.Dataset(
                data_vars=dict(
                    source_power=source_power.data,
                    latitude=track.latitude,
                    longitude=track.longitude,
                    transit_time=transit_time,
                ),
                coords=dict(
                    segment=segment,
                    direction=direction,
                    cpa_time=cpa_time,
                )
            )
            transit_results["received_power"] = received_power.data
            if hasattr(received_power, "snr"):
                transit_results["snr"] = received_power.snr.data
            results.append(transit_results.swap_dims(time="segment"))
        results = xr.concat(results, "transit")
        results.coords["transit"] = np.arange(results.sizes["transit"]) + 1
        return cls(results)

    @classmethod
    def analyze_transits_in_angle_segments(
        cls,
        *transits,
        filterbank=None,
        propagation_model=None,
        background_noise=None,
        aspect_angles=tuple(range(-45, 46, 5)),
        segment_min_length=100,
        segment_min_angle=None,
        segment_min_duration=None,
    ):
        if filterbank is None:
            filterbank = NthDecadeSpectrogram(
                bands_per_decade=10,
                lower_bound=20,
                upper_bound=20_000,
                frame_step=1
            )

        try:
            transit_padding = filterbank.frame_settings["duration"]
        except AttributeError:
            transit_padding = 10

        if background_noise is None:
            def background_noise(received_power, **kwargs):
                return received_power

        if propagation_model is None:
            propagation_model = propagation.MlogR(m=20)

        if isinstance(propagation_model, propagation.PropagationModel):
            propagation_model = propagation_model.compensate_propagation

        results = []
        for transit in transits:
            segments = transit.track.aspect_segments(
                reference=transit.recording.sensor,
                angles=aspect_angles,
                segment_min_length=segment_min_length,
                segment_min_angle=segment_min_angle,
                segment_min_duration=segment_min_duration,
            )
            transit = transit.subwindow(segments, extend=transit_padding)

            if np.min(np.abs(segments.segment)) == 0:
                cpa_time = segments.sel(segment=0, edge="center")["time"].data
            else:
                cpa_time = transit.track.closest_point(transit.recording.sensor)["time"].data

            direction = transit.track.average_course('eight')
            time_data = transit.recording.time_data()
            received_power = filterbank(time_data)

            segment_powers = []
            for segment_angle, segment in segments.groupby("segment", squeeze=False):
                segment_power = received_power.subwindow(segment).mean("time").data
                segment_power.coords["segment"] = segment_angle
                segment_powers.append(segment_power)
            segment_powers = xr.concat(segment_powers, "segment")
            segment_powers = _core.FrequencyData(segment_powers)

            compensated_power = background_noise(segment_powers)
            track = transit.track.resample(segments.sel(edge="center", drop=True).time)
            source_power = propagation_model(received_power=segment_powers, receiver=transit.recording.sensor, source=track)
            transit_time = (track._data["time"] - cpa_time) / np.timedelta64(1, "s")

            transit_results = xr.Dataset(
                data_vars=dict(
                    source_power=source_power.data,
                    received_power=compensated_power.data,
                    latitude=track.latitude,
                    longitude=track.longitude,
                    transit_time=transit_time,
                ),
                coords=dict(
                    direction=direction,
                    cpa_time=cpa_time,
                )
            )
            if hasattr(compensated_power, "snr"):
                transit_results["snr"] = compensated_power.snr.data
            results.append(transit_results)
        results = xr.concat(results, "transit")
        results.coords["transit"] = np.arange(results.sizes["transit"]) + 1
        return cls(results)

    @property
    def source_power(self):
        return _core.FrequencyData(self._data["source_power"])

    @property
    def source_level(self):
        return _core.dB(self.source_power, power=True)

    @property
    def received_power(self):
        return _core.FrequencyData(self._data["received_power"])

    @property
    def received_level(self):
        return _core.dB(self.received_power, power=True)

    def mean(self, dims, **kwargs):
        return type(self)(self._data.mean(dims, **kwargs))

    @property
    def snr(self):
        return _core.FrequencyData(self._data["snr"])

    def meets_snr_threshold(self, threshold):
        snr = self._data["snr"]
        meets_threshold = xr.where(
            snr.isnull(),
            np.nan,
            snr > threshold,
        )
        return _core.FrequencyData(meets_threshold)


def time_frame_settings(
    duration=None,
    step=None,
    overlap=None,
    resolution=None,
    num_frames=None,
    signal_length=None,
):
    """Calculates time frame overlap settings from various input parameters.

    Parameters
    ----------
    duration : float
        How long each frame is, in seconds
    step : float
        The time between frame starts, in seconds
    overlap : float
        How much overlap there is between the frames, as a fraction of the duration.
        If this is negative the frames will have extra space between them.
    resolution : float
        Desired frequency resolution in Hz. Equals `1/duration`
    num_frames : int
        The total number of frames in the signal
    signal_length : float
        The total length of the signal, in seconds

    Returns
    -------
    dict with keys
        - "duration"
        - "step"
        - "overlap"
        - "resolution"
        and if `signal_length` was given
        - "num_frames"
        - "signal_length"

    Raises
    ------
    ValueError
        If the inputs are not sufficient to determine the frame,
        or if priorities cannot be disambiguated.

    Notes
    -----
    The parameters will be used in the following priority:
    1. signal_length, num_frames
    2. step, duration
    3. resolution (only if duration not given)
    4. overlap (default 0)

    Each frame idx=[0, ..., num_frames - 1] has
    - start = idx * step
    - stop = idx * step + duration

    The last frame thus ends at (num_frames - 1) step + duration.
    The overlap relations are:
    - duration = step / (1 - overlap)
    - step = duration (1 - overlap)
    - overlap = 1 - step / duration

    This gives us the following total list of priorities:
    1) `signal_length` and `num_frames` are given
        a) `step` or `duration` (not both!) given
        b) `resolution` is given
        c) `overlap` is given
    2) `step` and `duration` given
    3) `step` given
        a) `resolution` is given
        b) `overlap` is given
    4) `duration` given (`resolution` is ignored, `overlap` is used)
    5) `resolution` given
    For cases 2-5, `num_frames` is calculated if `signal_length` is given,
    and a new truncated `signal_length` is returned.
    """
    if None not in (num_frames, signal_length):
        if None not in (duration, step):
            raise ValueError("Overdetermined time frames")
        elif step is not None:
            # We have the step, calculate the duration
            duration = signal_length - (num_frames - 1) * step
            overlap = 1 - step / duration
        elif duration is not None:
            # We have the duration, calculate the step
            step = (signal_length - duration) / (num_frames - 1)
            overlap = 1 - step / duration
        elif resolution is not None:
            duration = 1 / resolution
            step = (signal_length - duration) / (num_frames - 1)
            overlap = 1 - step / duration
        else:
            overlap = overlap or 0
            duration = signal_length / (num_frames + overlap - num_frames * overlap)
            step = duration * (1 - overlap)
    elif None not in (step, duration):
        overlap = 1 - step / duration
    elif step is not None:
        if resolution is not None:
            duration = 1 / resolution
            overlap = 1 - step / duration
        else:
            overlap = overlap or 0
            duration = step / (1 - overlap)
    elif duration is not None:
        overlap = overlap or 0
        step = duration * (1 - overlap)
    elif resolution is not None:
        duration = 1 / resolution
        overlap = overlap or 0
        step = duration * (1 - overlap)
    else:
        raise ValueError("Must give at least one of (`step`, `duration`, `resolution`) or the pair of `signal_length` and `num_frames`.")

    settings = {
        "duration": duration,
        "step": step,
        "overlap": overlap,
        "resolution": 1 / duration,
    }
    if signal_length is not None:
        num_frames = num_frames or np.math.floor((signal_length - duration) / step + 1)
        settings["num_frames"] = num_frames
        settings["signal_length"] = (num_frames - 1) * step + duration
    return settings


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
