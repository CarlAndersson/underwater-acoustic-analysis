"""Implementations for spectral filterbanks.

.. currentmodule:: uwacan._filterbank

Core processing and analysis
----------------------------
.. autosummary::
    :toctree: generated

    Filterbank
    FilterbankRoller
    spectrum
    linear_to_banded

"""


from . import _core
import xarray as xr
import numpy as np
import scipy.signal
import numba


def spectrum(time_data, window=None, scaling="density", nfft=None, detrend=True, samplerate=None, axis=None):
    """Compute the power spectrum of time-domain data.

    The `spectrum` function calculates the power spectrum of input time-series data. It supports
    various input types, including `~uwacan.TimeData`, `xarray.DataArray`, and NumPy arrays. The
    function applies windowing, detrending, and scaling as specified by the parameters to
    produce the frequency-domain representation of the data.

    Parameters
    ----------
    time_data : _core.TimeData or xr.DataArray or numpy.ndarray
        The input time-domain data to compute the spectrum for. The data can be one of the following:

        - `~uwacan.TimeData`: Wrapped time data from ``uwacan``.
        - `xarray.DataArray`: An xarray DataArray with a 'time' dimension.
        - `numpy.ndarray`: A NumPy array containing time-series data.

    window : str or array_like, optional
        The window function to apply to the data before computing the FFT. This can be:

        - A string specifying the type of window to use (e.g., ``"hann"``, ``"kaiser"``, ``"blackman"``).
        - An array-like sequence of window coefficients.
        - If ``None``, no window is applied. Default is ``None``.

    scaling : {'density', 'spectrum', 'dc-nyquist'} or numeric, optional
        Specifies the scaling of the power spectrum. Options include:

        - ``'density'``: Computes the power spectral density.
        - ``'spectrum'``: Computes the power spectrum.
        - ``'dc-nyquist'``: Halves the output at DC and Nyquist frequencies. Use with a pre-scaled window that takes care to scale the remainder of the single-sided spectrum.
        - any numeric value: The output of the fft will be scaled by this value.

        Default is ``"density"``.

    nfft : int, optional
        The number of points to use in the FFT computation. If ``None``, it defaults to the length
        of the input data along the specified axis.

    detrend : bool, default=True
        If ``True``, removes the mean from the data before computing the FFT to reduce spectral leakage.
        If ``False``, no detrending is performed.

    samplerate : float, optional
        The sampling rate of the input data in Hz. Required if ``time_data`` is an numpy array.
        If not provided, it defaults to 1. This parameter is used to compute the frequency axis and proper density scaling.

    axis : int, optional
        The axis along which to compute the FFT. If ``None``, the last axis is used.
        Only used for numpy inputs.
        This parameter allows flexibility in handling multi-dimensional data.

    Returns
    -------
    _core.FrequencyData or xr.DataArray or numpy.ndarray
        The computed power spectrum of the input data. The return type matches the input type:

        - If ``time_data`` is a `~uwacan.TimeData`, returns a `~uwacan.FrequencyData` object.
        - If ``time_data`` is an `xarray.DataArray`, returns an `xarray.DataArray` with a 'frequency' dimension.
        - If ``time_data`` is a `numpy.ndarray`, returns a NumPy array containing the power spectrum.

    """
    if isinstance(time_data, _core.TimeData):
        return _core.FrequencyData(spectrum(time_data.data, window=window, scaling=scaling, nfft=nfft, detrend=detrend))
    if isinstance(time_data, xr.DataArray):
        freq_data = xr.apply_ufunc(
            spectrum,
            time_data,
            input_core_dims=[["time"]],
            output_core_dims=[["frequency"]],
            kwargs=dict(
                window=window,
                scaling=scaling,
                nfft=nfft,
                detrend=detrend,
                samplerate=samplerate or time_data.time.attrs.get("rate", None),
            ),
        )
        freq_data.coords["frequency"] = np.fft.rfftfreq(nfft or time_data.time.size, 1 / time_data.time.rate)
        freq_data.coords["time"] = time_data.time[0] + np.timedelta64(
            round(time_data.time.size / 2 / time_data.time.rate * 1e9), "ns"
        )
        return freq_data

    if axis is not None:
        time_data = np.moveaxis(time_data, axis, -1)

    if detrend:
        time_data = time_data - time_data.mean(axis=-1, keepdims=True)

    if window is not None:
        if not isinstance(window, np.ndarray):
            window = scipy.signal.windows.get_window(window, time_data.shape[-1], False)
        time_data = time_data * window

    nfft = nfft or time_data.shape[-1]
    freq_data = np.fft.rfft(time_data, axis=-1, n=nfft)
    freq_data = np.abs(freq_data) ** 2

    if scaling == "density":
        samplerate = samplerate or 1
        if window is not None:
            scaling = 2 / (np.sum(window**2) * samplerate)
        else:
            scaling = 2 / (time_data.shape[-1] * samplerate)
    elif scaling == "spectrum":
        if window is not None:
            scaling = 2 / np.sum(window) ** 2
        else:
            scaling = 2 / time_data.shape[-1]
    elif scaling == "dc-nyquist":
        # Remove doubling of DC
        freq_data[..., 0] /= 2
        if nfft % 2 == 0:
            # Even size, remove doubling of Nyquist
            freq_data[..., -1] /= 2
        scaling = False

    if scaling:
        freq_data *= scaling

        # Remove doubling of DC
        freq_data[..., 0] /= 2
        if nfft % 2 == 0:
            # Even size, remove doubling of Nyquist
            freq_data[..., -1] /= 2

    if axis is not None:
        freq_data = np.moveaxis(freq_data, -1, axis)

    return freq_data


@numba.njit()
def _linear_to_banded(linear_spectrum, lower_edges, upper_edges, spectral_resolution):
    banded_spectrum = np.full(lower_edges.shape + linear_spectrum.shape[1:], np.nan)
    for band_idx, (lower_edge, upper_edge) in enumerate(zip(lower_edges, upper_edges)):
        lower_idx = int(np.floor(lower_edge / spectral_resolution + 0.5))  # (l_idx - 0.5) * Δf = l
        upper_idx = int(np.ceil(upper_edge / spectral_resolution - 0.5))  # (u_idx + 0.5) * Δf = u
        lower_idx = max(lower_idx, 0)
        upper_idx = min(upper_idx, linear_spectrum.shape[0] - 1)

        if lower_idx == upper_idx:
            # This can only happen if both frequencies l and u are within the same fft bin.
            # Since we don't allow the fft bins to be larger than the output bins, we thus have the exact same band.
            banded_spectrum[band_idx] = linear_spectrum[lower_idx]
        else:
            # weight edge bins by "(whole bin - what is not in the new band) / whole bin"
            # lower fft bin edge l_e = (l_idx - 0.5) * Δf
            # w_l = (Δf - (l - l_e)) / Δf = l_idx + 0.5 - l / Δf
            first_weight = lower_idx + 0.5 - lower_edge / spectral_resolution
            # upper fft bin edge u_e = (u_idx + 0.5) * Δf
            # w_u = (Δf - (u_e - u)) / Δf = 0.5 - u_idx + u / Δf
            last_weight = upper_edge / spectral_resolution - upper_idx + 0.5
            # Sum the components fully within the output bin `[l_idx + 1:u_idx]`, and weighted components partially in the band.
            this_band = (
                linear_spectrum[lower_idx + 1 : upper_idx].sum(axis=0)
                + linear_spectrum[lower_idx] * first_weight
                + linear_spectrum[upper_idx] * last_weight
            )
            banded_spectrum[band_idx] = this_band * (
                spectral_resolution / (upper_edge - lower_edge)
            )  # Rescale the power density.
    return banded_spectrum


def linear_to_banded(linear_spectrum, lower_edges, upper_edges, spectral_resolution, axis=0):
    """Aggregate a linear power spectrum into specified frequency bands.

    The `linear_to_banded` function converts a linear power spectrum into a banded spectrum by
    summing power within frequency bands defined by ``lower_edges`` and ``upper_edges``. It handles
    multi-dimensional spectra by allowing specification of the axis corresponding to frequency
    bins.

    Parameters
    ----------
    linear_spectrum : `numpy.ndarray`
        The input linear power spectrum. The axis specified by ``axis`` should correspond to
        frequency bins.
    lower_edges : array_like
        The lower frequency edges for each band. Must be in ascending order.
    upper_edges : array_like
        The upper frequency edges for each band. Must be in ascending order and greater
        than or equal to ``lower_edges``.
    spectral_resolution : float
        The frequency resolution (Δf) of the linear spectrum.
    axis : int, optional, default=0
        The axis of ``linear_spectrum`` that corresponds to frequency bins. If the frequency
        bins are not along the first axis, specify the appropriate axis index.

    Returns
    -------
    banded_spectrum : numpy.ndarray
        The aggregated banded power spectrum.

    """
    # TODO: add features here to allow non-numpy inputs. Simply unwrap and rewrap as needed.
    if axis:
        linear_spectrum = np.moveaxis(linear_spectrum, axis, 0)
    banded = _linear_to_banded(linear_spectrum, lower_edges, upper_edges, spectral_resolution)
    if axis:
        banded = np.moveaxis(banded, 0, axis)
    return banded


class FilterbankRoller(_core.Roller):
    """Rolling computation of power spectra and spectrograms, both linear and banded.

    Parameters
    ----------
    time_data : TimeData
        The time-data wrapper to process.
    duration : float, optional
        The duration of the fft windows, in seconds.
    step : float, optional
        The step size between consecutive fft windows, in seconds.
    overlap : float, optional
        The amount of overlap between fft windows, as a fraction of the duration.
    min_frequency : float
        The lowest frequency to include in the processing.
    max_frequency : float
        The highest frequency to include in the processing.
    bands_per_decade : float, optional
        The number of frequency bands per decade for logarithmic scaling.
    hybrid_resolution : float
        A frequency resolution to aim for. Only used if ``frame_duration`` is not given
    fft_window : str, default="hann"
        The window function to apply to each rolling window before computing the FFT.
        Can be a string specifying a window type (e.g., ``"hann"``, ``"kaiser"``, ``"blackman"``)
        or an array-like sequence of window coefficients..

    Notes
    -----
    The processing is done in stft frames determined by ``duration``, ``step``
    ``overlap``, and ``hybrid_resolution``. At least one of ``duration``, ``step``,
    or ``resolution`` has to be given, see `~_core.time_frame_settings` for further details.
    At least one of ``min_frequency`` and ``hybrid_resolution`` has to be given.
    Note that the ``duration`` and ``step`` can be auto-chosen from the overlap
    and required frequency resolution, either from ``hybrid_resolution`` or ``min_frequency``.

    Raises
    ------
    ValueError
        If the processing settings are not compatible, e.g.,
        - frequency bands with bandwidth smaller than the frame duration allows
    """

    def __init__(
        self,
        time_data,
        duration=None,
        step=None,
        overlap=None,
        min_frequency=None,
        max_frequency=None,
        bands_per_decade=None,
        hybrid_resolution=None,
        fft_window="hann",
    ):
        self.settings = _core.time_frame_settings(
            duration=duration,
            step=step,
            overlap=overlap,
            resolution=None if isinstance(hybrid_resolution, bool) else hybrid_resolution,
            signal_length=time_data.time_window.duration,
            samplerate=time_data.samplerate,
        )
        self.min_frequency = min_frequency or 0
        self.max_frequency = max_frequency or time_data.samplerate / 2

        self.bands_per_decade = bands_per_decade
        self.hybrid_resolution = hybrid_resolution
        self.fft_window = fft_window

        self.time_data = time_data
        self.roller = self.time_data.rolling(
            duration=self.settings["duration"], step=self.settings["step"], overlap=self.settings["overlap"]
        )

        self.processing_axis = self.roller.dims.index("time")
        self.check_frequency_resolution()
        self.make_frequency_vectors()

    @property
    def dims(self):  # noqa: D102
        dims = list(self.roller.dims)
        dims[self.processing_axis] = "frequency"
        return tuple(dims)

    @property
    def shape(self):  # noqa: D102
        shape = list(self.roller.shape)
        shape[self.processing_axis] = len(self.frequency)
        return tuple(shape)

    @property
    def coords(self):  # noqa: D102
        coords = dict(self.roller.coords)
        del coords["time"]
        coords["frequency"] = xr.DataArray(self.frequency, dims="frequency", coords={"frequency": self.frequency})
        return coords

    def check_frequency_resolution(self):
        """Validate the frequency resolution against the temporal window settings."""
        # TODO: Check what this used to do when you used parameters from a spectrogram object.
        if not self.bands_per_decade:
            self.bands_per_decade = False
            self.hybrid_resolution = False
        else:
            if not self.hybrid_resolution:
                self.hybrid_resolution = False
                # Not using hybrid, we need long enough frames to compute the lowest band.
                lowest_bandwidth = self.min_frequency * (
                    10 ** (0.5 / self.bands_per_decade) - 10 ** (-0.5 / self.bands_per_decade)
                )
                if lowest_bandwidth * self.settings["duration"] < 1:
                    raise ValueError(
                        f'{self.bands_per_decade}th-decade filter band at {self.min_frequency:.2f} Hz with bandwidth of {lowest_bandwidth:.2f} Hz '
                        f'cannot be calculated from temporal windows of {self.settings["duration"]:.2f} s.'
                    )
            else:
                # Get the hybrid resolution settings.
                if self.hybrid_resolution is True:
                    self.hybrid_resolution = 1 / self.settings["duration"]
                if self.hybrid_resolution * self.settings["duration"] < 1:
                    raise ValueError(
                        f'Hybrid filterbank with resolution of {self.hybrid_resolution:.2f} Hz '
                        f'cannot be calculated from temporal windows of {self.settings["duration"]:.2f} s.'
                    )

    def make_frequency_vectors(self):
        """Construct frequency vectors and band definitions based on spectrogram settings."""
        nfft = self.settings["samples_per_frame"]
        self.linear_frequency = np.arange(nfft // 2 + 1) * self.time_data.samplerate / nfft
        self.bandwidth = self.linear_frequency[1]

        if self.max_frequency < self.linear_frequency[-1]:
            upper_index = int(np.floor(self.max_frequency / self.time_data.samplerate * nfft))
        else:
            upper_index = None

        if self.min_frequency > 0:
            lower_index = int(np.ceil(self.min_frequency / self.time_data.samplerate * nfft))
        else:
            lower_index = None

        if upper_index or lower_index:
            self.linear_slice = (slice(None),) * self.processing_axis + (slice(lower_index, upper_index),)
        else:
            self.linear_slice = False

        if self.linear_slice:
            self.frequency = self.linear_frequency[self.linear_slice]

        if self.bands_per_decade:
            log_band_scaling = 10 ** (0.5 / self.bands_per_decade)
            if self.hybrid_resolution:
                # The frequency at which the logspaced bands cover at least one linspaced band
                minimum_bandwidth_frequency = self.hybrid_resolution / (log_band_scaling - 1 / log_band_scaling)
                first_log_idx = int(
                    np.ceil(self.bands_per_decade * np.log10(minimum_bandwidth_frequency / 1e3))
                )
                last_linear_idx = int(np.floor(minimum_bandwidth_frequency / self.hybrid_resolution))

                # Since the logspaced bands have pre-determined centers, we can't just start them after the linspaced bands.
                # Often, the bands will overlap at the minimum bandwidth frequency, so we look for the first band
                # that does not overlap, i.e., the upper edge of the last linspaced band is below the lower edge of the first
                # logspaced band
                while (last_linear_idx + 0.5) * self.hybrid_resolution > 1e3 * 10 ** (
                    (first_log_idx - 0.5) / self.bands_per_decade
                ):
                    # Condition is "upper edge of last linear band is higher than lower edge of first logarithmic band"
                    last_linear_idx += 1
                    first_log_idx += 1

                if last_linear_idx * self.hybrid_resolution > self.max_frequency:
                    last_linear_idx = int(np.floor(self.max_frequency / self.hybrid_resolution))
                first_linear_idx = int(np.ceil(self.min_frequency / self.hybrid_resolution))
            else:
                first_linear_idx = last_linear_idx = 0
                first_log_idx = np.round(self.bands_per_decade * np.log10(self.min_frequency / 1e3))

            last_log_idx = round(self.bands_per_decade * np.log10(self.max_frequency / 1e3))

            lin_centers = np.arange(first_linear_idx, last_linear_idx) * self.hybrid_resolution
            lin_lowers = lin_centers - 0.5 * self.hybrid_resolution
            lin_uppers = lin_centers + 0.5 * self.hybrid_resolution

            log_centers = 1e3 * 10 ** (np.arange(first_log_idx, last_log_idx + 1) / self.bands_per_decade)
            log_lowers = log_centers / log_band_scaling
            log_uppers = log_centers * log_band_scaling

            centers = np.concatenate([lin_centers, log_centers])
            lowers = np.concatenate([lin_lowers, log_lowers])
            uppers = np.concatenate([lin_uppers, log_uppers])

            if centers[0] < self.min_frequency:
                mask = centers >= self.min_frequency
                lowers = lowers[mask]
                centers = centers[mask]
                uppers = uppers[mask]
            if centers[-1] > self.max_frequency:
                mask = centers <= self.max_frequency
                lowers = lowers[mask]
                centers = centers[mask]
                uppers = uppers[mask]

            self.band_lower_edges = lowers
            self.band_centers = centers
            self.band_upper_edges = uppers
            self.frequency = centers
            self.bandwidth = uppers - lowers

    def numpy_frames(self):  # noqa: D102
        window = scipy.signal.windows.get_window(self.fft_window, self.settings["samples_per_frame"], False)
        window /= ((window**2).sum() * self.time_data.samplerate / 2) ** 0.5

        for idx, time_frame in enumerate(self.roller.numpy_frames()):
            freq_frame = spectrum(time_frame, window=window, scaling="dc-nyquist", axis=self.processing_axis)
            if self.bands_per_decade:
                freq_frame = linear_to_banded(
                    freq_frame,
                    lower_edges=self.band_lower_edges,
                    upper_edges=self.band_upper_edges,
                    spectral_resolution=self.settings["resolution"],
                    axis=self.processing_axis,
                )
            elif self.linear_slice:
                freq_frame = freq_frame[self.linear_slice]
            yield freq_frame

    def __iter__(self):
        start_time = _core.time_to_np(self.time_data.time_window.start)
        start_time += np.timedelta64(
            int(self.settings["samples_per_frame"] / 2 / self.time_data.samplerate * 1e9), "ns"
        )
        for frame_idx, frame in enumerate(self.numpy_frames()):
            time_since_start = frame_idx * self.settings["sample_step"] / self.time_data.samplerate
            time_since_start = np.timedelta64(int(time_since_start * 1e9), "ns")
            frame = _core.FrequencyData(
                frame,
                frequency=self.frequency,
                bandwidth=self.bandwidth,
                coords=self.coords,
                dims=self.dims,
            )
            frame.data["time"] = start_time + time_since_start
            yield frame

    def batches(self, batch_size):
        """Generate batches of spectrogram frames.

        Parameters
        ----------
        batch_size : int
            Number of frames to include in each batch.

        Yields
        ------
        TimeFrequencyData
            Batches of spectrogram frames with time and frequency information.
            Each batch contains up to `batch_size` frames, with the final batch
            potentially containing fewer frames if the total number of frames
            is not evenly divisible by the batch size.
        """
        batch_output = np.zeros((batch_size,) + self.shape)
        for frame_idx, frame in enumerate(self.numpy_frames()):
            if frame_idx % batch_size == 0:
                batch_start_time = self.time_data.time_window.start.add(seconds=frame_idx * self.settings["step"])
            batch_output[frame_idx % batch_size] = frame
            if (frame_idx + 1) % batch_size == 0 or frame_idx == self.num_frames - 1:
                batch_data = _core.TimeFrequencyData(
                    batch_output[:frame_idx % batch_size + 1].copy(),  # Only use the frames we've filled
                    frequency=self.frequency,
                    bandwidth=self.bandwidth,
                    samplerate=self.time_data.samplerate / self.settings["sample_step"],
                    start_time=batch_start_time.add(seconds=self.settings["duration"] / 2),
                    coords=self.coords,
                    dims=("time",) + self.dims,
                    attrs=dict(
                        frame_duration=self.settings["duration"],
                        frame_overlap=self.settings["overlap"],
                        frame_step=self.settings["step"],
                        bands_per_decade=self.bands_per_decade,
                        hybrid_resolution=self.hybrid_resolution,
                    ),
                )
                yield batch_data


class Filterbank:
    """Calculates spectrograms, both linear and banded.

    Parameters
    ----------
    bands_per_decade : float, optional
        The number of frequency bands per decade for logarithmic scaling.
    frame_step : float
        The time step between stft frames, in seconds.
    frame_duration : float
        The duration of each stft frame, in seconds.
    frame_overlap : float, default=0.5
        The overlap factor between stft frames. A negative value leaves
        gaps between frames.
    min_frequency : float
        The lowest frequency to include in the processing.
    max_frequency : float
        The highest frequency to include in the processing.
    hybrid_resolution : float
        A frequency resolution to aim for. Only used if ``frame_duration`` is not given
    fft_window : str, default="hann"
        The window function to apply to each rolling window before computing the FFT.
        Can be a string specifying a window type (e.g., ``"hann"``, ``"kaiser"``, ``"blackman"``)
        or an array-like sequence of window coefficients..

    Notes
    -----
    The processing is done in stft frames determined by ``frame_duration``, ``frame_step``
    ``frame_overlap``, and ``hybrid_resolution``. At least one of ``duration``, ``step``,
    or ``resolution`` has to be given, see `~_core.time_frame_settings` for further details.
    At least one of ``min_frequency`` and ``hybrid_resolution`` has to be given.
    Note that the ``frame_duration`` and ``frame_step`` can be auto-chosen from the overlap
    and required frequency resolution, either from ``hybrid_resolution`` or ``min_frequency``.

    Raises
    ------
    ValueError
        If the processing settings are not compatible, e.g.,
        - frequency bands with bandwidth smaller than the frame duration allows
    """

    def __init__(
        self,
        *,
        bands_per_decade=None,
        frame_step=None,
        frame_duration=None,
        frame_overlap=0.5,
        min_frequency=None,
        max_frequency=None,
        hybrid_resolution=None,
        fft_window="hann",
    ):
        self.frame_duration = frame_duration
        self.frame_overlap = frame_overlap
        self.frame_step = frame_step
        self.fft_window = fft_window
        self.bands_per_decade = bands_per_decade
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.hybrid_resolution = hybrid_resolution

    def rolling(self, time_data):
        roller = FilterbankRoller(
            time_data=time_data,
            duration=self.frame_duration,
            step=self.frame_step,
            overlap=self.frame_overlap,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            bands_per_decade=self.bands_per_decade,
            hybrid_resolution=self.hybrid_resolution,
            fft_window=self.fft_window,
        )
        return roller

    def __call__(self, time_data):
        """Process time data to spectrograms.

        Parameters
        ----------
        time_data : `~uwacan.TimeData` or `~uwacan.recordings.AudioFileRecording`
            The data to process.

        Returns
        -------
        filtered_data : `~uwacan.TimeFrequencyData`
        """
        roller = self.rolling(time_data)
        return next(roller.batches(roller.num_frames))
