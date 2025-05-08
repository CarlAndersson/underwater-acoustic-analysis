"""Implementations for spectral analysis.

.. currentmodule:: uwacan.spectral

Core processing and analysis
----------------------------
.. autosummary::
    :toctree: generated

    Spectrogram
    SpectralProbability

Helper functions and conversions
--------------------------------
.. autosummary::
    :toctree: generated

    spectrum
    linear_to_banded
    SpectrogramRollingComputation
    level_uncertainty
    required_averaging

"""

from uwacan import recordings
from . import _core
import xarray as xr
import numpy as np
import scipy.signal
import numba


def level_uncertainty(averaging_time, bandwidth):
    r"""Compute the level uncertainty for a specific averaging time and frequency bandwidth.

    The level uncertainty here is derived from the mean and standard deviation of the power
    of sampled white gaussian noise. The uncertainty is the decibel difference between
    one half standard deviation above the mean and one half standard deviation below the mean.
    This is very similar to taking the standard deviation of the levels instead of the powers.

    Parameters
    ----------
    averaging_time : float
        The averaging time in seconds.
    bandwidth : float
        The observed bandwidth, in Hz.
        For "full-band" sampled signals, this is half of the samplerate.

    Returns
    -------
    uncertainty : float
        Equals ``10 * log10((2 * mu ** 0.5 + 1) / (2 * mu ** 0.5 - 1))``
        for ``mu = averaging_time * bandwidth``.

    See Also
    --------
    required_averaging: Implements the opposite computation.

    Notes
    -----
    Start with gaussian white noise in the time domain,

    .. math:: x[n] \sim \mathcal{N}(0, \sigma^2).

    The DFT is computed

    .. math:: X[k] = \sum_{n=0}^{N-1} x[n] \exp(-2\pi i n k / N)

    using :math:`N` samples in the input signal.

    The trick is to write the DFT bins as real and complex, then they will be

    .. math:: X[k] \sim \mathcal{N}(0, N \sigma^2 / 2) + i \mathcal{N}(0, N \sigma^2 / 2)

    and rescale this to two standard normal distributions,

    .. math:: X[k] = Z_r[k] \sqrt{N/2} \sigma + i Z_i[k] \sqrt{N/2} \sigma = (Z_r[k] + i Z_i[k]) \sqrt{N/2} \sigma

    which then have

    .. math::
        Z_r[k] \sim \mathcal{N}(0, 1) \qquad Z_i[k] \sim \mathcal{N}(0, 1)

        Z_r^2[k] \sim \chi^2(1) \qquad Z_i^2[k] \sim \chi^2(1).

    We also need the chi-squared and Gamma relations (using shape :math:`k` and scale :math:`\theta` for Gamma distributions)

    .. math::
        \sum_l \chi^2(\nu_l) = \chi^2\left(\sum_l \nu_l\right)

        \chi^2(\nu) = \Gamma(\nu/2, 2)

        c\Gamma(k, \theta) = \Gamma(k, c\theta)

        \sum_l \Gamma(k_l, \theta) = \Gamma\left(\sum_l k_l, \theta\right)

    which directly lead to

    .. math::
        \sum_{l=1}^{L} c \chi^2(1) = \Gamma(L/2, 2c)

        \sum_{l=1}^{L} c \Gamma(k, \theta) = \Gamma(kL, c\theta)

    We have the PSD in each bin computed as

    .. math::
        PSD[k] &= (|X[k]|^2 + |X[-k]|^2) \frac{1}{N f_s} \\
        &= (\Re\{X[k]\}^2 + \Im\{X[k]\}^2 + \Re\{X[-k]\}^2 + \Im\{X[-k]\}^2) \frac{1}{N f_s} \\
        &= (Z_r^2[k] + Z_i^2[k] + Z_r^2[-k] + Z_i^2[-k]) \frac{N/2 \sigma^2}{N f_s} \\
        &= (Z_r^2[k] + Z_i^2[k] + Z_r^2[-k] + Z_i^2[-k]) \frac{\sigma^2}{2f_s} \\
        &= (Z_r^2[k] + Z_i^2[k]) \frac{\sigma^2}{f_s}

    where :math:`Z_r[k] = Z_r[-k]` and :math:`Z_i[k] = - Z_i[-k]` have been used in the last step.

    If we look at normalized PSD, defined as

    .. math:: NPSD[k] = PSD[k] \cdot \frac{f_s}{\sigma^2} = Z_r^2[k] + Z_i^2[k],

    it will have a distribution as

    .. math:: NPSD[k] \sim \chi^2(2) = \Gamma(1, 2).

    This then gives the distribution of the PSD as

    .. math:: PSD[k] \sim \Gamma(1, 2\sigma^2/f_s).


    When we compute the average PSD in a frequency band, we take the mean of a number of individual PSD bins.
    They are statistically independent samples of the same Gamma distribution (since we have white noise).
    The band level :math:`B[k_l, k_u]` is calculated as

    .. math:: B[k_l, k_u] = \frac{1}{k_u - k_l} \sum_{k=k_l}^{k_u - 1} PSD[k]

    with the distribution

    .. math::
        B[k_l, k_u] &\sim \frac{1}{k_u - k_l} \sum_{k=k_l}^{k_u - 1} \Gamma(1, 2\sigma^2/f_s)\\
        &= \Gamma\left(k_u - k_l, \frac{2\sigma^2}{f_s (k_u - k_l)}\right).

    Finally, taking :math:`L` averages of :math:`B[k_l, k_u]` gives us

    .. math::
        \tilde B[k_l, k_u] &\sim \frac{1}{L} \sum_{l=1}^{L} \Gamma\left(k_u - k_l, \frac{2\sigma^2}{f_s (k_u - k_l)}\right) \\
        &= \Gamma\left(L(k_u - k_l), \frac{2\sigma^2}{L f_s (k_u - k_l)}\right) \\
        &= \Gamma\left( \mu, \frac{2\sigma^2}{\mu f_s}\right)

    where we have defined the number of averaged values :math:`\mu = L(k_u - k_l)`, i.e., the number of time windows times the number of frequencies in a bin.
    Changing the number of time windows by a factor :math:`F` will change the number of frequency bins in a certain band by :math:`1/F`, so the number of averaged values remain constant.
    Taking the frequency band to be the entire spectrum gives us :math:`\mu = T f_s/2` values, where :math:`T` is the total sampling time.
    Looking back to the relation of summed and scaled chi-squared variables, we see that the first argument to the Gamma distribution is half of the number of independent chi-squared variables that are summed.
    This means that :math:`\mu = T f_s / 2` is consistent with that we have :math:`T f_s` independent samples.

    In the end, we want to know the mean and variance of this value, which we get from properties of the Gamma distribution

    .. math::
        E\left\{\Gamma(k, \theta)\right\} = k\theta

        \text{Var}\left\{\Gamma(k, \theta)\right\} = k \theta^2

    so we get the average band power density :math:`P=2\sigma^2/f_s` as expected (:math:`\sigma^2` power over :math:`f_s/2` bandwidth)
    and standard deviation :math:`\Delta P = \frac{2\sigma^2}{f_s\sqrt{\mu}} = P / \sqrt{\mu}`.

    For a frequency band covering :math:`[f_l, f_u]` we need to know how many bins fall in this band.
    For a time window of length :math:`T`, we have :math:`T f_s / 2` bins, so :math:`f[k] = k/T`, with :math:`k=0\ldots T f_s / 2`.
    Then

    .. math::
        k_l = f_l T \qquad k_u = f_u T \\

        \Rightarrow k_u - k_l = (f_u - f_l) T.

    Since the number of averaged values :math:`\mu` for multiple realizations of the same band average is independent of the number of bands used,
    we can always compute the standard deviation using the full length of the signal.

    Since the standard deviation is the mean times another value, the corresponding logarithmic standard deviation is independent of the logarithmic mean.
    Writing the power as :math:`P\pm\Delta P/2 = P(1 \pm \frac{1}{2\sqrt{\mu}})` (remembering :math:`\mu = T(f_u - f_l)`), we can compute the logarithmic spread as

    .. math::
        \Delta L &= 10\log(P + \Delta P/2) - 10\log(P - \Delta P/2) \\
        &= 10\log\left(P \left( 1 + \frac{1}{2\sqrt{\mu}}\right)\right) - 10\log\left(P \left( 1 - \frac{1}{2\sqrt{\mu}}\right)\right) \\
        &= 10\log\left(P \frac{2\sqrt{\mu} + 1}{2\sqrt{\mu}}\right) - 10\log\left(P \frac{2\sqrt{\mu} - 1}{2\sqrt{\mu}}\right) \\
        &= 10\log\left(\frac{2\sqrt{\mu} + 1}{2\sqrt{\mu} - 1}\right).

    For a spread of less than :math:`\Delta` dB, we get

    .. math::
        \Delta &\geq 10\log\left(\frac{2\sqrt{\mu} + 1}{2\sqrt{\mu} - 1}\right)

        \Rightarrow
        \mu &\geq \left(\frac{10^{\Delta/10} + 1}{10^{\Delta/10} - 1}\right)^2 / 4.

    """
    mu = averaging_time * bandwidth
    return 10 * np.log10((2 * mu**0.5 + 1) / (2 * mu**0.5 - 1))


def required_averaging(level_uncertainty, bandwidth):
    """Compute the required averaging time to obtain a certain uncertainty in levels.

    The level uncertainty here is derived from the mean and standard deviation of the power
    of sampled white gaussian noise. The uncertainty is the decibel difference between
    one half standard deviation above the mean and one half standard deviation below the mean.
    This is almost the same as the standard deviation of the decibel levels.

    Parameters
    ----------
    level_uncertainty : float
        The desired maximum uncertainty.
    bandwidth : float
        The observed bandwidth, in Hz.
        For "full-band" sampled signals, this is half of the samplerate.

    Returns
    -------
    averaging_time : float
        The minimum time to average.

    See Also
    --------
    level_uncertainty: Implements the opposite computation, has documentation of formulas and full derivation.

    """
    p = 10 ** (level_uncertainty / 10)
    mu = 0.25 * ((p + 1) / (p - 1)) ** 2
    return mu / bandwidth


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


class SpectrogramRollingComputation(_core.Roller):
    """Rolling computation of spectrograms.

    Parameters
    ----------
    spectrogram : Spectrogram
        The spectrogram configuration object containing settings such as frequency bounds,
        bands per decade, hybrid resolution, and FFT window type.
    time_data : TimeData
        The time-data wrapper to process.
    duration : float, optional
        The duration of the fft windows, in seconds.
    step : float, optional
        The step size between consecutive fft windows, in seconds.
    overlap : float, optional
        The amount of overlap between fft windows, as a fraction of the duration.
    """

    def __init__(self, spectrogram, time_data, duration=None, step=None, overlap=None):
        self.settings = _core.time_frame_settings(
            duration=duration,
            step=step,
            overlap=overlap,
            resolution=None if isinstance(spectrogram.hybrid_resolution, bool) else spectrogram.hybrid_resolution,
            signal_length=time_data.time_window.duration,
            samplerate=time_data.samplerate,
        )
        self.time_data = time_data
        self.spectrogram = spectrogram
        self.roller = self.time_data.rolling(
            duration=self.settings["duration"], step=self.settings["step"], overlap=self.settings["overlap"]
        )

        self.processing_axis = self.roller.dims.index("time")
        self.check_frequency_resolution()
        self.make_frequency_vectors()

    @property
    def min_frequency(self):
        """The lowest frequency to keep in the spectrogram."""
        return self.spectrogram.min_frequency or 0

    @property
    def max_frequency(self):
        """The highest frequency to keep in the spectrogram."""
        return min(self.spectrogram.max_frequency or np.inf, self.time_data.samplerate / 2)

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
        if not self.spectrogram.bands_per_decade:
            self.bands_per_decade = False
            self.hybrid_resolution = False
        else:
            self.bands_per_decade = self.spectrogram.bands_per_decade
            if not self.spectrogram.hybrid_resolution:
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
                if self.spectrogram.hybrid_resolution is True:
                    self.hybrid_resolution = 1 / self.settings["duration"]
                else:
                    self.hybrid_resolution = self.spectrogram.hybrid_resolution
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

        if self.spectrogram.max_frequency:
            upper_index = int(np.floor(self.spectrogram.max_frequency / self.time_data.samplerate * nfft))
        else:
            upper_index = None

        if self.spectrogram.min_frequency:
            lower_index = int(np.ceil(self.spectrogram.min_frequency / self.time_data.samplerate * nfft))
        else:
            lower_index = None

        if upper_index or lower_index:
            self.linear_slice = (slice(None),) * self.processing_axis + (slice(lower_index, upper_index),)
        else:
            self.linear_slice = False

        if self.linear_slice:
            self.frequency = self.linear_frequency[self.linear_slice]

        if self.spectrogram.bands_per_decade:
            log_band_scaling = 10 ** (0.5 / self.bands_per_decade)
            if self.hybrid_resolution:
                # The frequency at which the logspaced bands cover at least one linspaced band
                minimum_bandwidth_frequency = self.hybrid_resolution / (log_band_scaling - 1 / log_band_scaling)
                first_log_idx = int(
                    np.ceil(self.spectrogram.bands_per_decade * np.log10(minimum_bandwidth_frequency / 1e3))
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
        window = scipy.signal.windows.get_window(self.spectrogram.fft_window, self.settings["samples_per_frame"], False)
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


class SpectrogramData(_core.TimeFrequencyData):
    """Wrapper for spectrograms.

    This is a wrapper for storing spectrogram data.
    To calculate spectrograms, see `Spectrogram`.
    """

    def _figure_template(self, **kwargs):
        template = super()._figure_template(**kwargs)
        template.data.update(
            heatmap=[
                dict(
                    hovertemplate="%{x}<br>%{y:.5s}Hz<br>%{z}dB<extra></extra>",
                    colorbar_title_text="dB re 1μPa<sup>2</sup>/Hz",
                )
            ]
        )
        return template

    def plot(self, **kwargs):  # noqa: D102
        in_db = _core.dB(self)
        return super(SpectrogramData, in_db).plot(**kwargs)


@_core.compute_class(SpectrogramData)
class Spectrogram:
    """Calculates spectrograms, both linear and banded.

    If instantiated with a first positional-only argument of type `~uwacan.TimeData` or
    a `~recordings.AudioFileRecording`, that data will be processed into a spectrogram.
    If instantiated with any other first positional-only argument, that argument and all
    other arguments will be passed to `SpectrogramData`.
    If instantiated with no positional arguments, a callable processing instance is created.

    Parameters
    ----------
    bands_per_decade : float, optional
        The number of frequency bands per decade for logarithmic scaling.
    frame_duration : float
        The duration of each stft frame, in seconds.
    frame_step : float
        The time step between stft frames, in seconds.
    frame_overlap : float, default=0.5
        The overlap factor between stft frames. A negative value leaves
        gaps between frames.
    min_frequency : float
        The lowest frequency to include in the processing.
    max_frequency : float
        The highest frequency to include in the processing.
    hybrid_resolution : float
        A frequency resolution to aim for. Only used if ``frame_duration`` is not given
    scaling : str, default="power spectral density"
        The scaling to use for the output.

        - ``"power spectral density"`` scales the output as a power spectral density.
        - ``"power spectrum"`` scales the output as the total power in each band.

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

    @classmethod
    def _should_process(cls, data, *args, **kwargs):
        if isinstance(data, recordings.AudioFileRecording):
            return True
        if isinstance(data, _core.TimeData):
            return not isinstance(data, _core.TimeFrequencyData)
        if isinstance(data, xr.DataArray):
            if "time" in data.dims and "frequency" not in data.dims:
                return True
        return False

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

    def __call__(self, time_data, collect=True):
        """Process time data to spectrograms.

        Parameters
        ----------
        time_data : `~uwacan.TimeData` or `~uwacan.recordings.AudioFileRecording`
            The data to process.
        collect : bool, default=True
            Toggles collecting the results into a `Spectrogram`.

        Returns
        -------
        spectrogram : `Spectrogram` or `SpectrogramRollingComputation`

            - If ``collect=True``: the processed data as a `Spectrogram`.
            - If ``collect=False``: rolling windows of spectra, as `SpectrogramRollingComputation`.
        """
        if isinstance(time_data, type(self)):
            return time_data

        roller = SpectrogramRollingComputation(
            spectrogram=self,
            time_data=time_data,
            duration=self.frame_duration,
            step=self.frame_step,
            overlap=self.frame_overlap,
        )
        if not collect:
            return roller

        output = np.zeros((roller.num_frames,) + roller.shape)
        for idx, frame in enumerate(roller.numpy_frames()):
            output[idx] = frame
        output = SpectrogramData(
            output,
            frequency=roller.frequency,
            bandwidth=roller.bandwidth,
            samplerate=time_data.samplerate / roller.settings["sample_step"],
            start_time=time_data.time_window.start.add(seconds=roller.settings["duration"] / 2),
            coords=roller.coords,
            dims=("time",) + roller.dims,
            attrs=dict(
                frame_duration=roller.settings["duration"],
                frame_overlap=roller.settings["overlap"],
                frame_step=roller.settings["step"],
                bands_per_decade=roller.bands_per_decade,
                hybrid_resolution=roller.hybrid_resolution,
            ),
        )
        return output


class SpectralProbabilityData(_core.FrequencyData):
    """Wrapper to store spectral probability.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the frequency data.
    levels : array_like, optional
        The dB level represented by the bins. Mandatory if ``data`` is a `numpy.ndarray`.
    frequency : array_like, optional
        The frequencies corresponding to the data. Mandatory if ``data`` is a `numpy.ndarray`.
    bandwidth : array_like, optional
        The bandwidth of each frequency bin. Can be an array with per-frequency
        bandwidth or a single value valid for all frequencies.
    dims : str or [str], optional
        The dimensions of the data. Must have the same length as the number of dimensions in the data.
        Mandatory used for `numpy` inputs, not used for `xarray` inputs.
    coords : `xarray.DataArray.coords`
        Additional coordinates for this data.
    attrs : dict, optional
        Additional attributes to store with this data.

    """

    _coords_set_by_init = {"frequency", "levels"}

    def __init__(
        self,
        data,
        levels=None,
        binwidth=None,
        num_frames=None,
        scaling=None,
        averaging_time=None,
        frequency=None,
        bandwidth=None,
        dims=None,
        coords=None,
        attrs=None,
        **kwargs,
    ):
        super().__init__(
            data, dims=dims, coords=coords, attrs=attrs, frequency=frequency, bandwidth=bandwidth, **kwargs
        )
        if levels is not None:
            self.data.coords["levels"] = levels

        if binwidth is not None:
            self.data.attrs["binwidth"] = binwidth
        elif "binwidth" not in self.data.attrs:
            self.data.attrs["binwidth"] = np.mean(np.diff(self.levels))

        if scaling is not None:
            self.data.attrs["scaling"] = scaling

        if num_frames is not None:
            self.data.attrs["num_frames"] = num_frames

        if averaging_time is not None:
            self.data.attrs["averaging_time"] = averaging_time

    @property
    def levels(self):
        """The dB levels the probabilities are for."""
        return self.data.levels

    def rescale_probability(self, new_scale):
        """Rescale the probability data according to a new scaling method.

        This method adjusts the stored data to the specified ``new_scale``.
        The method supports conversions between three scales:

        - ``"counts"``: Represents raw event counts.
        - ``"probability"``: Represents probability, calculated as counts divided
          by the total number of frames.
        - ``"density"``: Represents density, calculated as probability divided
          by the bin width.

        The rescaling is based on metadata such as the number of frames and bin width.

        Parameters
        ----------
        new_scale : {"counts", "probability", "density"}
            The new scaling method to apply to the data.

        """
        if new_scale not in {"counts", "probability", "density"}:
            raise ValueError(f"Unknown probability scaling '{new_scale}'")

        current_scale = self.data.attrs["scaling"]
        if current_scale != new_scale:
            # We need to rescale
            # counts / num_frames = probability
            # probability / binwidth = density

            if "counts" in (new_scale, current_scale):
                if "num_frames" not in self.data.attrs:
                    raise ValueError(
                        f"Cannot rescale from '{current_scale} to {new_scale} without knowing the number of frames analyzed."
                    )
                num_frames = self.data.attrs["num_frames"]

            if "density" in (new_scale, current_scale):
                if "binwidth" not in self.data.attrs:
                    raise ValueError(
                        f"Cannot rescale from '{current_scale} to {new_scale} without knowing the binwidth."
                    )
                binwidth = self.data.attrs["binwidth"]

            if current_scale == "counts":
                if new_scale == "probability":
                    scale = 1 / num_frames
                elif new_scale == "density":
                    scale = 1 / (num_frames * binwidth)
            elif current_scale == "probability":
                if new_scale == "counts":
                    scale = num_frames
                elif new_scale == "density":
                    scale = 1 / binwidth
            elif current_scale == "density":
                if new_scale == "counts":
                    scale = num_frames * binwidth
                elif new_scale == "probability":
                    scale = binwidth

            self._data *= scale
            self._data.attrs["scaling"] = new_scale

    def _figure_template(self, **kwargs):
        template = super()._figure_template(**kwargs)
        template.layout.update(
            yaxis=dict(
                title_text="Level in dB. re 1μPa<sup>2</sup>/Hz",
            ),
        )
        template.data.update(
            heatmap=[
                dict(
                    colorscale="viridis",
                    colorbar_title_side="right",
                    hovertemplate="%{x:.5s}Hz<br>%{y}dBHz<br>%{z}<extra></extra>",
                )
            ]
        )
        return template

    def plot(self, logarithmic_probabilities=True, **kwargs):
        """Make a heatmap trace of this data.

        Parameters
        ----------
        logarithmic_probabilities : bool, default=True
            Toggles using a logarithmic colorscale for the probabilities.
        **kwargs : dict
            Keywords that will be passed to `~plotly.graph_objects.Heatmap`.
            Some useful keywords are:

            - ``colorscale`` chooses the colorscale, e.g., ``"viridis"``, ``"delta"``, ``"twilight"``.
            - ``zmin`` and ``zmax`` sets the color range.

        """
        import plotly.graph_objects as go

        if set(self.dims) != {"levels", "frequency"}:
            raise ValueError(
                f"Cannot make heatmap of spectral probability data with dimensions '{self.dims}'. "
                "Use the `.groupby(dim)` method to loop over extra dimensions."
            )

        data = self.data
        non_zero = (data != 0).any("frequency")
        min_level = data.levels[non_zero][0]
        max_level = data.levels[non_zero][-1]
        data = data.sel(levels=slice(min_level, max_level))

        hovertemplate = "%{x:.5s}Hz<br>%{y}dB<br>"
        if data.attrs["scaling"] == "probability":
            data = data * 100
            colorbar_title = "Probability in %"
            hovertemplate += "%{customdata:.5g}%"
        elif data.attrs["scaling"] == "density":
            data = data * 100
            colorbar_title = "Probability density in %/dB"
            hovertemplate += "%{customdata:.5g}%/dB"
        elif data.attrs["scaling"] == "counts":
            data = data
            colorbar_title = "Total occurrences"
            hovertemplate += "#%{customdata}"
        else:
            # This should never happen.
            raise ValueError(f"Unknown probability scaling '{data.attrs['scaling']}'")

        data = data.transpose("levels", "frequency")
        customdata = data

        if "zmax" in kwargs:
            p_max = kwargs["zmax"]
        else:
            p_max = data.max().item()

        if "zmin" in kwargs:
            p_min = kwargs["zmin"]
            if p_min == 0 and logarithmic_probabilities:
                # Cannot use a zero value to compute limits, since it maps to -inf
                p_min = data.where(data != 0).min().item()
                kwargs["zmin"] = p_min
        else:
            p_min = data.where(data != 0).min().item()

        if logarithmic_probabilities:
            p_max = np.log10(p_max)
            p_min = np.log10(p_min)

            if "zmax" in kwargs:
                kwargs["zmax"] = np.log10(kwargs["zmax"])
            if "zmin" in kwargs:
                kwargs["zmin"] = np.log10(kwargs["zmin"])

            with np.errstate(divide="ignore"):
                data = np.log10(data)

            # Making log-spaced ticks
            n_ticks = 5  # This is just a value to aim for. It usually works good to aim for 5 ticks.
            if np.ceil(p_max) - np.floor(p_min) + 1 >= n_ticks:
                # Ticks as 10^n, selecting every kth n as needed
                decimation = round((p_max - p_min + 1) / n_ticks)
                tick_max = int(np.ceil(p_max / decimation))
                tick_min = int(np.floor(p_min / decimation))
                tickvals = np.arange(tick_min, tick_max + 1) * decimation
            elif np.ceil(2 * (p_max - p_min)) + 1 >= n_ticks:
                # Ticks as [1, 3] * 10^n
                tick_max = int(np.ceil(p_max * 2))
                tick_min = int(np.floor(p_min * 2))
                tickvals = np.arange(tick_min, tick_max + 1) / 2
                # Round ticks so that 10**tick has one decimal
                tickvals = np.log10(np.round(10**tickvals / 10 ** np.floor(tickvals)) * 10 ** np.floor(tickvals))
            elif np.ceil(3 * (p_max - p_min)) + 1 >= n_ticks:
                # Ticks as [1, 2, 5] * 10^n
                tick_max = int(np.ceil(p_max * 3))
                tick_min = int(np.floor(p_min * 3))
                tickvals = np.arange(tick_min, tick_max + 1) / 3
                # Round ticks so that 10**tick has one decimal
                tickvals = np.log10(np.round(10**tickvals / 10 ** np.floor(tickvals)) * 10 ** np.floor(tickvals))
            else:
                # Linspaced ticks as [1, 2, 5] * n
                tick_min = 10**p_min
                tick_max = 10**p_max
                spacing = (tick_max - tick_min) / n_ticks
                # Round spacing to the nearest [1,2,5] * 10^n
                magnitude = np.floor(np.log10(spacing))
                mantissa = spacing / 10**magnitude
                if mantissa < 2:
                    mantissa = 1
                elif mantissa < 5:
                    mantissa = 2
                else:
                    mantissa = 5
                spacing = mantissa * 10**magnitude
                tick_min = int(np.floor(tick_min / spacing))
                tick_max = int(np.ceil(tick_max / spacing))
                tickvals = np.arange(tick_min, tick_max + 1) * spacing
                tickvals = np.log10(tickvals)

            ticktext = [f"{10.**tick:.3g}" for tick in tickvals]
        else:
            tickvals = ticktext = None

        trace = go.Heatmap(
            x=data.frequency,
            y=data.levels,
            z=data,
            customdata=customdata,
            hovertemplate=hovertemplate,
            colorbar_tickvals=tickvals,
            colorbar_ticktext=ticktext,
            colorbar_title_text=colorbar_title,
            zmax=p_max + (p_max - p_min) * 0.05,
            zmin=p_min - (p_max - p_min) * 0.05,
        )
        return trace.update(**kwargs)


@_core.compute_class(SpectralProbabilityData)
class SpectralProbability:
    """Compute spectral probability from time-series data.

    If instantiated with a first positional-only argument of type `~uwacan.TimeData` or
    a `~recordings.AudioFileRecording`, that data will be processed into a spectral probability.
    If instantiated with any other first positional-only argument, that argument and all
    other arguments will be passed to `SpectralProbabilityData`.
    If instantiated with no positional arguments, a callable processing instance is created.

    Parameters
    ----------
    filterbank : `Spectrogram`
        A pre-created instance used to filter the time data.
    binwidth : float, default=1
        The width of each level bin, in dB.
    min_level : float, default=0
        The lowest level to include in the processing, in dB.
    max_level : float, default=200
        The highest level to include in the processing, in dB.
    averaging_time : float or None
        The duration over which to average psd frames.
        This is used to average the output frames from the filterbank.
    scaling : str, default="density"
        The desired scaling of the probabilities for the level bins in each frequency band.
        Must be one of:

        - ``"counts"``: the number of frames with that level;
        - ``"probability"``: how often a certain level occurred, i.e., ``counts / num_frames``;
        - ``"density"``: the probability density at a certain level, i.e., ``probability / binwidth``.

        Note that the sum of counts (at a given frequency) is the total number of frames,
        the sum of probability is 1, while the integral of the density is 1.

    Notes
    -----
    To have representative values, each frequency bin needs sufficient averaging time.
    A coarse recommendation can be computed from the bandwidth and a desired uncertainty,
    see `required_averaging`. The uncertainty should ideally be smaller than the level binwidth.
    For computational efficiency, it is often faster to use a filterbank which has much shorter
    frames than this, even if frequency binning is used in the filterbank.
    This is why there is an option to have additional averaging of the PSD while computing
    the spectral probability, set using the ``averaging_time`` parameter.
    """

    @classmethod
    def _should_process(cls, data, *args, **kwargs):
        if isinstance(data, recordings.AudioFileRecording):
            return True
        if isinstance(data, _core.TimeData):
            return not isinstance(data, _core.TimeFrequencyData)
        if isinstance(data, xr.DataArray):
            if "time" in data.dims and "frequency" not in data.dims:
                return True
        return False

    def __init__(self, *, filterbank, binwidth=1, min_level=0, max_level=200, averaging_time=None, scaling="density"):
        self.binwidth = binwidth
        self.averaging_time = averaging_time
        self.filterbank = filterbank
        self.scaling = scaling

        max_level = int(np.ceil(max_level / self.binwidth))
        min_level = int(np.floor(min_level / self.binwidth))
        levels = np.arange(min_level, max_level + 1) * self.binwidth
        self.levels = levels

    def __call__(self, time_data):
        """Process time data to spectral probability.

        Parameters
        ----------
        time_data : `~uwacan.TimeData` or `~uwacan.recordings.AudioFileRecording`
            The data to process.

        Returns
        -------
        probabilites : `SpectralProbability`
            The processed data wrapped in this class.
        """
        levels = self.levels
        edges = levels[:-1] + 0.5 * self.binwidth
        edges = 10 ** (edges / 10)

        roller = self.filterbank(time_data, collect=False)
        if self.averaging_time:
            frames_to_average = int(np.ceil(self.averaging_time / roller.settings["step"]))
        else:
            frames_to_average = 1

        counts = np.zeros(roller.shape + (levels.size,))
        indices = np.indices(roller.shape)
        frames = roller.numpy_frames()
        for frame_idx in range(roller.num_frames // frames_to_average):
            frame = next(frames)
            # Running average
            for n in range(1, frames_to_average):
                frame += next(frames)
            if frames_to_average > 1:
                frame /= frames_to_average

            bin_index = np.digitize(frame, edges)
            counts[*indices, bin_index] += 1

        new = SpectralProbabilityData(
            counts,
            levels=levels,
            frequency=roller.frequency,
            dims=roller.dims + ("levels",),
            coords=roller.coords | {"time": _core.time_to_np(time_data.time_window.center)},
            scaling="counts",
            binwidth=self.binwidth,
            num_frames=frame_idx + 1,
            averaging_time=frames_to_average * roller.settings["step"],
        )
        new.rescale_probability(self.scaling)
        return new

    def analyze_segments(self, recording, segment_duration, filepath=None, status=None):
        """Compute spectral probability segments in a recording.

        Parameters
        ----------
        recording : `recordings.AudioFileRecording`
            The recording to process.
        segment_duration : float
            The duration of each time segment, in seconds.
        filepath : str, optional
            The file path where the results should be saved, if desired. If ``None`` (default), the results are
            concatenated in memory and returned. If provided, each segment result is saved to this file, and the
            concateneted results on disk are returned.
        status : bool or callable, optional
            Status reporting mechanism for the segments being processed. If ``True``, a default status message is
            printed to the console showing the time window being processed. If a callable function
            is provided, it will be called with the segment's ``time_window``.

        Returns
        -------
        SpectralProbabilityData
            If ``filepath`` is ``None``, returns a `SpectralProbabilityData` object created by concatenating the results
            of the analyzed segments along the time dimension. If ``filepath`` is provided, it loads and returns the
            results from the saved file, also as `SpectralProbabilityData`.

        """
        if not status:

            def status(time_window):
                pass
        elif status == True:

            def status(time_window):
                print(f"Computing segment {time_window.start.format_rfc3339()} to {time_window.stop.format_rfc3339()}")

        if filepath is None:
            results = []

        for segment in recording.rolling(duration=segment_duration, overlap=0):
            status(segment.time_window)
            segment = self(segment)
            if filepath is None:
                results.append(segment)
            else:
                segment.save(filepath, append_dim="time")

        if filepath is None:
            return _core.concatenate(results, dim="time")
        else:
            return SpectralProbabilityData.load(filepath)
