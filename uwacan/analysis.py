"""Various analysis protocols and standards for recorded underwater noise from ships.

.. currentmodule:: uwacan.analysis

Core processing and analysis
----------------------------
.. autosummary::
    :toctree: generated

    Spectrogram
    SpectralProbability
    ShipLevel

Helper functions and conversions
--------------------------------
.. autosummary::
    :toctree: generated

    convert_to_radiated_noise
    level_uncertainty
    required_averaging

"""

import numpy as np
from . import (
    _core,
    _filterbank,
    propagation,
)
import xarray as xr


class ShipLevel(_core.DatasetWrap):
    """Calculates and stores measured ship levels.

    This class has all functionality to analyze ship transits and
    post-process the resulting radiated noise levels.
    The analysis methods are all implemented as classmethods with the
    ``analyze_transits`` prefix.

    Parameters
    ----------
    data : `xarray.Dataset`
        The dataset with measurement results.
        This dataset must have a "source_power" variable.
    """

    @classmethod
    def analyze_transits(
        cls,
        *transits,
        filterbank=None,
        propagation_model=None,
        background_noise=None,
        transit_min_angle=None,
        transit_min_duration=None,
        transit_min_length=None,
    ):
        """Analyze ship transits to estimate source power and related metrics.

        Parameters
        ----------
        *transits : Transit objects
            One or more `Transit` objects to be analyzed.
        filterbank : callable, optional
            A callable that applies a filterbank to the time data of the recording. If not provided, defaults to
            `~spectral.Spectrogram` with 10 bands per decade between 20 Hz and 20 kHz, and a frame step of 1.
            The callable should have the signature::

                f(time_data: uwacan.TimeData) -> uwacan.TimeFrequencyData

        propagation_model : callable or `~uwacan.propagation.PropagationModel`, optional
            A callable that compensates for the propagation effects on the received power. If not provided, defaults
            to a `~uwacan.propagation.MlogR` propagation model with ``m=20``.
            The callable should have the signature::

                propagation_model(
                    received_power: uwacan.FrequencyData,
                    receiver: uwacan.Position,
                    source: uwacan.Track
                ) -> uwacan.FrequencyData

            with the frequency data optionally also having a time dimension.
        background_noise : callable, optional
            A callable that models the background noise.
            The callable should have the signature::

                f(received_power: uwacan.FrequencyData) -> uwacan.FrequencyData

            If not provided, defaults to a no-op function that returns the input ``received_power``.
            A suitable callable can be created using the `uwacan.background.Background` class.
        transit_min_angle : float, optional
            Minimum angle for segment selection during transit analysis, in degrees.
            The segment analyzed will cover at least this aspect angle on each side of the CPA.
            E.g., ``transit_min_angle=30`` means the segment covers from -30° to +30°.
        transit_min_duration : float, optional
            Minimum duration for segment selection during transit analysis, in seconds.
            The segment analyzed will cover at least this duration in total.
        transit_min_length : float, optional
            Minimum length for segment selection during transit analysis, in meters.
            The segment analyzed will cover at least this length in total.

        Returns
        -------
        ship_levels : `ShipLevels`
            An instance of the class containing the analysis results for each transit, including source power,
            latitude, longitude, transit time, and optionally signal-to-noise ratio (SNR).

        Notes
        -----
        This method processes each transit individually by:
            1. Determining the closest point of approach (CPA) time.
            2. Optionally selecting a segment around CPA.
            3. Applying the filterbank to the time data of the recording.
            4. Compensating for background noise and propagation effects.

        The method returns a concatenated dataset containing the results for all provided transits.
        The core dimension for each transit is "segment", which indicates the time segments used
        in the filterbank.
        """
        if filterbank is None:
            filterbank = _filterbank.Filterbank(bands_per_decade=10, min_frequency=20, max_frequency=20_000, frame_step=1)

        if background_noise is None:

            def background_noise(received_power, **kwargs):
                return received_power

        if propagation_model is None:
            propagation_model = propagation.MlogR(m=20)

        if isinstance(propagation_model, propagation.PropagationModel):
            propagation_model = propagation_model.compensate_propagation

        results = []
        for transit in transits:
            if (transit_min_angle, transit_min_duration, transit_min_length) == (None, None, None):
                cpa_time = transit.track.closest_point(transit.recording.sensor)["time"].data
            else:
                segment = transit.track.aspect_segments(
                    reference=transit.recording.sensor,
                    angles=0,
                    segment_min_duration=transit_min_duration,
                    segment_min_angle=transit_min_angle * 2,
                    segment_min_length=transit_min_length,
                )
                cpa_time = segment.time.sel(edge="center").data
                transit = transit.subwindow(segment)

            direction = transit.track.average_course("eight")
            time_data = transit.recording.time_data()
            received_power = filterbank(time_data)

            received_power = background_noise(received_power)
            track = transit.track.resample(received_power.time)
            source_power = propagation_model(
                received_power=received_power, receiver=transit.recording.sensor, source=track
            )
            transit_time = (received_power.data["time"] - cpa_time) / np.timedelta64(1, "s")
            closest_to_cpa = np.abs(transit_time).argmin("time").item()
            segment = xr.DataArray(
                np.arange(transit_time.time.size) - closest_to_cpa, coords={"time": received_power.time}
            )
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
                ),
            )
            transit_results["received_power"] = received_power.data
            if hasattr(received_power, "snr"):
                transit_results["snr"] = received_power.snr
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
        """Analyze ship transits in constant angle segments to estimate source power and related metrics.

        Parameters
        ----------
        *transits : Transit objects
            One or more `Transit` objects to be analyzed.
        filterbank : callable, optional
            A callable that applies a filterbank to the time data of the recording. If not provided, defaults to
            `NthDecadeSpectrogram` with 10 bands per decade between 20 Hz and 20 kHz, and a frame step of 1.
            The callable should have the signature::

                f(time_data: uwacan.TimeData) -> uwacan.TimeFrequencyData

        propagation_model : callable or `~uwacan.propagation.PropagationModel`, optional
            A callable that compensates for the propagation effects on the received power. If not provided, defaults
            to a `~uwacan.propagation.MlogR` propagation model with ``m=20``.
            The callable should have the signature::

                propagation_model(
                    received_power: uwacan.FrequencyData,
                    receiver: uwacan.Position,
                    source: uwacan.Track
                ) -> uwacan.FrequencyData

            with the frequency data optionally also having a time dimension.
        background_noise : callable, optional
            A callable that models the background noise.
            The callable should have the signature::

                f(received_power: uwacan.FrequencyData) -> uwacan.FrequencyData

            If not provided, defaults to a no-op function that returns the input ``received_power``.
            A suitable callable can be created using the `uwacan.background.Background` class.
        aspect_angles : array_like
            The angles where to center each segment, in degrees.
            Defaults to each 5° from -45° to 45°.
        segment_min_angle : float, optional
            Minimum angle width for the segments, in degrees.
        segment_min_duration : float, optional
            Minimum duration for the segments, in seconds.
        segment_min_length : float, optional
            Minimum length for the segments, in meters.

        Returns
        -------
        ship_levels : `ShipLevels`
            An instance of the class containing the analysis results for each transit, including source power,
            latitude, longitude, transit time, and optionally signal-to-noise ratio (SNR).

        Notes
        -----
        This method processes each transit individually by:

        1. Determining the closest point of approach (CPA) time.
        2. Finding the segments centered at each aspect_angle.
           See `uwacan.Track.aspect_segments` for more details on how the segments are computed.
        3. Applying the filterbank to the time data of the recording.
        4. Averaging the received sound power within each segment.
        5. Compensating for background noise and propagation effects in each segment.

        The method returns a concatenated dataset containing the results for all provided transits.
        The core dimension for each transit is "segment", which indicates the aspect angles specified.
        """
        if filterbank is None:
            filterbank = _filterbank.Filterbank(bands_per_decade=10, min_frequency=20, max_frequency=20_000, frame_step=1)

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

            direction = transit.track.average_course("eight")
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
            source_power = propagation_model(
                received_power=segment_powers, receiver=transit.recording.sensor, source=track
            )
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
                ),
            )
            if hasattr(compensated_power, "snr"):
                transit_results["snr"] = compensated_power.snr
            results.append(transit_results)
        results = xr.concat(results, "transit")
        results.coords["transit"] = np.arange(results.sizes["transit"]) + 1
        return cls(results)

    @property
    def source_power(self):
        """The source power of the transits."""
        return _core.FrequencyData(self._data["source_power"])

    @property
    def source_level(self):
        """The source level of the transits."""
        return _core.dB(self.source_power, power=True)

    @property
    def received_power(self):
        """The received power during the transits."""
        return _core.FrequencyData(self._data["received_power"])

    @property
    def received_level(self):
        """The received level during the transits."""
        return _core.dB(self.received_power, power=True)

    def power_average(self, dim=..., **kwargs):
        """Power-wise average of data.

        This calculates the power average of the ship levels
        over some dimensions, and the linear average of non-power
        quantities. SnR is always averaged on a level basis.

        See `xarray.DataArray.mean` for more details.
        """
        return type(self)(self._data.mean(dim, **kwargs))

    def level_average(self, dims, **kwargs):
        """Level-wise average of data.

        This calculates the level average of the ship levels
        over some dimensions, and the linear average of non-power
        quantities. SnR is always averaged on a level basis.

        See `xarray.DataArray.mean` for more details.
        """
        source_power = 10 ** (self.source_level.mean(dims, **kwargs) / 10)
        received_power = 10 ** (self.received_level.mean(dims, **kwargs) / 10)

        others = self._data.drop_vars(["source_power", "received_power"])
        others = others.mean(dims, **kwargs)
        data = others.merge(
            {
                "source_power": source_power.data,
                "received_power": received_power.data,
            }
        )
        return type(self)(data)

    @property
    def snr(self):
        """The signal to noise ratio in the measurement, in dB."""
        return _core.FrequencyData(self._data["snr"])

    def meets_snr_threshold(self, threshold):
        """Check where the measurement meets a specific SnR threshold.

        This thresholds the SnR to a specific level and returns 1 where
        the threshold is met, 0 where the threshold is not met, and NaN
        where there is no SnR information (typically segments that were
        not measured).

        Parameters
        ----------
        threshold : float
            The threshold to compare against, in dB.

        Returns
        -------
        meets_threshold : `~uwacan.FrequencyData`
            Whether the SnR meets the threshold or not.

        Notes
        -----
        This is useful to compute statistics of how often the measurement
        meets a SnR threshold. By taking the average of the output from here,
        we get a measure of how often we meet that threshold. By taking the
        average before or after we compare to the threshold, we can control
        on what granularity we measure. E.g., for a measurement with multiple
        sensors, segments, and transits, we can get the finest granularity::

            ship_levels.meets_snr_threshold(3).mean(["sensor", "segment", "transit"]) * 100

        or we can choose to only look at how many of the transits meet the SnR
        threshold after averaging over sensors and segments::

            ship_levels.power_average(["sensor", "segment"]).meets_snr_threshold(3).mean("transit") * 100

        We multiply both by 100 to get the value in percent.
        """
        snr = self._data["snr"]
        meets_threshold = xr.where(
            snr.isnull(),
            np.nan,
            snr > threshold,
        )
        return _core.FrequencyData(meets_threshold)


def convert_to_radiated_noise(source, source_depth, mode="iso", power=False):
    r"""Convert a monopole source level to a radiated noise level.

    Parameters
    ----------
    source : `~_core.FrequencyData`
        The source level or source power.
    source_depth : float
        The source depth to use for the conversion.
    mode : str, default="iso"
        Which type of conversion to perform
    power : bool, default=False
        If the input and output are powers or levels.

    Notes
    -----
    There are several conversion formulas implemented in this function.
    They are described below with a conversion factor :math:`F(η)` such
    as

    .. math::

        P_{RNL} = P_{MSL} F(η) \\
        η = kd

    with :math:`k` being the wavenumber and :math:`d` being the
    source depth.

    The most commonly used one is the "iso" mode:

    .. math::

        F = \frac{14 η^2 + 2 η^4}{14 + 2 η^2 + η^4}

    This is designed to convert a monopole source level to radiated
    noise levels measured at deep waters with hydrophone depression
    angles of 15°, 30°, and 45°. This has a high-frequency compensation
    of 2 (+3 dB) and a low-frequency compensation of η^2 (+20 dB/decade).

    An alternative is the "average farfield" which averages all
    depression angles

    .. math::

        F = 2 / (1 + 1 / η^2)

    This has a high-frequency compensation of 2 (+3 dB) and a low frequency
    compensation of 2η^2 (+3 dB + 20 dB/decade).

    A third one is "isomatch", which averages up to a depression angle of θ=54.3°,
    (measured in radians in the formulas below)

    .. math::

        F = 2 / (1 + 1 / G)\\
        G = η^2 (θ - \sin(θ) \cos(θ)) / θ\\

    This has the same asymptotical compensations as the "iso" method:
    high-frequency of 2 (+3 dB) and low-frequency of η^2 (+20 dB/decade).

    """
    if mode is None or not mode:
        return source
    kd = 2 * np.pi * source.frequency / 1500 * source_depth
    mode = mode.lower()
    if mode == "iso":
        compensation = (14 * kd**2 + 2 * kd**4) / (14 + 2 * kd**2 + kd**4)
    elif mode == "average farfield":
        compensation = 1 / (1 / 2 + 1 / (2 * kd**2))
    elif mode == "isomatch":
        truncation_angle = np.radians(54.3)
        lf_comp = (
            2 * kd**2 * (truncation_angle - np.sin(truncation_angle) * np.cos(truncation_angle)) / truncation_angle
        )
        compensation = 1 / (1 / 2 + 1 / lf_comp)
    elif mode == "none":
        compensation = 1
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    if power:
        return source * compensation
    else:
        return source + 10 * np.log10(compensation)


class Spectrogram(_core.TimeFrequencyData):
    """Handling of spectrogram data, both linear and banded.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the time-frequency data.
    start_time : time_like, optional
        The start time for the first sample in the signal.
        This should ideally be a proper time type, but it will be parsed if it is a string.
        Defaults to "now" if not given.
    samplerate : float, optional
        The samplerate for this data, in Hz. This is not the samplerate of the underlying time signal,
        but time steps of the time axis on the time-frequency data.
        If the `data` is a `numpy.ndarray`, this has to be given.
        If the `data` is a `xarray.DataArray` which already has a time coordinate,
        this can be omitted.
    frequency : array_like, optional
        The frequencies corresponding to the data. Mandatory if `data` is a `numpy.ndarray`.
    bandwidth : array_like, optional
        The bandwidth of each data point. Can be an array with per-frequency
        bandwidth or a single value valid for all frequencies.
    dims : str or [str], optional
        The dimensions of the data. Must have the same length as the number of dimensions in the data.
        Mandatory used for `numpy` inputs, not used for `xarray` inputs.
    coords : `xarray.DataArray.coords`
        Additional coordinates for this data.
    attrs : dict, optional
        Additional attributes to store with this data.
    """

    @classmethod
    def analyze_timedata(
        cls,
        time_data,
        *,
        bands_per_decade=None,
        frame_step=None,
        frame_duration=None,
        frame_overlap=0.5,
        min_frequency=None,
        max_frequency=None,
        hybrid_resolution=None,
        fft_window="hann",
        filepath=None,
        batch_size=None,
        status=None,
    ):
        """Compute a spectrogram from time data.

        Parameters
        ----------
        time_data : `~uwacan.TimeData` or `~uwacan.recordings.AudioFileRecording`
            The data to process.
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
            or an array-like sequence of window coefficients.
        filepath : str, optional
            If provided, save results to this file path. If None, return results in memory.
        batch_size : int, optional
            Number of frames to process before saving to file. If None and filepath is provided,
            a suitable batch size will be computed based on available memory.
        status : bool or callable, optional
            Status reporting mechanism for the segments being processed. If ``True``, a default status message is
            printed to the console showing the time window being processed. If a callable function
            is provided, it will be called with the segment's ``time_window``.

        Returns
        -------
        processed_data : `~uwacan.analysis.Spectrogram`
            The processed spectrogram data.
        """
        if not status:
            def status(time_window):
                pass
        elif status == True:
            def status(time_window):
                print(f"\rComputed segment {time_window.start.format_rfc3339()} to {time_window.stop.format_rfc3339()}", end="")

        filterbank = _filterbank.Filterbank(
            bands_per_decade=bands_per_decade,
            frame_step=frame_step,
            frame_duration=frame_duration,
            frame_overlap=frame_overlap,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            hybrid_resolution=hybrid_resolution,
            fft_window=fft_window,
        )

        if filepath is None:
            # Process all frames at once and return in memory
            processed = filterbank(time_data)
            return cls(processed)

        roller = filterbank.rolling(time_data)
        # If no batch size provided, compute a suitable one based on memory usage
        if batch_size is None:
            # Estimate memory needed per frame (in bytes)
            memory_per_frame = 8 * np.prod(roller.shape)
            target_memory = 128 * 1024 * 1024  # 128MB per batch as default
            batch_size = max(1, int(target_memory / memory_per_frame))

        # Process and save batches one at a time
        for batch in roller.batches(batch_size):
            # Report status after each batch is processed
            status(batch.time_window)
            batch = cls(batch)  # Convert to Spectrogram to save with correct class name
            batch.save(filepath, append_dim="time")

        return cls.load(filepath)

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
        return super(Spectrogram, in_db).plot(**kwargs)


class SpectralProbability(_core.FrequencyData):
    """Handles spectral probability data.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the frequency data.
    levels : array_like, optional
        The dB level represented by the bins. Mandatory if ``data`` is a `numpy.ndarray`.
    binwidth : float, optional
        The width of the bins. Computed from the levels if not given.
    frequency : array_like, optional
        The frequencies corresponding to the data. Mandatory if ``data`` is a `numpy.ndarray`.
    bandwidth : array_like, optional
        The bandwidth of each frequency bin. Can be an array with per-frequency
        bandwidth or a single value valid for all frequencies.
    scaling : str, default="density"
            The scaling of the probabilities for the level bins in each frequency band.
            Must be one of:

            - ``"counts"``: the number of frames with that level;
            - ``"probability"``: how often a certain level occurred, i.e., ``counts / num_frames``;
            - ``"density"``: the probability density at a certain level, i.e., ``probability / binwidth``.

            Note that the sum of counts (at a given frequency) is the total number of frames,
            the sum of probability is 1, and the integral of the density is 1.
    num_frames : int
        The number of spectra used to compute this spectral probability.
    averaging_time : float
        The duration from which each spectrum was averaged over, in seconds.
    dims : str or [str], optional
        The dimensions of the data. Must have the same length as the number of dimensions in the data.
        Mandatory used for `numpy` inputs, not used for `xarray` inputs.
    coords : `xarray.DataArray.coords`
        Additional coordinates for this data.
    attrs : dict, optional
        Additional attributes to store with this data.

    """

    _coords_set_by_init = {"frequency", "levels"}

    @classmethod
    def analyze_timedata(
        cls,
        data,
        *,
        filterbank,
        binwidth=1,
        min_level=0,
        max_level=200,
        averaging_time=None,
        scaling="density",
    ):
        """Compute spectral probability from time data.

        Parameters
        ----------
        data : `~uwacan.TimeData` or `~uwacan.recordings.AudioFileRecording`
            The data to process.
        filterbank : `~uwacan.Filterbank`
            A pre-created instance used to filter the time data. Includes specification
            of the frequency bands to use.
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
            the sum of probability is 1, and the integral of the density is 1.

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
        max_level = int(np.ceil(max_level / binwidth))
        min_level = int(np.floor(min_level / binwidth))
        levels = np.arange(min_level, max_level + 1) * binwidth

        edges = levels[:-1] + 0.5 * binwidth
        edges = 10 ** (edges / 10)

        roller = filterbank.rolling(data)
        if averaging_time:
            frames_to_average = int(np.ceil(averaging_time / roller.settings["step"]))
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

        new = cls(
            counts,
            levels=levels,
            frequency=roller.frequency,
            dims=roller.dims + ("levels",),
            coords=roller.coords | {"time": _core.time_to_np(data.time_window.center)},
            scaling="counts",
            binwidth=binwidth,
            num_frames=frame_idx + 1,
            averaging_time=frames_to_average * roller.settings["step"],
        )
        return new.with_probability_scale(scaling)

    def __init__(
        self,
        data,
        levels=None,
        binwidth=None,
        frequency=None,
        bandwidth=None,
        scaling=None,
        num_frames=None,
        averaging_time=None,
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

    def with_probability_scale(self, new_scale):
        """Rescale the probability data according to a new scaling method.

        This method scales the data to the specified ``new_scale``.
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

        Returns
        -------
        scaled : `SpectralProbability`
            Data with the new scaling.

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

            new = self * scale
            new.data.attrs["scaling"] = new_scale
            return new

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


class SpectralProbabilitySeries(SpectralProbability, _core.TimeData):
    """Handling of spectral probability series data.

    The spectral probability series is a series of specral probabilites
    computed from shorter segments of time data, i.e., a time-series
    of spectral probabilites.
    As such, it describes a time-varying spectral probability, e.g.,
    the probability density function (histogram) at a certain frequency
    band as a function of sound pressure level, but varying over time.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the data.
    time : array_like, optional
        A `numpy.ndarray` with ``dtype=datetime64[ns]`` containing time stamps for each spectral probability.
    start_time : time_like, optional
        The start time for the first spectral probability.
        This should ideally be a proper time type, but it will be parsed if it is a string.
        Defaults to "now" if not given.
    samplerate : float, optional
        The samplerate for this data, in Hz. This refers to the rate of the spectral probabilities,
        not the time signal used to compute the spectra.
        If the ``data`` is a `numpy.ndarray`, this has to be given.
        If the ``data`` is a `xarray.DataArray` which already has a time coordinate,
        this can be omitted.
    levels : array_like, optional
        The dB level represented by the bins. Mandatory if ``data`` is a `numpy.ndarray`.
    binwidth : float, optional
        The width of the bins. Computed from the levels if not given.
    frequency : array_like, optional
        The frequencies corresponding to the data. Mandatory if ``data`` is a `numpy.ndarray`.
    bandwidth : array_like, optional
        The bandwidth of each frequency bin. Can be an array with per-frequency
        bandwidth or a single value valid for all frequencies.
     scaling : str, default="density"
            The scaling of the probabilities for the level bins in each frequency band.
            Must be one of:

            - ``"counts"``: the number of frames with that level;
            - ``"probability"``: how often a certain level occurred, i.e., ``counts / num_frames``;
            - ``"density"``: the probability density at a certain level, i.e., ``probability / binwidth``.

            Note that the sum of counts (at a given frequency) is the total number of frames,
            the sum of probability is 1, and the integral of the density is 1.
    num_frames : int
        The number of spectra used to compute each spectral probability.
    averaging_time : float
        The duration from which each spectrum was averaged over. This is separate from the duration that each specral probability covers.
    dims : str or [str], optional
        The dimensions of the data. Must have the same length as the number of dimensions in the data.
        Mandatory used for `numpy` inputs, not used for `xarray` inputs.
    coords : `xarray.DataArray.coords`
        Additional coordinates for this data.
    attrs : dict, optional
        Additional attributes to store with this data
    """

    _coords_set_by_init = {"frequency", "levels", "time"}

    @classmethod
    def analyze_timedata(
        cls,
        data,
        *,
        filterbank,
        binwidth=1,
        min_level=0,
        max_level=200,
        averaging_time=None,
        scaling="density",
        segment_duration=None,
        segment_overlap=None,
        segment_step=None,
        filepath=None,
        status=None,
    ):
        """Compute spectral probability segments in a recording.

        Parameters
        ----------
        data : `~uwacan.TimeData` or `recordings.AudioFileRecording`
            The recording to process.
        filterbank : `~uwacan.Filterbank`
            A pre-created instance used to filter the time data. Includes specification
            of the frequency bands to use
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
            the sum of probability is 1, and the integral of the density is 1.
        segment_duration : float
            The duration of each time segment used to compute one spectral probability.
        filepath : str, optional
            The file path where the results should be saved, if desired. If ``None`` (default), the results are
            concatenated in memory and returned. If provided, each segment result is saved to this file, and the
            concateneted results on disk are returned.
        status : bool or callable, optional
            Status reporting mechanism for the segments being processed. If ``True``, a default status message is
            printed to the console showing the time window being processed. If a callable function
            is provided, it will be called with the segment's ``time_window``.

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
        if not status:

            def status(time_window):
                pass
        elif status == True:

            def status(time_window):
                print(f"\rComputing segment {time_window.start.format_rfc3339()} to {time_window.stop.format_rfc3339()}", end="")

        if filepath is None:
            results = []

        for segment in data.rolling(duration=segment_duration, overlap=segment_overlap, step=segment_step):
            status(segment.time_window)
            segment = SpectralProbability.analyze_timedata(
                segment,
                filterbank=filterbank,
                binwidth=binwidth,
                min_level=min_level,
                max_level=max_level,
                averaging_time=averaging_time,
                scaling=scaling,
            )
            if filepath is None:
                results.append(segment)
            else:
                segment.save(filepath, append_dim="time")

        if filepath is None:
            return _core.concatenate(results, dim="time", cls=cls)
        else:
            return cls.load(filepath)

    def __init__(
        self,
        data,
        time=None,
        samplerate=None,
        start_time=None,
        levels=None,
        binwidth=None,
        frequency=None,
        bandwidth=None,
        scaling=None,
        num_frames=None,
        averaging_time=None,
        dims=None,
        coords=None,
        attrs=None,
        **kwargs,
    ):
        super().__init__(
            data, dims=dims, coords=coords, attrs=attrs,
            frequency=frequency, bandwidth=bandwidth,
            time=time, start_time=start_time, samplerate=samplerate,
            levels=levels, binwidth=binwidth, num_frames=num_frames, scaling=scaling, averaging_time=averaging_time,
            **kwargs
        )

    def plot(self, **kwargs):
        raise ValueError(f"Cannot directly plot SpectralProbabilitySeries! You likely want to first average over time.")

    @classmethod
    def from_dataset(cls, data, **kwargs):
        if "time" not in data.dims:
            return SpectralProbability.from_dataset(data, **kwargs)
        return super().from_dataset(data, **kwargs)


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
