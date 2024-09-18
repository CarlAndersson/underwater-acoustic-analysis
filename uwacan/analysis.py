"""Various analysis protocols and standards for recorded underwater noise from ships.

.. currentmodule:: uwacan.analysis

Core processing and analysis
----------------------------
.. autosummary::
    :toctree: generated

    ShipLevel

Helper functions and conversions
--------------------------------
.. autosummary::
    :toctree: generated

    convert_to_radiated_noise

"""

import numpy as np
from . import _core, propagation, spectral
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
            filterbank = spectral.Spectrogram(bands_per_decade=10, min_frequency=20, max_frequency=20_000, frame_step=1)

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
            filterbank = spectral.Spectrogram(bands_per_decade=10, min_frequency=20, max_frequency=20_000, frame_step=1)

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
                transit_results["snr"] = compensated_power.snr.data
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
