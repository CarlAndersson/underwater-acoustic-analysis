"""Classes and methods to evaluate, model, and compensate for background noise in measurements.

.. autosummary::
    :toctree: generated

    Background

"""

from . import _core
import xarray as xr
import numpy as np


class Background(_core.FrequencyData):
    """A class for simple measured background noise.

    Parameters
    ----------
    data : `~uwacan.FrequencyData`
        The measured background noise, as a a power spectral density.
    snr_requirement : float
        The required SnR for a measurement to be valid.
        The compensation will output NaN for invalid
        data points.
    """

    def __init__(self, data, snr_requirement=3, **kwargs):
        super().__init__(data, **kwargs)
        self.snr_requirement = snr_requirement

    def __call__(self, sensor_power):
        """Compensate a recorded power spectral density.

        Parameters
        ----------
        sensor_power : `~uwacan.FrequencyData`
            The measured power spectral density to compensate.
            The background measurement will be interpolated
            to the required frequencies if needed.

        Notes
        -----
        We have requirements on the sensor information on the
        background data and the sensor data.

        1) If the background data has sensor information,
           the recorded power also needs to have sensor data.
        2) If the background data has no sensor information, it does
           not matter if the recorded power has sensor information.
        3) If both have sensor information, all the sensors in the
           recorded power has to exist in the background data.

        """
        background = self.data
        sensor_power_xr = sensor_power.data if isinstance(sensor_power, _core.xrwrap) else sensor_power

        # if bg has sensors, data needs to have at least the same sensors
        if "sensor" in background.coords:
            if "sensor" not in sensor_power_xr.coords:
                raise ValueError("Cannot apply sensor-wise background compensation to sensor-less recording")
            if "sensor" not in background.dims:
                # Single sensor in background, expand it to a dim so we can select from it
                background = background.expand_dims("sensor")
            # Pick the correct sensors from the background
            background = background.sel(sensor=sensor_power_xr.coords["sensor"])

        if not sensor_power_xr.frequency.equals(background.frequency):
            background_interp = background.interp(
                frequency=sensor_power_xr.frequency,
                method="linear",
            )
            # Extrapolating using the lowest and highest frequency in the background
            background_interp = xr.where(
                background_interp.frequency <= background.frequency[0],
                background.isel(frequency=0),
                background_interp,
            )
            background_interp = xr.where(
                background_interp.frequency >= background.frequency[-1],
                background.isel(frequency=-1),
                background_interp,
            )
            background = background_interp

        snr = _core.dB(sensor_power_xr / background, power=True)
        compensated = xr.where(
            snr > self.snr_requirement,
            sensor_power_xr - background,
            np.nan,
        )
        compensated = sensor_power.from_dataset(compensated)
        compensated.snr = snr
        return compensated
