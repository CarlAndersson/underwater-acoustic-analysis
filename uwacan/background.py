from . import _core
import xarray as xr
import numpy as np


class Background(_core.FrequencyData):
    def __init__(self, data, snr_requirement=3, **kwargs):
        super().__init__(data, **kwargs)
        self.snr_requirement = snr_requirement

    def __call__(self, sensor_power):
        """Compensate a recorded frequency power spectral density.

        Notes
        -----
        We have requirements on the sensor information on the
        background data and the sensor data.
        1) If the background data has sensor information, the recorded
        power also needs to have sensor data.
        2) If the background data has no sensor information, it does
        not matter if the recorded power has sensor information.
        3) If both have sensor information, all the sensors in the
        recorded power has to exist in the background data.

        """
        background = self.data

        # if bg has sensors, data needs to have at least the same sensors
        if "sensor" in background.coords:
            if "sensor" not in sensor_power.coords:
                raise ValueError("Cannot apply sensor-wise background compensation to sensor-less recording")
            if "sensor" not in background.dims:
                # Single sensor in background, expand it to a dim so we can select from it
                background = background.expand_dims("sensor")
            # Pick the correct sensors from the background
            background = background.sel(sensor=sensor_power.coords["sensor"])

        if not sensor_power.frequency.equals(background.frequency):
            background_interp = background.interp(
                frequency=sensor_power.frequency,
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

        snr = _core.dB(sensor_power / background, power=True)
        compensated = xr.where(
            snr > self.snr_requirement,
            sensor_power - background,
            np.nan,
        )
        compensated.snr = snr
        return compensated
