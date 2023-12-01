from . import analysis, positional
import xarray as xr
import numpy as np


class MeasuredDecidecadeBackgroundCompensation:
    def __init__(self, segments, filterbank=None, snr_requirement=3):
        if filterbank is None:
            filterbank = analysis.DecidecadeFilterbank(window_duration=1, lower_bound=10, upper_bound=50_000)
        self.snr_requirement = snr_requirement

        segment_powers = []
        for segment in segments:
            power = filterbank(segment.time_data)
            power = power.mean(dim='time').assign_coords(segment=positional._datetime_to_np(segment.sampling.window.center))
            segment_powers.append(power)
        self.segment_powers = xr.concat(segment_powers, dim='segment')
        self.average_power = self.segment_powers.pipe(np.log).mean(dim='segment').pipe(np.exp)

    def __call__(self, recorded_power, **kwargs):
        bad_snr = recorded_power / self.average_power < 10**(self.snr_requirement / 10)
        return xr.where(bad_snr, np.nan, recorded_power - self.average_power)
