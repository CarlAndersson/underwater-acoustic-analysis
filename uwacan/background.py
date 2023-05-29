from . import analysis
import xarray as xr
import numpy as np


class MeasuredDecidecadeBackgroundCompensation:
    def __init__(self, segments, frequency_range=(10, 50000), snr_requirement=3):
        filterbank = analysis.DecidecadeFilterbank(frequency_range=frequency_range)
        self.snr_requirement = snr_requirement

        segment_powers = []
        for segment in segments:
            signal = segment.time_data
            spectrogram = analysis.spectrogram(signal, 1)
            power = filterbank(spectrogram)
            power = power.mean(dim='time').assign_coords(segment=segment.sampling.window.center)
            segment_powers.append(power)
        self.segment_powers = xr.concat(segment_powers, dim='segment')
        self.average_power = self.segment_powers.pipe(np.log).mean(dim='segment').pipe(np.exp)

    def __call__(self, recorded_power, **kwargs):
        bad_snr = recorded_power / self.average_power < 10**(self.snr_requirement / 10)
        return xr.where(bad_snr, np.nan, recorded_power - self.average_power)
