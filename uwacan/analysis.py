"""Various analysis protocols and standards for recorded underwater noise from ships."""

import numpy as np


class BureauVeritasSourceSpectrum:
    aspect_window_resolution = 5
    aspect_window_range = (-45, 45)

    def __init__(
        self,
        ship_passbys,
        transmission_model,
        background_noise,
        frequency_range,
        bandtypes='thirds',
    ):
        self.ship_passbys = ship_passbys
        self.transmission_model = transmission_model
        self.background_noise = background_noise
        self.frequency_range = frequency_range

        if isinstance(bandtypes, str):
            bandtypes = [bandtypes]
        self.bandtypes = bandtypes

        self.passby_powers = []

    def process_passby(self, passby):
        # Get aspect angles between ship position and hydrophone position
        time_windows = passby.ship_track.aspect_windows(
            reference_point=passby.hydrophone.position,
            resolution=self.aspect_window_resolution,
            range=self.aspect_window_range,
            length=self.aspect_window_length
        )

        # Calculate the spectrogram of the passby
        spectrogram = passby.hydrophone[time_windows[0].start_time:time_windows[-1].stop_time].spectrogram()
        source_power = {}

        for band in self.bandtypes:
            # Calculate the received power in the frequency bands at each time instance
            received_power_bands = self.filterbank(band)(spectrogram)
            # Average the power over time for each analysis window, for each hydrophone
            received_power_windows = np.stack([
                np.mean(received_power_bands[window.start_time:window.stop_time], axis='t_psd')
                for window in time_windows],
                axis='t_aspect'
            )
            # Compensate each analysis window, band, and hydrophone for background noise and transmission loss
            received_power_windows = self.background_noise.apply_compensation(received_power_windows)
            source_power_windows = received_power_windows / self.transmission_model.power_transfer(
                source_track=passby.ship_track,
                receiver=passby.hydrophone,
                frequencies=self.filterbank(band).frequency,
                source_times=time_windows,
            )
            # Average each band and window over the hydrophones
            source_power_windows = np.mean(source_power_windows, axis='channels')
            # Average each band over the windows
            source_power[band] = np.mean(source_power_windows, axis='windows')
        self.passby_powers.append(source_power)
