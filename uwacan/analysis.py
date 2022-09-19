"""Various analysis protocols and standards for recorded underwater noise from ships."""

import numpy as np
from . import positional, signals


class BureauVeritasSourceSpectrum:
    aspect_window_resolution = 5
    aspect_window_range = (-45, 45)

    def __init__(
        self,
        recording,
        track,
        passby_timestamps,
        transmission_model,
        background_noise,
        frequency_range,
        bandtypes='thirds',
        passby_search_duration=10 * 60
    ):
        self.recording = recording
        self.track = track
        self.transmission_model = transmission_model
        self.background_noise = background_noise
        self.frequency_range = frequency_range

        if isinstance(bandtypes, str):
            bandtypes = [bandtypes]
        self.bandtypes = bandtypes

        self.passby_powers = []

    def process_passby(self, time):
        if not isinstance(time, positional.TimeWindow):
            time = positional.TimeWindow(center=time, duration=self.passby_search_duration)
        recording = self.recording[time]
        track = self.track[time]

        # TODO: some kind of check that the coarse selection of time window is good enough to cover the needed range for the aspect windows.
        # If we fail this check, recursively call the processing function with an extended coarse window.
        # Get aspect angles between ship position and hydrophone position
        time_windows = track.aspect_windows(
            reference_point=recording.position,
            resolution=self.aspect_window_resolution,
            range=self.aspect_window_range,
            length=self.aspect_window_length,
        )

        time_signal = recording[time_windows[0].start:time_windows[-1].stop]
        spectrogram = signals.Spectrogram(time_signal, window_duration=1)

        source_powers = {}
        for band in self.bandtypes:
            received_power = self.filterbanks[band](time_signal=time_signal, spectrogram=spectrogram)
            source_power = []
            for window in time_windows:
                window_power = received_power[window].data.mean(axis=1)
                window_power = self.background_noise.compensate(window_power)
                window_power = window_power * self.transmission_model.power_loss(
                    receiver=recording,
                    source_track=track[window].mean,  # The Bureau Veritas method evaluates the TL at window center only.
                    frequencies=received_power.frequencies,
                    time=window.center
                )
                source_power.append(np.mean(window_power, axis='channels'))
            source_power = np.mean(source_power, axis=0)
            source_powers[band] = source_power
        return source_powers


        # Calculate the spectrogram of the passby
        spectrogram = passby.hydrophone.time_range(time_windows[0].start_time, time_windows[-1].stop_time).spectrogram()
        source_power = {}

        for band in self.bandtypes:
            # Calculate the received power in the frequency bands at each time instance
            received_power_bands = self.filterbank(band)(spectrogram)
            # Average the power over time for each analysis window, for each hydrophone
            received_power_windows = np.stack([
                np.mean(received_power_bands.time_range(window.start_time, window.stop_time).signal, axis='t_psd')
                for window in time_windows],
                axis='t_aspect'
            )
            # Compensate each analysis window, band, and hydrophone for background noise and transmission loss
            received_power_windows = self.compensate_for_background(
                received_power_windows,
                frequencies=self.filterbank(band).frequency,
            )
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
