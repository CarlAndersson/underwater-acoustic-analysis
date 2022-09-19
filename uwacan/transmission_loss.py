import numpy as np

# TODO: As the models become more advanced we have to include more inputs to the calculation.
# Otherwise we have no way of knowing at what frequencies to evaluate the TL.
# The question is how to get it in?


class TransmissionModel:
    def pressure_transfer(self, receiver, source_track, frequencies=None, time=None):
        return 1 / self.pressure_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time)

    def pressure_loss(self, receiver, source_track, frequencies=None, time=None):
        return 1 / self.pressure_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time)

    def power_transfer(self, receiver, source_track, frequencies=None, time=None):
        return 1 / self.power_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time)

    def power_loss(self, receiver, source_track, frequencies=None, time=None):
        return 1 / self.power_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time)

    def level_transfer(self, receiver, source_track, frequencies=None, time=None):
        return -self.level_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time)

    def level_loss(self, receiver, source_track, frequencies=None, time=None):
        return -self.level_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time)


class PressureModel(TransmissionModel):
    def power_transfer(self, receiver, source_track, frequencies=None, time=None):
        return self.pressure_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time) ** 2

    def power_loss(self, receiver, source_track, frequencies=None, time=None):
        return self.pressure_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time) ** 2

    def level_transfer(self, receiver, source_track, frequencies=None, time=None):
        return 20 * np.log10(self.pressure_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time))

    def level_loss(self, receiver, source_track, frequencies=None, time=None):
        return 20 * np.log10(self.pressure_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time))


class PowerModel(TransmissionModel):
    def pressure_transfer(self, receiver, source_track, frequencies=None, time=None):
        return self.power_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time) ** 0.5

    def pressure_loss(self, receiver, source_track, frequencies=None, time=None):
        return self.power_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time) ** 0.5

    def level_transfer(self, receiver, source_track, frequencies=None, time=None):
        return 10 * np.log10(self.power_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time))

    def level_loss(self, receiver, source_track, frequencies=None, time=None):
        return 10 * np.log10(self.power_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time))


class LevelModel(TransmissionModel):
    def pressure_transfer(self, receiver, source_track, frequencies=None, time=None):
        return 10 ** (self.level_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time) / 20)

    def pressure_loss(self, receiver, source_track, frequencies=None, time=None):
        return 10 ** (self.level_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time) / 20)

    def power_transfer(self, receiver, source_track, frequencies=None, time=None):
        return 10 ** (self.level_transfer(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time))

    def power_loss(self, receiver, source_track, frequencies=None, time=None):
        return 10 ** (self.level_loss(receiver=receiver, source_track=source_track, frequencies=frequencies, time=time))


class MlogR(LevelModel):
    def __init__(self, M=20):
        self.M = M

    def level_loss(self, receiver, source_track, frequencies=None, time=None):
        return self.M * np.log10(source_track.distance_to(receiver.position))
