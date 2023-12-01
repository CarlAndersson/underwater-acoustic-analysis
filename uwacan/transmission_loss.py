import numpy as np


class MlogR:
    def __init__(self, m=20):
        self.m = m

    def distance_factor(self, distance, **kwargs):
        """The power reduction factor for distance.

        The factor is -M log(r), but in power-scale.
        For `m=20`, this is equivalent with 1 / r**2.
        """
        level = self.m * np.log10(distance)
        return 10 ** (-level / 10)

    def __call__(self, input_power, receiver, source, **kwargs):
        distance = receiver.position.distance_to(source)
        return input_power / self.distance_factor(distance)


class SmoothLloydMirror(MlogR):
    def __init__(self, m=20, speed_of_sound=1500):
        super().__init__(m=m)
        self.speed_of_sound = speed_of_sound

    def surface_factor(self, grazing_angle, source_depth, frequency):
        """The factor by which a surface modifies a source power.

        For high frequencies this is a plain factor 2, since we are interested in the average value.
        For low frequencies, this is (2kd sin(θ))**2, where θ is the grazing angle.
        This is mixed as 1 / (1 / lf + 1 / hf).
        """
        kd = 2 * np.pi * frequency * source_depth / self.speed_of_sound

        mirror_lf = 4 * kd**2 * np.sin(grazing_angle)**2
        mirror_hf = 2
        mirror_reduction = 1 / (1 / mirror_lf + 1 / mirror_hf)
        return mirror_reduction

    def __call__(self, input_power, receiver, source, **kwargs):
        horizontal_distance = receiver.position.distance_to(source)
        distance = (horizontal_distance**2 + receiver.depth**2)**0.5
        grazing_angle = np.arctan2(receiver.depth, horizontal_distance)

        # kd = 2 * np.pi * input_power.frequency * source.depth / self.speed_of_sound

        # mirror_lf = 4 * kd**2 * np.sin(np.sin(grazing_angle))**2
        # mirror_hf = 2
        # mirror_reduction = (1 / mirror_lf + 1 / mirror_hf)

        # distance_level = self.m * np.log10(distance)
        # distance_reduction = 10 ** (distance_level / 10)

        total_factor = self.distance_factor(distance) * self.surface_factor(grazing_angle=grazing_angle, source_depth=source.depth, frequency=input_power.frequency)
        return input_power / total_factor


class SeabedCriticalAngle(SmoothLloydMirror):
    def __init__(self, water_depth, m=20, speed_of_sound=1500, substrate_compressional_speed=1500):
        super().__init__(m=m, speed_of_sound=speed_of_sound)
        self.substrate_compressional_speed = substrate_compressional_speed
        self.water_depth = water_depth

    def bottom_factor(self, frequency, source_depth, distance):
        """The factor by which a bottom contains the power from a source.

        The general idea is that power radiated towards the bottom will either
        stay in the water column, and thus arrive at the receiver at some point,
        or get transmitted into the substrate.
        The grazing angle below which power will be contained is the critical angle ψ.
        For high frequencies, the bottom retains 2ψ of the energy, on average.
        For low frequencies, we have a retention of 2 (kd)**2 (ψ - sin(ψ) cos(ψ)).
        This is then distance propagated by 1/(rH) instead of 1/r**2, to accommodate the cylindrical domain.
        The low-high frequency mixing is then done as 1 / (1 / lf + 1 / hf), as for smooth Lloyd mirror.

        Note that the output from this function is compensated for the spherical distance propagation.
        It is thus expected to additionally multiply by 1 / r**2, i.e. the output here ~ r/H.
        """
        critical_angle = np.arccos(self.speed_of_sound / self.substrate_compressional_speed)
        kd = 2 * np.pi * frequency * source_depth / self.speed_of_sound
        lf_approx = 2 * kd**2 * (critical_angle - np.sin(critical_angle) * np.cos(critical_angle))
        hf_approx = 2 * critical_angle
        bottom_approx = 1 / (1 / lf_approx + 1 / hf_approx) / self.water_depth * distance  # We multiply with the distance here since we will add a MlogR as well.
        return bottom_approx

    def __call__(self, input_power, receiver, source, **kwargs):
        horizontal_distance = receiver.position.distance_to(source)
        distance = (horizontal_distance**2 + receiver.depth**2)**0.5
        grazing_angle = np.arctan2(receiver.depth, horizontal_distance)

        distance_factor = self.distance_factor(distance=distance)
        direct_surface_factor = self.surface_factor(grazing_angle=grazing_angle, source_depth=source.depth, frequency=input_power.frequency)
        bottom_factor = self.bottom_factor(frequency=input_power.frequency, source_depth=source.depth, distance=distance)

        total_factor = distance_factor * (direct_surface_factor + bottom_factor)
        return input_power / total_factor


"""Seabed properties.

Properties included are grain size and speed of sound (compressional).
Based on Ainslie, M.A. Principles of Sonar Performance Modeling, Springer-Verlag Berlin Heidelberg, 2010.
"""
seabed_properties = {
    'very coarse sand': {
        'grain size': -0.5,
        'speed of sound': 1500 * 1.307,
    },
    'coarse sand': {
        'grain size': 0.5,
        'speed of sound': 1500 * 1.250,
    },
    'medium sand': {
        'grain size': 1.5,
        'speed of sound': 1500 * 1.198,
    },
    'fine sand': {
        'grain size': 2.5,
        'speed of sound': 1500 * 1.152,
    },
    'very fine sand': {
        'grain size': 3.5,
        'speed of sound': 1500 * 1.112,
    },
    'coarse silt': {
        'grain size': 4.5,
        'speed of sound': 1500 * 1.077,
    },
    'medium silt': {
        'grain size': 5.5,
        'speed of sound': 1500 * 1.048,
    },
    'fine silt': {
        'grain size': 6.5,
        'speed of sound': 1500 * 1.024,
    },
    'very fine silt': {
        'grain size': 7.5,
        'speed of sound': 1500 * 1.005,
    },
}
