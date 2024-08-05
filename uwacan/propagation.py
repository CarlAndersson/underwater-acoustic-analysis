import numpy as np
from . import positional
import abc


class PropagationModel(abc.ABC):
    @abc.abstractmethod
    def compensate_propagation(self, input_power, receiver, source):
        ...


class NonlocalPropagationModel(PropagationModel):
    @abc.abstractmethod
    def power_propagation(self, distance, frequency, receiver_depth, source_depth):
        return 1

    @staticmethod
    def slant_range(horizontal_distance, receiver_depth, source_depth=None):
        if receiver_depth is None:
            return None
        source_depth = source_depth or 0  # Optionally used to calculate the distance between source and receiver, instead of source surface to receiver.
        return (horizontal_distance**2 + (receiver_depth - source_depth)**2)**0.5

    def compensate_propagation(self, received_power, receiver, source):
        """Compensates received power for propagation loss.

        Propagation models that require source/receiver depths will not evaluate
        properly if they are not given in the input arguments.
        Similarly frequencies are sometimes required, and should be given as
        coordinates for the received power.

        Parameters
        ----------
        received_power : xr.DataArray
            The received power to compensate.
            The frequencies to evaluate the propagation will be taken from this object.
        receiver : xr.Dataset
            Specification of the receiver, with latitude, longitude, and optionally depth.
        source : xr.Dataset
            Specification of the source, with latitude, longitude, and optionally depth.

        Returns
        -------
        source_power : xr.DataArray
            The calculated source power. The dimensions of this will depend on the
            dimensions of the three inputs.
        """
        distance = receiver.distance_to(source)
        try:
            receiver_depth = receiver.depth
        except AttributeError:
            try:
                receiver_depth = receiver["depth"]
            except KeyError:
                receiver_depth = None

        try:
            source_depth = source.depth
        except AttributeError:
            try:
                source_depth = source["depth"]
            except KeyError:
                source_depth = None

        try:
            frequency = received_power.frequency
        except AttributeError:
            frequency = None

        power_loss = self.power_propagation(
            distance=distance,
            frequency=frequency,
            source_depth=source_depth,
            receiver_depth=receiver_depth,
        )
        return received_power / power_loss


class MlogR(NonlocalPropagationModel):
    """Geometrical spreading loss model.

    This implements a simple M log(r) + M_0 model, using
    the slant range if available and the horizontal
    range otherwise.

    Parameters
    ----------
    m : numeric, default 20
        The spreading factor.
        20 -> spherical spreading, 10 -> cylindrical spreading.
    offset : numeric, default 0
        The offset to the propagation model, M_0 in the above.
    """
    def __init__(self, m=20, offset=0, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.offset = offset

    def power_propagation(self, distance, receiver_depth=None, **kwargs):
        """Calculates simple geometrical spreading.

        This function calculates the fraction of power lost due
        to geometrical spreading of the energy, i.e., distance**(-m / 10).
        """
        if receiver_depth is not None:
            distance = self.slant_range(distance, receiver_depth)
        return distance ** (-self.m / 10) * 10 ** (-self.offset / 10)


class SmoothLloydMirror(MlogR):
    """Geometrical spreading and average Lloyd mirror reflection loss model.

    This model compensates geometrical spreading as well as source interaction with the water surface.
    For high frequencies this is a plain factor 2, since we are interested in the average value.
    For low frequencies, this is (2kd sin(θ))**2, where θ is the grazing angle.
    This is mixed as 1 / (1 / lf + 1 / hf).
    This surface factor is then multiplied with the geometrical spreading, see MlogR.

    Parameters
    ----------
    m : int, default 20
        The spreading factor.
        20 -> spherical spreading, 10 -> cylindrical spreading.
    speed_of_sound : numeric, default 1500
        The speed of sound in the water, used to calculate wave numbers.
    """
    def __init__(self, m=20, speed_of_sound=1500, **kwargs):
        super().__init__(m=m, **kwargs)
        self.speed_of_sound = speed_of_sound

    def power_propagation(self, distance, frequency, receiver_depth, source_depth, **kwargs):
        """Calculates surface interactions and geometrical spreading.

        This function calculates the fraction of power lost due
        to geometrical spreading of the energy, i.e., distance**(-m / 10),
        and compensates for the average interactions of a pressure release surface.
        """
        geometric_spreading = super().power_propagation(distance=distance, frequency=frequency, receiver_depth=receiver_depth, source_depth=source_depth, **kwargs)

        kd = 2 * np.pi * frequency * source_depth / self.speed_of_sound
        slant_range = self.slant_range(distance, receiver_depth)
        mirror_lf = 4 * kd**2 * (receiver_depth / slant_range)**2
        mirror_hf = 2
        mirror_reduction = 1 / (1 / mirror_lf + 1 / mirror_hf)

        return geometric_spreading * mirror_reduction


class SeabedCriticalAngle(SmoothLloydMirror):
    """The seabed critical angle propagation model.

    This model accounts for geometrical spreading, surface interactions, and simple bottom interactions.
    The model is split in two parts, one spherical and one cylindrical. The spherical part is
    identical to the SmoothLloydMirror model.
    The general idea for the cylindrical part is that power radiated towards the bottom will either
    stay in the water column, and thus arrive at the receiver at some point,
    or get transmitted into the substrate.
    The grazing angle below which power will be contained is the critical angle ψ.
    For high frequencies, the bottom retains 2ψ of the energy, on average.
    For low frequencies, we have a retention of 2 (kd)**2 (ψ - sin(ψ) cos(ψ)).
    This is then distance propagated by 1/(rH) instead of 1/r**2, to accommodate the cylindrical domain.
    The low-high frequency mixing is then done as 1 / (1 / lf + 1 / hf), as for smooth Lloyd mirror.

    Parameters
    ----------
    water_depth : numeric
        The water depth to use for the cylindrical spreading.
    n : numeric, default 10
        The geometrical spreading factor to use for the cylindrical spreading.
    m : numeric, default 20
        The geometrical spreading factor to use for the spherical spreading.
    speed_of_sound : numeric, default 1500
        The speed of sound in the water. Used to calculate wave numbers and the critical angle.
    substrate_compressional_speed, numeric, default 1500
        The speed of sound in the water. Used to calculate the critical angle.
    """
    def __init__(self, water_depth, n=10, substrate_compressional_speed=1500, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.substrate_compressional_speed = substrate_compressional_speed
        self.water_depth = water_depth

    def power_propagation(self, distance, frequency, receiver_depth, source_depth, **kwargs):
        """Calculates geometrical spreading and interactions with the surface and the bottom.

        This function calculates the fraction of power lost due
        to geometrical spreading of the energy, i.e., distance**(-m / 10),
        and compensates for the average interactions of a pressure release surface,
        as well as a very simplified model for the bottom.
        """
        surface_effect = super().power_propagation(distance=distance, frequency=frequency, receiver_depth=receiver_depth, source_depth=source_depth, **kwargs)

        slant_range = self.slant_range(distance, receiver_depth)
        critical_angle = np.arccos(self.speed_of_sound / self.substrate_compressional_speed)
        kd = 2 * np.pi * frequency * source_depth / self.speed_of_sound
        lf_approx = 2 * kd**2 * (critical_angle - np.sin(critical_angle) * np.cos(critical_angle))
        hf_approx = 2 * critical_angle
        cylindrical_spreading = 1 / (self.water_depth * slant_range ** (self.n / 10))
        bottom_effect = 1 / (1 / lf_approx + 1 / hf_approx)

        return surface_effect + bottom_effect * cylindrical_spreading


def cutoff_frequency(water_depth, substrate_compressional_speed=np.inf, speed_of_sound=1500):
    """
    Computational ocean acoustics, (1.38)
    """
    speed_ratio = speed_of_sound / substrate_compressional_speed
    return speed_of_sound / (water_depth * 4 * (1 - speed_ratio**2)**0.5)


def perkins_cutoff(water_depth, substrate_compressional_speed=np.inf, speed_of_sound=1500, mode_order=1):
    """
    Computational ocean acoustics, (2.191)
    This gives the same as the `cutoff_frequency` above for `mode_order == 1`.
    """
    speed_ratio = speed_of_sound / substrate_compressional_speed
    return (mode_order - 0.5) * speed_of_sound / (2 * water_depth * (1 - speed_ratio**2)**0.5)


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


def read_valeport_data(filepath):
    from io import StringIO
    import pandas
    import re
    import xarray as xr
    # filepath = r"D:\Tern Island Vinga\SVP\VL_72960_230830110544.vp2"
    with open(filepath, 'r') as file:
        contents = file.read()

    lat = float(re.search(r"Latitude=([\d.]*)", contents).groups()[0])
    lon = float(re.search(r"Longitude=([\d.]*)", contents).groups()[0])

    data_idx = contents.find('[DATA]')
    data = contents[data_idx:].splitlines()
    data_stream = StringIO('\n'.join([data[1].strip()] + data[3:]))
    df = pandas.read_csv(data_stream, delimiter='\t', parse_dates=['Date/Time'])
    ds = xr.Dataset.from_dataframe(df).set_coords('Depth').swap_dims(index='Depth').drop_vars('index')
    return ds.assign(latitude=lat, longitude=lon)
