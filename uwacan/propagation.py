"""Classes for modeling and compensating propagation of sound.

Main propagation models
-----------------------
.. autosummary::
    :toctree: generated

    MlogR
    SmoothLloydMirror
    SeabedCriticalAngle

Utilites
--------
.. autosummary::
    :toctree: generated

    cutoff_frequency
    perkins_cutoff
    read_valeport_data
    seabed_properties

Implementation interfaces
-------------------------
.. autosummary::
    :toctree: generated

    PropagationModel
    NonlocalPropagationModel

"""

import numpy as np
import abc


class PropagationModel(abc.ABC):
    """Base class for all propagation models with compensation."""

    @abc.abstractmethod
    def compensate_propagation(self, received_power, receiver, source):
        """Compensate the propagation of measured power.

        This takes a sound power measured at a receiver and compensates
        the propagation from a source.
        The ``received_power``, ``receiver``, and ``source`` can all have
        a time axis, but then they should be aligned.

        Parameters
        ----------
        received_power : `~uwacan.FrequencyData`
            The measured power.
        receiver : `~uwacan.positional.Coordinates`
            The placement of the receiver. Has latitude, longitude, and optionally depth.
        source : `~uwacan.positional.Coordinates`
            The placement of the source. Has latitude, longitude, and optionally depth.

        Returns
        -------
        source_power : `~uwacan.FrequencyData`
            The source power.

        Notes
        -----
        Subclasses should implement this function to apply the compensation.
        """


class NonlocalPropagationModel(PropagationModel):
    """Class for propagation models and only depend on relative coordinates."""

    @abc.abstractmethod
    def power_propagation(self, distance, frequency, receiver_depth, source_depth):
        """Compute the propagation factor.

        The propagation factor is the ratio of received power to sent power.
        Typically this is a value smaller than one, and is decreasing with
        increasing distances.

        Parameters
        ----------
        distance : `xarray.DataArray`
            The horizontal distance between source and receiver.
        frequency : `xarray.DataArray`
            The frequency to evaluate at.
        receiver_depth : `xarray.DataArray`
            The depth of the receiver.
        source_depth : `xarray.DataArray`
            The depth of the source.

        Returns
        -------
        F : `xarray.DataArray`
            The evaluated propagation factor.

        Notes
        -----
        Subclasses must implement this method to compute the propagation factor.
        The method has to accept all the arguments, but can ignore some of them
        if they are not relevant.
        """
        return 1

    @staticmethod
    def slant_range(horizontal_distance, receiver_depth, source_depth=None):
        """Compute the slant range from source to receiver.

        Parameters
        ----------
        horizontal_distance : array_like
            The horizontal distance between source and receiver.
        receiver_depth : array_like
            The depth below surface of the receiver.
        source_depth : array_like
            The depth below surface of the source.
            If the source depth is None, it defaults to 0.

        Returns
        -------
        slant_range : array_like or None
            The computed slant range if the receiver depth is not None,
            otherwise None.
        """
        if receiver_depth is None:
            return None
        # Optionally used to calculate the distance between source and receiver, instead of source surface to receiver.
        source_depth = source_depth or 0
        return (horizontal_distance**2 + (receiver_depth - source_depth) ** 2) ** 0.5

    def compensate_propagation(self, received_power, receiver, source):  # noqa: D102, takes the docstring from the superclass
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

    This implements a simple ``m log(r) + offset`` model

    Parameters
    ----------
    m : numeric, default 20
        The spreading factor.
        ``m=20`` gives spherical spreading, ``m=10`` gives cylindrical spreading.
    offset : numeric, default 0
        The offset to the propagation model.

    Notes
    -----
    This models calculates the fraction of power lost due
    to geometrical spreading of the energy, i.e::

        F = distance**(-m / 10) * 10 ** (-offset / 10)

    The distance is the slant range if a receiver depth is
    given, and the horizontal range otherwise.
    """

    def __init__(self, m=20, offset=0, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.offset = offset

    def power_propagation(self, distance, receiver_depth=None, **kwargs):  # noqa: D417, kwargs not documented
        """Calculate simple geometrical spreading.

        Parameters
        ----------
        distance : array_like
            The horizontal distance between source and receiver.
        receiver_depth : array_like, optional
            An optional receiver depth. If given, the spreading is
            evaluated on the slant range instead of the horizontal distance.

        Returns
        -------
        F : `xarray.DataArray`
            The evaluated propagation factor.
        """
        if receiver_depth is not None:
            distance = self.slant_range(distance, receiver_depth)
        return distance ** (-self.m / 10) * 10 ** (-self.offset / 10)


class SmoothLloydMirror(MlogR):
    """Geometrical spreading and average Lloyd mirror reflection loss model.

    This model compensates geometrical spreading as well as source interaction with the water surface.

    Parameters
    ----------
    m : int, default 20
        The spreading factor.
        ``m=20`` gives spherical spreading, ``m=10`` gives cylindrical spreading.
    speed_of_sound : numeric, default 1500
        The speed of sound in the water, used to calculate wave numbers.

    Notes
    -----
    For high frequencies the surface interaction is a plain factor ``hf = 2``,
    since we are interested in the average value. I.e., the surface boosts the
    radiation output.
    For low frequencies, the interaction factor is::

        lf = (2kd sin(θ))**2

    where θ is the grazing angle, measured to the receiver from the surface above the source.
    The low and high frequencies are mixed as::

        F_s = 1 / (1 / lf + 1 / hf).

    This surface factor F_s is then multiplied with the geometrical spreading::

        F_g = distance**(-m / 10) * 10 ** (-offset / 10)

    with the distance evaluated using the slant range.
    """

    def __init__(self, m=20, speed_of_sound=1500, **kwargs):
        super().__init__(m=m, **kwargs)
        self.speed_of_sound = speed_of_sound

    def power_propagation(self, distance, frequency, receiver_depth, source_depth, **kwargs):  # noqa: D417, ignored kwargs doc
        """Calculate surface interactions and geometrical spreading.

        Parameters
        ----------
        distance : array_like
            The horizontal distance between source and receiver.
        frequency : array_like
            The frequencies at which to evaluate.
        receiver_depth : array_like
            The receiver depth. Used to compute grazing angles and slant ranges.
        source_depth : array_like
            The source depth. Used to compute the ``kd`` term.

        Returns
        -------
        F : `xarray.DataArray`
            The evaluated propagation factor.
        """
        geometric_spreading = super().power_propagation(
            distance=distance, frequency=frequency, receiver_depth=receiver_depth, source_depth=source_depth, **kwargs
        )

        kd = 2 * np.pi * frequency * source_depth / self.speed_of_sound
        slant_range = self.slant_range(distance, receiver_depth)
        mirror_lf = 4 * kd**2 * (receiver_depth / slant_range) ** 2
        mirror_hf = 2
        mirror_reduction = 1 / (1 / mirror_lf + 1 / mirror_hf)

        return geometric_spreading * mirror_reduction


class SeabedCriticalAngle(SmoothLloydMirror):
    """The seabed critical angle propagation model.

    This model accounts for geometrical spreading, surface interactions, and simple bottom interactions.

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

    Notes
    -----
    The model is split in two parts, one spherical and one cylindrical. The spherical part is
    identical to the `SmoothLloydMirror` model, and gives the propagation factor::

        F_sphere = SmoothLloydMirror(...).power_propagation(...)

    The general idea for the cylindrical part is that power radiated towards the bottom will either
    stay in the water column, and thus arrive at the receiver at some point,
    or get transmitted into the substrate.
    The grazing angle below which power will be contained is the critical angle ψ.
    For high frequencies, the bottom retains an average factor::

        hf = 2ψ

    of the energy.
    For low frequencies, we have a retention of::

        lf = 2 (kd)**2 (ψ - sin(ψ) cos(ψ))

    The low-high frequency mixing and the distance propagation is then done as::

        F_cylindrical = 1 / (rH) / (1 / lf + 1 / hf)

    where we use the same low-high frequency mixing as for smooth Lloyd mirror,
    but 1 / (rH) to accommodate the cylindrical domain.
    The final propagation factor is the sum of the spherical and cylindrical energy::

        F = F_spherical + F_cylindrical

    """

    def __init__(self, water_depth, n=10, substrate_compressional_speed=1500, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.substrate_compressional_speed = substrate_compressional_speed
        self.water_depth = water_depth

    def power_propagation(self, distance, frequency, receiver_depth, source_depth, **kwargs):  # noqa: D417, no docs for kwargs
        """Calculate geometrical spreading and interactions with the surface and the bottom.

        Parameters
        ----------
        distance : array_like
            The horizontal distance between source and receiver.
        frequency : array_like
            The frequencies at which to evaluate.
        receiver_depth : array_like
            The receiver depth. Used to compute grazing angles and slant ranges.
        source_depth : array_like
            The source depth. Used to compute the ``kd`` term.

        Returns
        -------
        F : `xarray.DataArray`
            The evaluated propagation factor.
        """
        surface_effect = super().power_propagation(
            distance=distance, frequency=frequency, receiver_depth=receiver_depth, source_depth=source_depth, **kwargs
        )

        slant_range = self.slant_range(distance, receiver_depth)
        critical_angle = np.arccos(self.speed_of_sound / self.substrate_compressional_speed)
        kd = 2 * np.pi * frequency * source_depth / self.speed_of_sound
        lf_approx = 2 * kd**2 * (critical_angle - np.sin(critical_angle) * np.cos(critical_angle))
        hf_approx = 2 * critical_angle
        cylindrical_spreading = 1 / (self.water_depth * slant_range ** (self.n / 10))
        bottom_effect = 1 / (1 / lf_approx + 1 / hf_approx)

        return surface_effect + bottom_effect * cylindrical_spreading


def cutoff_frequency(water_depth, substrate_compressional_speed=np.inf, speed_of_sound=1500):
    """Calculate cutoff frequency for shallow waters.

    For shallow waters, the long distance cutoff describes
    a frequency below which there will be no waveguide.

    Parameters
    ----------
    water_depth : array_like
        The water depth, in meters.
    substrate_compressional_speed : array_like
        The speed of sound in the seabed, in m/s.
    speed_of_sound : array_like, default=1500
        The speed of sound in the water, in m/s.

    Notes
    -----
    This is evaluated as [1]_::

        speed_ratio = speed_of_sound / substrate_compressional_speed
        speed_of_sound / (water_depth * 4 * (1 - speed_ratio**2)**0.5)

    This means that the cutoff frequency goes up for softer substrates,
    and up for shallower depths.

    References
    ----------
    .. [1] F. B. Jensen, Wi. A. Kuperman, M. B. Porter, and H. Schmidt,
           "Computational Ocean Acoustics", 2nd ed. Springer New York, 2011.
           Eq. (1.38).
    """
    speed_ratio = speed_of_sound / substrate_compressional_speed
    return speed_of_sound / (water_depth * 4 * (1 - speed_ratio**2) ** 0.5)


def perkins_cutoff(water_depth, substrate_compressional_speed=np.inf, speed_of_sound=1500, mode_order=1):
    """Calculate modal cutoff for shallow waters.

    For shallow waters, the long distance cutoff describes
    a frequency below which a specific mode cannot propagate.

    Parameters
    ----------
    water_depth : array_like
        The water depth, in meters.
    substrate_compressional_speed : array_like
        The speed of sound in the seabed, in m/s.
    speed_of_sound : array_like, default=1500
        The speed of sound in the water, in m/s.
    mode_order : array_like, default=1
        Which mode order to evaluate.
        A mode order of 1 yields the same as `cutoff_frequency`.

    Notes
    -----
    This is evaluated as [1]_::

        speed_ratio = speed_of_sound / substrate_compressional_speed
        (mode_order - 0.5) * speed_of_sound / (2 * water_depth * (1 - speed_ratio**2)**0.5)

    This means that the cutoff frequency goes up for softer substrates,
    and up for shallower depths, and is higher for higher modes.

    References
    ----------
    .. [1] F. B. Jensen, Wi. A. Kuperman, M. B. Porter, and H. Schmidt,
           "Computational Ocean Acoustics", 2nd ed. Springer New York, 2011.
           Eq. (2.191).
    """
    speed_ratio = speed_of_sound / substrate_compressional_speed
    return (mode_order - 0.5) * speed_of_sound / (2 * water_depth * (1 - speed_ratio**2) ** 0.5)


seabed_properties = {
    "very coarse sand": {
        "grain size": -0.5,
        "speed of sound": 1500 * 1.307,
    },
    "coarse sand": {
        "grain size": 0.5,
        "speed of sound": 1500 * 1.250,
    },
    "medium sand": {
        "grain size": 1.5,
        "speed of sound": 1500 * 1.198,
    },
    "fine sand": {
        "grain size": 2.5,
        "speed of sound": 1500 * 1.152,
    },
    "very fine sand": {
        "grain size": 3.5,
        "speed of sound": 1500 * 1.112,
    },
    "coarse silt": {
        "grain size": 4.5,
        "speed of sound": 1500 * 1.077,
    },
    "medium silt": {
        "grain size": 5.5,
        "speed of sound": 1500 * 1.048,
    },
    "fine silt": {
        "grain size": 6.5,
        "speed of sound": 1500 * 1.024,
    },
    "very fine silt": {
        "grain size": 7.5,
        "speed of sound": 1500 * 1.005,
    },
}
"""Dict with seabed properties.

Properties included are grain size and speed of sound (compressional).
Included substrates are

- very coarse sand
- coarse sand
- medium sand
- fine sand
- very fine sand
- coarse silt
- medium silt
- fine silt
- very fine silt

These substrates are the keys to the dict.

Based on Ainslie, M.A. Principles of Sonar Performance Modeling, Springer-Verlag Berlin Heidelberg, 2010.
"""


def read_valeport_data(filepath):
    """Read Valeport swift data.

    This reads data stored in the ``.vp2`` format.
    """
    from io import StringIO
    import pandas
    import re
    import xarray as xr

    with open(filepath, "r") as file:
        contents = file.read()

    lat = float(re.search(r"Latitude=([\d.]*)", contents).groups()[0])
    lon = float(re.search(r"Longitude=([\d.]*)", contents).groups()[0])

    data_idx = contents.find("[DATA]")
    data = contents[data_idx:].splitlines()
    data_stream = StringIO("\n".join([data[1].strip()] + data[3:]))
    df = pandas.read_csv(data_stream, delimiter="\t", parse_dates=["Date/Time"])
    ds = xr.Dataset.from_dataframe(df).set_coords("Depth").swap_dims(index="Depth").drop_vars("index")
    return ds.assign(latitude=lat, longitude=lon)
