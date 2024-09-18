"""Implementations for computations on coordinates.

This module holds all the implementations and wrappers to handle
coordinate computations in this package.

A few of the most useful classes are lifted to the main package namespace.
They should be accessed from there when used externally, but from here when
used internally.

.. autosummary::

    uwacan.Position
    uwacan.Track
    uwacan.sensor

Other coordinate wrappers
-------------------------
.. autosummary::
    :toctree: generated

    Coordinates
    Positions
    Line
    BoundingBox
    Sensor
    SensorPosition
    SensorArray
    LocatedSensorArray
    ColocatedSensorArray

Unit conversions
----------------
.. autosummary::
    :toctree: generated

    nm_to_m
    m_to_nm
    mps_to_knots
    knots_to_kmph
    kmph_to_knots
    wrap_angle
    local_mercator_to_wgs84
    wgs84_to_local_mercator

Geodesic computations
---------------------
.. autosummary::
    :toctree: generated

    distance_to
    bearing_to
    shift_position
    average_angle
    angle_between

"""

import re
import numpy as np
import xarray as xr
from . import _core
from pathlib import Path
import functools


_WGS84_equatorial_radius = 6_378_137.0
_WGS84_polar_radius = 6_356_752.3
_mercator_scale_factor = 0.9996


def nm_to_m(nm):
    """Convert nautical miles to meters."""
    return nm * 1852


def m_to_nm(m):
    """Convert meters to nautical miles."""
    return m / 1852


def mps_to_knots(mps):
    """Convert meters per second to knots."""
    return mps * (3600 / 1852)


def knots_to_mps(knots):
    """Convert knots to meters per second."""
    return knots * (1852 / 3600)


def knots_to_kmph(knots):
    """Convert knots to kilometers per hour."""
    return knots * 1.852


def kmph_to_knots(kmph):
    """Convert kilometers per hour to knots."""
    return kmph / 1.852


def wrap_angle(degrees):
    """Wrap an angle to (-180, 180]."""
    return 180 - np.mod(180 - degrees, 360)


def local_mercator_to_wgs84(easting, northing, reference_latitude, reference_longitude):
    r"""Convert local mercator coordinates into wgs84 coordinates.

    Conventions here are :math:`λ` as the longitude and :math:`φ` as the latitude,
    :math:`x` is easting and :math:`y` is northing.

    .. math::

        λ &= λ_0 + x/R \\
        φ &= 2\arctan(\exp((y + y_0)/R)) - π/2

    The northing offset :math:`y_0` is computed by converting the reference point into
    a mercator projection with the `wgs84_to_local_mercator` function, using (0, 0) as
    the reference coordinates.
    """
    radius = _WGS84_equatorial_radius * _mercator_scale_factor

    ref_east, ref_north = wgs84_to_local_mercator(reference_latitude, reference_longitude, 0, 0)
    longitude = reference_longitude + np.degrees(easting / radius)
    latitude = 2 * np.arctan(np.exp((northing + ref_north) / radius)) - np.pi / 2
    latitude = np.degrees(latitude)
    return latitude, longitude


def wgs84_to_local_mercator(latitude, longitude, reference_latitude, reference_longitude):
    r"""Convert wgs84 coordinates into a local mercator projection.

    Conventions here are :math:`λ` as the longitude and :math:`φ` as the latitude,
    :math:`x` is easting and :math:`y` is northing.

    .. math::
        x &= R(λ - λ_0)\\
        y &= R\ln(\tan(π/4 + (φ - φ_0)/2))
    """
    radius = _WGS84_equatorial_radius * _mercator_scale_factor
    local_longitude = np.radians(longitude - reference_longitude)
    local_latitude = np.radians(latitude - reference_latitude)
    easting = radius * local_longitude
    northing = radius * np.log(np.tan(np.pi / 4 + local_latitude / 2))
    return easting, northing


_re_rp_2 = (_WGS84_equatorial_radius / _WGS84_polar_radius) ** 2


def _geodetic_to_geocentric(lat):
    r"""Compute the geocentric latitude from geodetic, in radians.

    The geocentric latitude is the latitude as seen from the center
    of the earth. The geodetic latitude is the angle formed with the
    equatorial plane when drawing the normal at a surface on the earth.

    The conversion for the geocentric latitude :math:`\hat φ` is

    .. math::
        \hat φ = \arctan(\tan(φ) b^2/a^2)

    with the geodetic latitude :math:`φ`, and the equatorial and polar
    earth radii :math:`a, b` respectively.
    """
    return np.arctan(np.tan(lat) / _re_rp_2)


def _geocentric_to_geodetic(lat):
    r"""Compute the geodetic latitude from geocentric, in radians.

    The geodetic latitude is the angle formed with the
    equatorial plane when drawing the normal at a surface on the earth.
    The geocentric latitude is the latitude as seen from the center
    of the earth.

    The conversion for the geocentric latitude :math:`φ` is

    .. math::
         φ = \arctan(\tan(\hat φ) a^2/b^2)

    with the geocentric latitude :math:`\hat φ`, and the equatorial and polar
    earth radii :math:`a, b` respectively.
    """
    return np.arctan(np.tan(lat) * _re_rp_2)


def _local_earth_radius(lat):
    r"""Compute the earth radius at a given latitude, in radians.

    The formula is

    .. math::
        R( φ) = \sqrt{\frac{(a^2\cos φ)^2+(b^2\sin φ)^2}{(a\cos φ)^2+(b\sin φ)^2}}

    with the geodetic latitude :math:`φ`, and the equatorial and polar earth radii
    :math:`a, b` respectively, see https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii.
    """
    return (
        ((_WGS84_equatorial_radius**2 * np.cos(lat)) ** 2 + (_WGS84_polar_radius**2 * np.sin(lat)) ** 2)
        / ((_WGS84_equatorial_radius * np.cos(lat)) ** 2 + (_WGS84_polar_radius * np.sin(lat)) ** 2)
    ) ** 0.5


def _haversine(theta):
    r"""Compute the haversine of an angle, in radians.

    This is the same as

    .. math:: \sin^2(θ/2)
    """
    return np.sin(theta / 2) ** 2


def distance_to(lat_1, lon_1, lat_2, lon_2):
    r"""Calculate the distance between two coordinates.

    Conventions here are λ as the longitude and φ as the latitude.
    This implementation uses the Haversine formula:

    .. math::
        c &= H(Δφ) + (1 - H(Δφ) - H(φ_1 + φ_2)) ⋅ H(Δλ)\\
        d &= 2 R(φ) ⋅ \arcsin(\sqrt{c})\\
        H(θ) &= \sin^2(θ/2)

    implemented internally with conversions to geocentric coordinates
    and a latitude-dependent earth radius.

    Parameters
    ----------
    lat_1 : float
        Latitude of the first point in degrees.
    lon_1 : float
        Longitude of the first point in degrees.
    lat_2 : float
        Latitude of the second point in degrees.
    lon_2 : float
        Longitude of the second point in degrees.

    Returns
    -------
    float
        Distance between the two points in meters.

    """
    lat_1 = np.radians(lat_1)
    lon_1 = np.radians(lon_1)
    lat_2 = np.radians(lat_2)
    lon_2 = np.radians(lon_2)
    r = _local_earth_radius((lat_1 + lat_2) / 2)
    lat_1 = _geodetic_to_geocentric(lat_1)
    lat_2 = _geodetic_to_geocentric(lat_2)
    central_angle = _haversine(lat_2 - lat_1) + (
        1 - _haversine(lat_1 - lat_2) - _haversine(lat_1 + lat_2)
    ) * _haversine(lon_2 - lon_1)
    d = 2 * r * np.arcsin(central_angle**0.5)
    return d


def bearing_to(lat_1, lon_1, lat_2, lon_2):
    r"""Calculate the heading from one coordinate to another.

    Conventions here are λ as the longitude and φ as the latitude.
    The implementation is based on spherical trigonometry, with
    conversions to geocentric coordinates.
    This can be written as

    .. math::
        Δx &= \cos(φ_1) \sin(φ_2) - \sin(φ_1)\cos(φ_2)\cos(φ_2 - φ_1) \\
        Δy &= \sin(λ_2 - λ_1)\cos(φ_2) \\
        θ &= \arctan(Δy/Δx)

    Parameters
    ----------
    lat_1 : float
        Latitude of the first point in degrees.
    lon_1 : float
        Longitude of the first point in degrees.
    lat_2 : float
        Latitude of the second point in degrees.
    lon_2 : float
        Longitude of the second point in degrees.

    Returns
    -------
    float
        Bearing from the first point to the second point in degrees, wrapped to (-180, 180].
    """
    lat_1 = np.radians(lat_1)
    lon_1 = np.radians(lon_1)
    lat_2 = np.radians(lat_2)
    lon_2 = np.radians(lon_2)
    lat_1 = _geodetic_to_geocentric(lat_1)
    lat_2 = _geodetic_to_geocentric(lat_2)

    dy = np.sin(lon_2 - lon_1) * np.cos(lat_2)
    dx = np.cos(lat_1) * np.sin(lat_2) - np.sin(lat_1) * np.cos(lat_2) * np.cos(lat_2 - lat_1)

    bearing = np.arctan2(dy, dx)
    return wrap_angle(np.degrees(bearing))


def shift_position(lat, lon, distance, bearing):
    r"""Shifts a position given by latitude and longitude by a certain distance and bearing.

    The implementation is based on spherical trigonometry, with internal
    conversions to geocentric coordinates, and using the local radius of the earth.
    This is expressed as

    .. math::
        φ_2 &= \arcsin(\sin(φ_1) ⋅ \cos(δ) + \cos(φ_1) ⋅ \sin(δ) ⋅ \cos(θ)) \\
        λ_2 &= λ_1 + \arctan(\frac{\sin(θ) ⋅ \sin(δ) ⋅ \cos(φ_1)}{\cos(δ) - \sin(φ_1) ⋅ \sin(φ_2)})

    where: φ is latitude, λ is longitude, θ is the bearing (clockwise from north),
    δ is the angular distance d/R; d being the distance traveled, R the earth's radius.

    Parameters
    ----------
    lat : float
        Latitude of the initial position in degrees.
    lon : float
        Longitude of the initial position in degrees.
    distance : float
        Distance to move from the initial position in meters.
    bearing : float
        Direction to move from the initial position in degrees.

    Returns
    -------
    new_lat : float
        Latitude of the new position in degrees.
    new_lon : float
        Longitude of the new position in degrees.
    """
    lat = np.radians(lat)
    lon = np.radians(lon)
    bearing = np.radians(bearing)
    r = _local_earth_radius(lat)
    lat = _geodetic_to_geocentric(lat)
    dist = distance / r  # angular distance
    new_lat = np.arcsin(np.sin(lat) * np.cos(dist) + np.cos(lat) * np.sin(dist) * np.cos(bearing))
    new_lon = lon + np.arctan2(
        np.sin(bearing) * np.sin(dist) * np.cos(lat), np.cos(dist) - np.sin(lat) * np.sin(new_lat)
    )
    new_lat = _geocentric_to_geodetic(new_lat)
    return np.degrees(new_lat), np.degrees(new_lon)


def average_angle(angle, resolution=None):
    """Calculate the average angle and optionally round it to a specified resolution.

    Parameters
    ----------
    angle : array_like
        Array of angles in degrees to be averaged.
    resolution : int, str, optional
        Specifies the resolution for rounding the angle. It can be an integer specifying the number
        of divisions (e.g., 4, 8, 16) or a string ('4', '8', '16', 'four', 'eight', 'sixteen').

    Returns
    -------
    float or str
        If resolution is None, returns the average angle in degrees.
        If resolution is an integer, returns the average angle rounded to this fraction of a turn.
        If resolution is a string, returns the closest named direction (e.g., 'North', 'Southwest').

    Raises
    ------
    ValueError
        If an unknown resolution specifier is provided.

    Notes
    -----
    The function converts the input angles to complex numbers, computes their mean, and then converts back to an angle.
    If a string resolution is specified, the function maps the average angle to the nearest named direction.

    Examples
    --------
    >>> average_angle([350, 10, 40, 40])
    20.15962133607971
    >>> average_angle([350, 10, 30], resolution=10)
    36.0
    >>> average_angle([350, 10, 30], resolution='four')
    'North'
    >>> average_angle([350, 10, 20], resolution='sixteen')
    'North-northeast'
    """
    complex_angle = np.exp(1j * np.radians(angle))
    angle = wrap_angle(np.degrees(np.angle(complex_angle.mean())))
    if resolution is None:
        return angle

    if not isinstance(resolution, str):
        return wrap_angle(np.round(angle / 360 * resolution) * 360 / resolution)

    resolution = resolution.lower()
    if "4" in resolution or "four" in resolution:
        resolution = 4
    elif "8" in resolution or "eight" in resolution:
        resolution = 8
    elif "16" in resolution or "sixteen" in resolution:
        resolution = 16
    else:
        raise ValueError(f"Unknown resolution specifier '{resolution}'")

    names = [
        (-180.0, "south"),
        (-90.0, "west"),
        (0.0, "north"),
        (90.0, "east"),
        (180.0, "south"),
    ]

    if resolution >= 8:
        names.extend(
            [
                (-135.0, "southwest"),
                (-45.0, "northwest"),
                (45.0, "northeast"),
                (135.0, "southeast"),
            ]
        )
    if resolution >= 16:
        names.extend(
            [
                (-157.5, "south-southwest"),
                (-112.5, "west-southwest"),
                (-67.5, "west-northwest"),
                (-22.5, "north-northwest"),
                (22.5, "north-northeast"),
                (67.5, "east-northeast"),
                (112.5, "east-southeast"),
                (157.5, "south-southeast"),
            ]
        )
    name = min([(abs(deg - angle), name) for deg, name in names], key=lambda x: x[0])[1]
    return name.capitalize()


def angle_between(lat, lon, lat_1, lon_1, lat_2, lon_2):
    """Calculate the angle between two coordinates, as seen from a center vertex.

    The angle is counted positive if the second point is clockwise of the first point,
    as seen from the center vertex.

    Parameters
    ----------
    lat : float
        Latitude of the center vertex in degrees.
    lon : float
        Longitude of the center vertex in degrees.
    lat_1 : float
        Latitude of the first point in degrees.
    lon_1 : float
        Longitude of the first point in degrees.
    lat_2 : float
        Latitude of the second point in degrees.
    lon_2 : float
        Longitude of the second point in degrees.

    Returns
    -------
    float
        The angle between the two points as seen from the center vertex, in degrees. The angle is normalized to the range (-180, 180].
    """
    bearing_1 = bearing_to(lat, lon, lat_1, lon_1)
    bearing_2 = bearing_to(lat, lon, lat_2, lon_2)
    return wrap_angle(bearing_2 - bearing_1)


class Coordinates(_core.DatasetWrap):
    """Container for latitude and longitude.

    This class has a number of useful methods to perform computations
    on coordinates.

    Parameters
    ----------
    coordinates : `xarray.Dataset`
        Dataset with at least "latitude" and "longitude"
    latitude : float
        A latitude value to store.
    longitude : float
        A longitude value to store.
    """

    def __init__(self, coordinates=None, /, latitude=None, longitude=None):
        if coordinates is None:
            coordinates = xr.Dataset(data_vars={"latitude": latitude, "longitude": longitude})
        if isinstance(coordinates, Coordinates):
            coordinates = coordinates._data.copy()
        super().__init__(coordinates)

    @property
    def coordinates(self):
        """The latitude and longitude for this coordinate, as `~xarray.Dataset`."""
        return self._data[["latitude", "longitude"]]

    @property
    def latitude(self):
        """The latitude for coordinate, as `~xarray.DataArray`."""
        return self._data["latitude"]

    @property
    def longitude(self):
        """The longitude for coordinate, as `~xarray.DataArray`."""
        return self._data["longitude"]

    def distance_to(self, other):
        """Calculate the distance to another coordinate.

        See `uwacan.positional.distance_to` for details on the implementation.

        Parameters
        ----------
        other : `Coordinates` or convertible
            The coordinate to calculate the distance to.
            If this is an object with ``latitude`` and ``longitude``
            attributes, they will be used. Otherwise, the
            `Position` parser will be called.

        Returns
        -------
        distance : `~xarray.DataArray`
            The distance to the other coordinate.
        """
        other = self._ensure_latlon(other)
        return distance_to(self.latitude, self.longitude, other.latitude, other.longitude)

    def bearing_to(self, other):
        """Calculate the bearing to another coordinate.

        See `uwacan.positional.bearing_to` for details on the implementation.

        Parameters
        ----------
        other : `Coordinates` or convertible
            The coordinate to calculate the bearing to.
            If this is an object with ``latitude`` and ``longitude``
            attributes, they will be used. Otherwise, the
            `Position` parser will be called.

        Returns
        -------
        bearing : `~xarray.DataArray`
            The bearing to the other coordinate.
        """
        other = self._ensure_latlon(other)
        return bearing_to(self.latitude, self.longitude, other.latitude, other.longitude)

    def shift_position(self, distance, bearing):
        """Shift this coordinate by a distance in a certain bearing.

        See `uwacan.positional.shift_position` for details on the implementation.

        Parameters
        ----------
        distance : array_like
            The distance to shift, in meters.
        bearing : array_like
            The bearing to shift towards, in degrees clockwise from true north.

        Returns
        -------
        shifted : cls
            An object of the same class as the one used for shifting,
            with modified latitude and longitude.
        """
        lat, lon = shift_position(self.latitude, self.longitude, distance, bearing)
        data = self.data.assign(latitude=lat, longitude=lon)
        return type(self)(data)

    @classmethod
    def _ensure_latlon(cls, data):
        if not (hasattr(data, "latitude") and hasattr(data, "longitude")):
            # If it doesn't have lat and long we need to construct an object which
            # has them. If we get lists of values we will get a `Position` with lat,lon
            # arrays here. This usually doesn't work, but we only need to access them,
            # which will work fine. The `Position` can handle many of the other
            # useful stuff, like strings and tuples
            data = Position(data)
        return data

    @classmethod
    def from_local_mercator(cls, easting, northing, reference_coordinate, **kwargs):
        r"""Convert local mercator coordinates into wgs84 coordinates.

        See `uwacan.positional.local_mercator_to_wgs84` for details on the implementation.

        Parameters
        ----------
        easting : array_like
            How far east from the reference coordinate, in meters.
        northing : array_like
            How far north from the reference coordinate, in meters.
        reference_coordinate : `Coordinate`
            An object with ``latitude`` and ``longitude`` attributes.
        **kwargs : dict
            Remaining parameters will be passed to the class initializer.
        """
        reference_coordinate = Position(reference_coordinate)
        lat, lon = local_mercator_to_wgs84(
            easting, northing, reference_coordinate.latitude, reference_coordinate.longitude
        )
        return cls(latitude=lat, longitude=lon, **kwargs)

    def to_local_mercator(self, reference_coordinate):
        r"""Convert wgs84 coordinates into a local mercator projection.

        See `uwacan.positional.wgs84_to_local_mercator` for details on the implementation.

        Parameters
        ----------
        reference_coordinate : `Coordinate`
            Object with ``latitude`` and ``longitude`` attributes.
        """
        reference_coordinate = Position(reference_coordinate)
        easting, northing = wgs84_to_local_mercator(
            self.latitude, self.longitude, reference_coordinate.latitude, reference_coordinate.longitude
        )
        return easting, northing

    def local_length_scale(self):
        """How many nautical miles one longitude minute is.

        This gives the apparent length scale for the x-axis in
        mercator projections, i.e., cos(latitude).
        The scaleratio for an x-axis should be set to this value,
        if equal length x- and y-axes are desired, e.g.::

            xaxis=dict(
                title_text='Longitude',
                constrain='domain',
                scaleanchor='y',
                scaleratio=pos.local_length_scale(),
            ),
            yaxis=dict(
                title_text='Latitude',
                constrain='domain',
            ),
        """
        # We take the mean so that it works with subclasses with arrays, e.g., Line.
        return np.cos(np.radians(self.latitude.mean().item()))

    def plot(self, use_minutes=True, include_time=True, name=None, text=None, hover_data=None, **kwargs):
        """Create a plotly trace for the coordinates.

        Parameters
        ----------
        use_minutes : bool, default=True
            Uses degrees and decimal minutes for the latitude and longitude hover.
        include_time : bool, default=True
            Controls if a time value should be included in the hover.
            If a string is passed, it will be used to format the times using strftime syntax.
            Default formatting is ``"%Y-%m-%d %H:%M:%S"``.
        name : str, optional
            The name or label of this trace. Used for legend and hover.
        text : [str], optional
            A list of text labels to show on hover for each point.
        hover_data : dict, optional
            Mapping to add properties to the hover. The keys should match keys in the
            track data. The values are either ``True``, ``False``, or a d3-style formatting
            specification, e.g., ``":.3f"``.
            For data with time types, the formatting string is a strftime specification,
            defaulting to ``"%Y-%m-%d %H:%D:%S"``.
        **kwargs
            All other keywords are passed to `~plotly.graph_objects.Scattermapbox`.
            Useful keywords are:

            - ``mode`` to choose ``"lines"``, ``"markers"``, or ``"lines+markers"``
            - ``line_color`` and ``marker_color``

            Note that the ``hovertemplate``, ``customdata``, ``meta``, ``lat``, ``lon`` keywords will be overwritten.
        """
        import plotly.graph_objects as go

        customdata = []
        meta = []
        hovertemplate = ""

        lat = np.atleast_1d(self.latitude)
        lon = np.atleast_1d(self.longitude)

        if name is not None:
            hovertemplate += name + "<br>"

        if use_minutes:
            ns = np.where(lat > 0, "N", "S")
            latdeg, latmin = np.divmod(np.abs(lat), 1)
            latmin *= 60
            ew = np.where(lon > 0, "E", "W")
            londeg, lonmin = np.divmod(np.abs(lon), 1)
            lonmin *= 60
            customdata.extend([latdeg, latmin, ns, londeg, lonmin, ew])
            # The degrees are floats here, so we format with .0f.
            # Floats can properly handle nan, so it's better to keep them as floats.
            hovertemplate += "%{customdata[0]:.0f}º %{customdata[1]:.3f}' %{customdata[2]} %{customdata[3]:.0f}º %{customdata[4]:.3f}' %{customdata[5]}"
        else:
            hovertemplate += "%{lat:.6f}º %{lon:.6f}º"

        if include_time and hasattr(self, "time"):
            if include_time is True:
                include_time = "%Y-%m-%d %H:%M:%S"
            try:
                time = self.time.dt.strftime(include_time).data
            except AttributeError:
                hovertemplate += f"<br>%{{meta[{len(meta)}]}}"
                meta.append(self.time.data)
            else:
                hovertemplate += f"<br>%{{customdata[{len(customdata)}]}}"
                customdata.append(np.atleast_1d(time))

        hover_data = hover_data or {}
        extra_fields = []
        hovertemplate += "<extra>"
        if text is not None:
            extra_fields.append("%{text}")
        for key, item in hover_data.items():
            if not item:
                continue
            if item is True:
                item = ""
            data = self[key]
            if np.issubdtype(data.dtype, np.datetime64):
                # Pre-format time data into strings.
                item = item or "%Y-%m-%d %H:%M:%S"
                data = data.dt.strftime(item).data
                item = ""
            if data.size == 1:
                extra_fields.append(f"{key}=%{{meta[{len(meta)}]{item}}}")
                meta.append(data)
            else:
                extra_fields.append(f"{key}=%{{customdata[{len(customdata)}]{item}}}")
                customdata.append(data)
        hovertemplate += "<br>".join(extra_fields)
        hovertemplate += "</extra>"

        customdata = np.stack(customdata, axis=1)
        kwargs["lat"] = lat
        kwargs["lon"] = lon
        kwargs["name"] = name
        kwargs["hovertemplate"] = hovertemplate
        kwargs["customdata"] = customdata
        kwargs["text"] = text
        kwargs["meta"] = meta

        trace = go.Scattermapbox(**kwargs)
        return trace


class Position(Coordinates):
    """Class for single positions.

    This class is designed for handling of single positions,
    and includes reasonable operations that can only be performed
    with single coordinates at once.

    This class supports multiple call signatures:

    - ``Position(str)``
        A single string which will be parsed. This handles the
        most common "readable" formats. Important that the string
        has the degree symbol °, minute symbol ', and second symbol ".
        The last two are optional, but minutes and seconds will only
        be parsed if the string includes the relevant symbols.
    - ``Position(pos)``
        A single argument with ``latitude`` and ``longitude`` attributes or keys.
    - ``Position(lat, lon)``
        A pair of two values will be interpreted as the latitude and longitude directly.
    - ``Position(lat_deg, lat_min, lon_deg, lon_min)``
        Four values will be interpreted as latitude and longitude with degrees and minutes.
    - ``Position(lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec)``
        Six values will be interpreted as latitude and longitude with degrees, minutes, and seconds.
    - ``Position(latitude=lat, longitude=lon)``
        Using keyword arguments for latitude and longitude is also supported.
    """

    @staticmethod
    def _parse_coordinates(*args, latitude=None, longitude=None):
        if latitude is not None and longitude is not None:
            return latitude, longitude

        if len(args) == 1:
            arg = args[0]
            try:
                return arg.latitude, arg.longitude
            except AttributeError:
                pass
            try:
                return arg["latitude"], arg["longitude"]
            except (KeyError, TypeError):
                pass
            if isinstance(arg, str):
                matches = re.match(
                    r"""((?P<latdeg>[+\-\d.]+)°?)?((?P<latmin>[\d.]+)')?((?P<latsec>[\d.]+)")?(?P<lathemi>[NS])?"""
                    r"""[,]?"""
                    r"""((?P<londeg>[+\-\d.]+)°?)?((?P<lonmin>[\d.]+)')?((?P<lonsec>[\d.]+)")?(?P<lonhemi>[EW])?""",
                    re.sub(r"\s", "", arg),
                ).groupdict()
                if not matches["latdeg"] or not matches["londeg"]:
                    raise ValueError(f"Cannot parse coordinate string '{arg}'")

                digits_to_parse = len(re.sub(r"\D", "", arg))
                digits_parsed = 0
                latitude = float(matches["latdeg"])
                lat_sign = 1 if latitude >= 0 else -1
                digits_parsed += len(re.sub(r"\D", "", matches["latdeg"]))
                longitude = float(matches["londeg"])
                lon_sign = 1 if longitude >= 0 else -1
                digits_parsed += len(re.sub(r"\D", "", matches["londeg"]))

                if matches["latmin"]:
                    latitude += lat_sign * float(matches["latmin"]) / 60
                    digits_parsed += len(re.sub(r"\D", "", matches["latmin"]))
                if matches["lonmin"]:
                    longitude += lon_sign * float(matches["lonmin"]) / 60
                    digits_parsed += len(re.sub(r"\D", "", matches["lonmin"]))

                if matches["latsec"]:
                    latitude += lat_sign * float(matches["latsec"]) / 3600
                    digits_parsed += len(re.sub(r"\D", "", matches["latsec"]))
                if matches["lonsec"]:
                    longitude += lon_sign * float(matches["lonsec"]) / 3600
                    digits_parsed += len(re.sub(r"\D", "", matches["lonsec"]))

                if not digits_parsed == digits_to_parse:
                    raise ValueError(
                        f"Could not parse coordinate string '{arg}', used only {digits_parsed} of {digits_to_parse} digits"
                    )

                if matches["lathemi"] == "S":
                    latitude = -abs(latitude)
                if matches["lonhemi"] == "W":
                    longitude = -abs(longitude)

                return latitude, longitude

            else:
                # We should never have just a single argument, try unpacking.
                (*args,) = arg

        if len(args) == 2:
            latitude, longitude = args
            return latitude, longitude
        elif len(args) == 4:
            (latitude_degrees, latitude_minutes, longitude_degrees, longitude_minutes) = args
            latitude = latitude_degrees + latitude_minutes / 60
            longitude = longitude_degrees + longitude_minutes / 60
            return latitude, longitude
        elif len(args) == 6:
            (
                latitude_degrees,
                latitude_minutes,
                latitude_seconds,
                longitude_degrees,
                longitude_minutes,
                longitude_seconds,
            ) = args
            latitude = latitude_degrees + latitude_minutes / 60 + latitude_seconds / 3600
            longitude = longitude_degrees + longitude_minutes / 60 + longitude_seconds / 3600
            return latitude, longitude
        else:
            raise TypeError(f"Undefined number of arguments for Position. {len(args)} was given, expects 2, 4, or 6.")

    def __init__(self, *args, latitude=None, longitude=None):
        if len(args) == 1 and isinstance(args[0], (type(self), xr.Dataset)):
            super().__init__(args[0])
        else:
            latitude, longitude = self._parse_coordinates(*args, latitude=latitude, longitude=longitude)
            super().__init__(latitude=latitude, longitude=longitude)

    def __repr__(self):
        return f"{type(self).__name__}({self.latitude.item():.4f}, {self.longitude.item():.4f})"

    def angle_between(self, first, second):
        """Calculate the angle between two positions, as seen from this position.

        The angle between two coordinates is the span that they cover,
        measured positive clockwise and negative anti-clockwise.

        Parameters
        ----------
        first : `~uwacan.positional.Coordinates` or convertable
            The first coordinate.
        second : `~uwacan.positional.Coordinates` or convertable
            The second coordinate.

        Returns
        -------
        angle : float or `xarray.DataArray`
            The angle between the two coordinates.
            This will be a float if both coordinates is a single `Position`,
            otherwise it will be a `DataArray`.
        """
        if not isinstance(first, Coordinates):
            first = Position(first)
        if not isinstance(second, Coordinates):
            second = Position(second)
        return angle_between(
            self.latitude,
            self.longitude,
            first.latitude,
            first.longitude,
            second.latitude,
            second.longitude,
        )

    def make_figure(self, radius, height=1200, mapbox_accesstoken=None, mapbox_style="light"):
        """Make a figure centered on this position.

        Parameters
        ----------
        radius : float
            An approximate radius to show on the figure, in meters.
        height : int, default=1200
            The height of the figure, in pixels.
        mapbox_accesstoken : str
            A mapbox token to use. If none is specified, the environment will be queried
            using ``os.getenv("MAPBOX_ACCESSTOKEN")`` and ``dotenv``.
        mapbox_style : str, default="light"
            A named mapbox style to use.
            Builtin mapbox styles: basic, streets, outdoors, light, dark, satellite, satellite-streets.
            Builtin plotly styles: carto-darkmatter, carto-positron, open-street-map, stamen-terrain, stamen-toner, stamen-watercolor, white-bg.
        """
        bounding_box = BoundingBox.from_center_and_size(
            latitude=self.latitude.item(),
            longitude=self.longitude.item(),
            width=radius * 2,
            height=radius * 2,
        )
        fig = bounding_box.make_figure(height=height, mapbox_accesstoken=mapbox_accesstoken, mapbox_style=mapbox_style)
        return fig


class BoundingBox:
    """Representation of a bounding box.

    Parameters
    ----------
    west : float
        Western edge of the bounding box.
    south : float
        Southern edge of the bounding box.
    east : float
        Eastern edge of the bounding box.
    north : float
        Western edge of the bounding box.
    """

    @classmethod
    def from_center_and_size(cls, latitude, longitude, width, height):
        """Create a bounding box from a center, width, and height."""
        east = shift_position(lat=latitude, lon=longitude, distance=width / 2, bearing=90)[1]
        west = shift_position(lat=latitude, lon=longitude, distance=width / 2, bearing=-90)[1]
        north = shift_position(lat=latitude, lon=longitude, distance=height / 2, bearing=0)[0]
        south = shift_position(lat=latitude, lon=longitude, distance=height / 2, bearing=180)[0]
        return cls(west=west, east=east, south=south, north=north)

    def __init__(self, west, south, east, north):
        self.west = west
        self.south = south
        self.east = east
        self.north = north

    def __repr__(self):
        return f"{type(self).__name__}({self.west}, {self.south}, {self.east}, {self.north})"

    @property
    def north_west(self):
        """The north-west corner of the bounding box."""
        return Position(self.north, self.west)

    @property
    def north_east(self):
        """The north-eest corner of the bounding box."""
        return Position(self.north, self.east)

    @property
    def south_west(self):
        """The south-west corner of the bounding box."""
        return Position(self.south, self.west)

    @property
    def south_east(self):
        """The south-east corner of the bounding box."""
        return Position(self.south, self.east)

    @property
    def center(self):
        """The center of the bounding box."""
        return Position(latitude=(self.north + self.south) / 2, longitude=(self.west + self.east) / 2)

    def __contains__(self, position):
        """Check if a position is within the bounding box.

        Parameters
        ----------
        position : Position
            The position to check.

        Returns
        -------
        bool
            True if the position is within the bounding box, False otherwise.
        """
        position = Position(position)
        if (self.west <= position.longitude <= self.east) and (self.south <= position.latitude <= self.north):
            return True

    def overlaps(self, other):
        """Check if another bounding box overlaps with this one.

        Two bounding boxes are considered overlapping if any of the
        corners in one of the two bounding boxes are within the other one.

        Parameters
        ----------
        other : `BoundingBox`
            Another `BoundingBox` object.

        Returns
        -------
        bool
            True if the bounding boxes overlap, False otherwise.
        """
        return (
            other.north_west in self
            or other.north_east in self
            or other.south_west in self
            or other.south_east in self
            or self.north_west in other
            or self.north_east in other
            or self.south_west in other
            or self.south_east in other
        )

    def zoom_level(self, pixels=800):
        """Calculate a zoom level for the bounding box.

        Parameters
        ----------
        pixels : int, optional
            The number of pixels for the zoom calculation, by default 800.
            The zoom level is calculated so that the bounding box fits on
            this many pixels in height or width.

        Returns
        -------
        float
            The calculated zoom level.

        Notes
        -----
        This is useful when plotting a track on a mapbox map::

            bb = track.bounding_box
            mapbox=dict(
                zoom=bb.zoom_level(),
                center=dict(lat=float(bb.center.latitude), lon=float(bb.center.longitude)),
            )
            fig.update_layout(mapbox=mapbox)
        """
        center = self.center
        westing, northing = self.north_west.to_local_mercator(center)
        easting, southing = self.south_east.to_local_mercator(center)
        extent = max((northing - southing) / center.local_length_scale(), (easting - westing))
        # This has something to do with the size of a tile in pixels (256),
        # the length of the equator (40_000_000), and then some manual scaling
        # to fix the remainder of issues. Worked nice in plotly 5.18, calling mapbox.
        zoom = np.log2(40_000_000 * pixels / 256 / extent).item() - 1.2
        return zoom

    def make_figure(self, height=1200, mapbox_accesstoken=None, mapbox_style="light"):
        """Create a plotly figure with mapbox settings.

        Parameters
        ----------
        height : int, default=1200
            Figure height in pixels.
        mapbox_accesstoken : str
            A mapbox token to use. If none is specified, the environment will be queried
            using ``os.getenv("MAPBOX_ACCESSTOKEN")`` and ``dotenv``.
        mapbox_style : str, default="light"
            A named mapbox style to use.
            Builtin mapbox styles: basic, streets, outdoors, light, dark, satellite, satellite-streets.
            Builtin plotly styles: carto-darkmatter, carto-positron, open-street-map, stamen-terrain, stamen-toner, stamen-watercolor, white-bg.
        """
        import plotly.graph_objects as go

        if mapbox_accesstoken is None:
            import os

            mapbox_accesstoken = os.getenv("MAPBOX_ACCESSTOKEN")
            if mapbox_accesstoken is None:
                import dotenv

                mapbox_accesstoken = dotenv.get_key(dotenv.find_dotenv(), "MAPBOX_ACCESSTOKEN")
        return go.Figure(
            layout=dict(
                mapbox=dict(
                    accesstoken=mapbox_accesstoken,
                    style=mapbox_style,
                    zoom=self.zoom_level(height),
                    center=dict(lat=float(self.center.latitude), lon=float(self.center.longitude)),
                ),
                height=height,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )
        )


class Positions(Coordinates):
    """Container for arrays of coordinates."""

    def __repr__(self):
        return f"{type(self).__name__} with {self.latitude.size} points"

    @property
    def bounding_box(self):
        """A `BoundingBox` for these coordinates."""
        try:
            return self._bounding_box
        except AttributeError:
            pass
        west = self.longitude.min().item()
        east = self.longitude.max().item()
        north = self.latitude.max().item()
        south = self.latitude.min().item()
        self._bounding_box = BoundingBox(west=west, south=south, east=east, north=north)
        return self._bounding_box

    @functools.wraps(BoundingBox.make_figure)
    def make_figure(self, *args, **kwargs):  # noqa: D102, we wrap to get the docs
        return self.bounding_box.make_figure(*args, **kwargs)

    @classmethod
    def concatenate(cls, parts, dim=None, nan_between_parts=False, **kwargs):
        """Concatenates multiple positions.

        Parameters
        ----------
        parts : iterable of Positions
            `Position` objects to concatenate.
        dim : str, optional
            The dimension along which to concatenate the parts. If None, it tries to infer the dimension
            from the first part's coordinates. If the dimension is not provided and the parts have multiple
            dimensions, a ``ValueError`` is raised.
        nan_between_parts : bool, default=False
            If True, inserts a ``NaN`` element between each part. This is useful for
            visualization purposes, as it makes most plotting libraries split the lines.
        **kwargs : dict, optional
            Additional keyword arguments passed to the class constructor.

        Returns
        -------
        positions
            A new instance of the `cls` class with the concatenated parts.

        Raises
        ------
        ValueError
            If the dimension cannot be inferred and the lines have multiple dimensions.
        """
        first_coords = parts[0].coordinates
        if dim is None:
            if len(first_coords.dims) != 1:
                raise ValueError("Cannot guess concatenation dimensions for multi-dimensional line.")
            dim = next(iter(first_coords.dims))

        parts = [part.data if isinstance(part, _core.xrwrap) else part for part in parts]
        if nan_between_parts:
            # We make nan_data with only lat and lon variables. They are guaranteed to be floats.
            # The rest of the variables will be filled with nan by xr.concat anyhow.
            nan_data = xr.full_like(first_coords.isel({dim: 0}), np.nan).expand_dims(dim)
            parts = sum([[part, nan_data] for part in parts], [])
        data = xr.concat(parts, dim=dim)
        return cls(data, **kwargs)


class Line(Positions):
    """A simple line of coordinates."""

    @classmethod
    def at_position(cls, position, length, bearing, n_points=100, symmetric=False, dim="line"):
        """Create a line at a given position with a specified length and bearing.

        Parameters
        ----------
        position : `Position`
            The starting position of the line.
        length : float
            The total length of the line, in meters.
        bearing : float
            The bearing (direction) of the line in degrees.
        n_points : int, default=100
            The number of points to generate along the line.
        symmetric : bool, default=False
            If True, the line is centered on the given position.
        dim : str, default="line"
            The dimension name for the line points.

        Returns
        -------
        Line
            A new instance of the `Line` class representing the line at the specified position.
        """
        if symmetric:
            n_points += (n_points + 1) % 2
            distance = np.linspace(-length / 2, length / 2, n_points)
        else:
            distance = np.linspace(0, length, n_points)

        distance = xr.DataArray(distance, dims=dim)
        position = Position(position)
        lat, lon = shift_position(position.latitude, position.longitude, distance, bearing)
        return cls(latitude=lat, longitude=lon)


class Track(Positions):
    """A class representing a GPS track, which is a sequence of coordinates over time.

    Typically instances of this class are created by reading a data file, using the
    implemented classmethods.

    Parameters
    ----------
    data : xarray.Dataset
        The dataset containing the track data.
    calculate_course : bool, optional
        Whether to calculate the course of the track, by default False.
    calculate_speed : bool, optional
        Whether to calculate the speed of the track, by default False.
    """

    @classmethod
    def read_nmea_gps(cls, filepath, **kwargs):
        """Read a GPS track from an NMEA file.

        Parameters
        ----------
        filepath : str or Path
            The path to the NMEA file.
        **kwargs : dict, optional
            Additional keyword arguments passed to the class constructor.

        Returns
        -------
        Track
            A `Track` object containing the GPS data.

        Notes
        -----
        An NMEA file is a plain text file, with one NMEA message per line.
        These NMEA messages look something like::

            $GPRMC,102100.000,A,5741.4478,N,01152.4111,E,0.00,319.61,300823,,,A*60

        If you have a different header (GPRMC), it's likely that this read function will not work properly.
        """
        latitudes = []
        longitudes = []
        headings = []
        speeds = []
        times = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip("\"'\n")
                (
                    header,
                    utc,
                    status,
                    lat,
                    lat_dir,
                    lon,
                    lon_dir,
                    speed,
                    heading,
                    date,
                    mag_var,
                    mag_var_dir,
                    mode_chsum,
                ) = line.split(",")
                times.append(_core.time_to_np(date + utc, fmt="%d%m%y%H%M%S.%f"))
                latitudes.append((int(lat[:2]) + float(lat[2:]) / 60) * (1 if lat_dir == "N" else -1))
                longitudes.append((int(lon[:3]) + float(lon[3:]) / 60) * (1 if lon_dir == "E" else -1))
                headings.append(float(heading))
                speeds.append(float(speed))
        track = xr.Dataset(
            data_vars=dict(
                latitude=xr.DataArray(latitudes, coords={"time": times}),
                longitude=xr.DataArray(longitudes, coords={"time": times}),
                heading=xr.DataArray(headings, coords={"time": times}),
                speed=xr.DataArray(speeds, coords={"time": times}, attrs={"unit": "knots"}),
            )
        ).drop_duplicates("time")
        return cls(track, **kwargs)

    @classmethod
    def read_blueflow(cls, filepath, renames=None, **kwargs):
        """Read a track from a BlueFlow file (Excel or CSV).

        Parameters
        ----------
        filepath : str or Path
            The path to the BlueFlow file.
        renames : dict, optional
            A dictionary for renaming columns, by default None.
        **kwargs : dict, optional
            Additional keyword arguments passed to the class constructor.

        Returns
        -------
        Track
            A `Track` object containing the BlueFlow data.

        """
        import pandas

        filepath = Path(filepath)
        if filepath.suffix == ".xlsx":
            data = pandas.read_excel(filepath.as_posix())
        elif filepath.suffix == ".csv":
            data = pandas.read_csv(filepath.as_posix())
        else:
            raise ValueError(f"Unknown fileformat for blueflow file '{filepath}'. Only xlsx and csv supported.")
        data = data.to_xarray()
        names = {}
        exp = r"([^\(\)\[\]]*) [\[\(]([^\(\)\[\]]*)[\]\)]"
        for key in list(data):
            name, unit = re.match(exp, key).groups()
            names[key] = name.strip()
            data[key].attrs["unit"] = unit
        data = data.rename(names)

        renames = {
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Timestamp": "time",
            "Time": "time",
            "Tidpunkt": "time",
            "Latitud": "latitude",
            "Longitud": "longitude",
        } | (renames or {})

        renames = {key: value for key, value in renames.items() if key in data}
        data = data.rename(renames).set_coords("time").swap_dims(index="time").drop("index")
        if not np.issubdtype(data.time.dtype, np.datetime64):
            data["time"] = xr.apply_ufunc(np.datetime64, data.time, vectorize=True, keep_attrs=True)
        return cls(data, **kwargs)

    @classmethod
    def read_gpx(cls, filepath, **kwargs):
        """Read a GPS track from a GPX file.

        Parameters
        ----------
        filepath : str or Path
            The path to the GPX file.
        **kwargs : dict, optional
            Additional keyword arguments passed to the class constructor.

        Returns
        -------
        Track
            A `Track` object containing the GPX data.s
        """
        try:
            import gpxpy
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Module 'gpxpy' required to read gpx data")
        with open(filepath, "r") as f:
            contents = gpxpy.parse(f)
        latitudes = []
        longitudes = []
        times = []
        for point in contents.get_points_data():
            latitudes.append(point.point.latitude)
            longitudes.append(point.point.longitude)
            time = np.datetime64(int(point.point.time.timestamp() * 1e9), "ns")
            times.append(time)
        times = np.asarray(times)
        data = xr.Dataset(
            data_vars=dict(
                latitude=xr.DataArray(latitudes, coords={"time": times}),
                longitude=xr.DataArray(longitudes, coords={"time": times}),
            )
        )
        return cls(data, **kwargs)

    def __init__(self, data, calculate_course=False, calculate_speed=False):
        super().__init__(data)
        if calculate_course:
            self["course"] = self.calculate_course()
        if calculate_speed:
            self["speed"] = self.calculate_speed()

    @property
    def time(self):
        """The time coordinates for the track."""
        return self._data["time"]

    def calculate_course(self):
        """Calculate the course of the track."""
        coords = self.coordinates
        before = coords.shift(time=1).dropna("time")
        after = coords.shift(time=-1).dropna("time")
        interior_course = bearing_to(
            before.latitude,
            before.longitude,
            after.latitude,
            after.longitude,
        )
        first_course = bearing_to(
            coords.isel(time=0).latitude,
            coords.isel(time=1).longitude,
            coords.isel(time=1).latitude,
            coords.isel(time=1).longitude,
        ).assign_coords(time=coords.time[0])
        last_course = bearing_to(
            coords.isel(time=-2).latitude,
            coords.isel(time=-2).longitude,
            coords.isel(time=-1).latitude,
            coords.isel(time=-1).longitude,
        ).assign_coords(time=coords.time[-1])
        course = xr.concat([first_course, interior_course, last_course], dim="time")
        return course

    def calculate_speed(self):
        """Calculate the speed of the track."""
        coords = self.coordinates
        before = coords.shift(time=1).dropna("time")
        after = coords.shift(time=-1).dropna("time")

        distance_delta = distance_to(before.latitude, before.longitude, after.latitude, after.longitude)
        # We cannot reuse the previous shift here, since the time coordinate is not shifted there
        time_delta = (
            coords.time.shift(time=-1).dropna("time") - coords.time.shift(time=1).dropna("time")
        ) / np.timedelta64(1, "s")
        interior_speed = distance_delta / time_delta

        first_distance = distance_to(
            coords.isel(time=0).latitude,
            coords.isel(time=1).longitude,
            coords.isel(time=1).latitude,
            coords.isel(time=1).longitude,
        )
        first_time = (coords.time[1] - coords.time[0]) / np.timedelta64(1, "s")
        first_speed = (first_distance / first_time).assign_coords(time=coords.time[0])

        last_distance = distance_to(
            coords.isel(time=-2).latitude,
            coords.isel(time=-2).longitude,
            coords.isel(time=-1).latitude,
            coords.isel(time=-1).longitude,
        )
        last_time = (coords.time[-1] - coords.time[-2]) / np.timedelta64(1, "s")
        last_speed = (last_distance / last_time).assign_coords(time=coords.time[-1])
        speed = xr.concat([first_speed, interior_speed, last_speed], dim="time")
        speed.attrs["unit"] = "m/s"
        return speed

    def average_course(self, resolution=None):
        """Calculate the average course in the track.

        The average course is the average angle of the course,
        calculated to handle wrapping.
        See `average_angle` for more details.

        Parameters
        ----------
        resolution : str, int, optional
            If a numerical resolution is given, the output will be
            rounded to this many parts of a turn.
            If a string "four", "eight", or "sixteen", is given,
            the output is a string with the closest named bearing.
        """
        try:
            course = self["course"]
        except KeyError:
            course = self.calculate_course()
        return average_angle(course, resolution=resolution)

    def closest_point(self, other):
        """Get the point in this track closest to a position.

        Parameters
        ----------
        other : `Position`
            The point to compute distances to.

        Returns
        -------
        `Position`
            The closest position in this track to the input.
            Includes latitude, longitude, time, and distance.
        """
        distances = self.distance_to(other)
        idx = distances.argmin(...)
        return Position(self._data.isel(idx).assign(distance=distances.isel(idx)))

    def aspect_segments(
        self,
        reference,
        angles,
        segment_min_length=None,
        segment_min_angle=None,
        segment_min_duration=None,
    ):
        """Get time segments corresponding to specific aspect angles.

        Aspect angles are measured between the reference and cpa.

        Parameters
        ----------
        reference : Position
            Reference position from which to measure cpa.
        angles : array_like
            The aspect angles to find.
        segment_min_length : numeric, optional
            The minimum spatial extent of each segment, in meters.
        segment_min_angle : numeric, optional
            The minimum angular extent of each segment, in degrees.
        segment_min_duration : numeric, optional
            The minimum temporal extent of each window, in seconds.

        Returns
        -------
        segments : xarray.Dataset
            A dataset with coordinates:
                - segment : the angles specified to center the segments around
                - edge : ["start", "center", "stop"], indicating the start, stop, and center of segment
            and data variables:
                - latitude (segment, edge) : the latitudes for the segment
                - longitude (segment, edge) : the longitudes for the segment
                - time (segment, edge) : the times for the segment
                - aspect_angle (segment, edge) : the actual aspect angles for the segment
                - length (segment) : the spatial extent of each segment, in m
                - angle_span (segment) : the angular extent of each segment, in degrees
                - duration (segment) : the temporal extent of each segment, in seconds
        """
        track = self.coordinates
        cpa = self.closest_point(reference)

        try:
            iter(angles)
        except TypeError:
            single_segment = True
            angles = [angles]
        else:
            single_segment = False

        angles = np.sort(angles)
        track = track.assign(aspect_angle=reference.angle_between(cpa, track))
        if track.aspect_angle[0] > track.aspect_angle[-1]:
            # We want the angles to be negative before cpa and positive after
            track["aspect_angle"] *= -1

        angles = xr.DataArray(angles, coords={"segment": angles})
        center_indices = abs(angles - track.aspect_angle).argmin("time")
        segment_centers = track.isel(time=center_indices)

        # Run a check that we get the windows we want. A sane way might be to check that the
        # first and last windows are closer to their targets than the next window.
        if angles.size > 1:
            actual_first_angle = track.aspect_angle.sel(time=segment_centers.isel(segment=0).time)
            if abs(actual_first_angle - angles.isel(segment=0)) > abs(actual_first_angle - angles.isel(segment=1)):
                raise ValueError(
                    f"Could not find window centered at {angles.isel(segment=0):.1f}⁰, found at most {actual_first_angle:.1f}⁰."
                )
            actual_last_angle = track.aspect_angle.sel(time=segment_centers.isel(segment=-1).time)
            if abs(actual_last_angle - angles.isel(segment=-1)) > abs(actual_last_angle - angles.isel(segment=-2)):
                raise ValueError(
                    f"Could not find window centered at {angles.isel(segment=-1):.1f}⁰, found at most {actual_last_angle:.1f}⁰."
                )

        segments = []
        track_angles = track.aspect_angle.data
        track_lat = track.latitude.data
        track_lon = track.longitude.data
        track_time = track.time.data
        for angle, segment_center in segment_centers.groupby("segment", squeeze=False):
            segment_center = segment_center.squeeze()
            # Finding the start of the window
            # The inner loops here are somewhat slow, likely due to indexing into the xr.Dataset all the time
            # At the time of writing (2023-12-14), there seems to be no way to iterate over a dataset in reverse order.
            # The `groupby` method can be used to iterate forwards, which solves finding the end of the segment,
            # but calling `track.sel(time=slice(t, None, -1)).groupby('time')` still iterates in the forward order.
            center_idx = int(np.abs(track.time - segment_center.time).argmin())
            start_idx = center_idx
            if segment_min_angle:
                while abs(segment_center.aspect_angle - track_angles[start_idx]) < segment_min_angle / 2:
                    start_idx -= 1
                    if start_idx < 0:
                        raise ValueError(
                            f"Start of window at {angle}⁰ not found in track. Not sufficiently high angles from window center."
                        )
            if segment_min_duration:
                while (
                    abs(segment_center.time - track_time[start_idx]) / np.timedelta64(1, "s") < segment_min_duration / 2
                ):
                    start_idx -= 1
                    if start_idx < 0:
                        raise ValueError(
                            f"Start of window at {angle}⁰ not found in track. Not sufficient time from window center."
                        )
            if segment_min_length:
                while (
                    distance_to(
                        segment_center.latitude, segment_center.longitude, track_lat[start_idx], track_lon[start_idx]
                    )
                    < segment_min_length / 2
                ):
                    start_idx -= 1
                    if start_idx < 0:
                        raise ValueError(
                            f"Start of window at {angle}⁰ not found in track. Not sufficient distance from window center."
                        )
            # Finding the end of the window
            stop_idx = center_idx
            if segment_min_angle:
                while abs(segment_center.aspect_angle - track_angles[stop_idx]) < segment_min_angle / 2:
                    stop_idx += 1
                    if stop_idx == track.sizes["time"]:
                        raise ValueError(
                            f"End of window at {angle}⁰ not found in track. Not sufficiently high angles from window center."
                        )
            if segment_min_duration:
                while (
                    abs(segment_center.time - track_time[stop_idx]) / np.timedelta64(1, "s") < segment_min_duration / 2
                ):
                    stop_idx += 1
                    if stop_idx == track.sizes["time"]:
                        raise ValueError(
                            f"End of window at {angle}⁰ not found in track. Not sufficient time from window center."
                        )
            if segment_min_length:
                while (
                    distance_to(
                        segment_center.latitude, segment_center.longitude, track_lat[stop_idx], track_lon[stop_idx]
                    )
                    < segment_min_length / 2
                ):
                    stop_idx += 1
                    if stop_idx == track.sizes["time"]:
                        raise ValueError(
                            f"End of window at {angle}⁰ not found in track. Not sufficient distance from window center."
                        )

            # Creating the window and saving some attributes
            if start_idx == stop_idx:
                segments.append(segment_center.assign(length=0, angle_span=0, duration=0).reset_coords("time"))
            else:
                segment_start, segment_stop = track.isel(time=start_idx), track.isel(time=stop_idx)
                segments.append(
                    xr.concat([segment_start, segment_center, segment_stop], dim="time")
                    .assign_coords(edge=("time", ["start", "center", "stop"]))
                    .swap_dims(time="edge")
                    .assign(
                        length=distance_to(
                            segment_start.latitude,
                            segment_start.longitude,
                            segment_stop.latitude,
                            segment_stop.longitude,
                        ),
                        angle_span=segment_stop.aspect_angle - segment_start.aspect_angle,
                        duration=(segment_stop.time - segment_start.time) / np.timedelta64(1, "s"),
                    )
                    .reset_coords("time")
                )

        if single_segment:
            return segments[0]
        return xr.concat(segments, dim="segment")

    @property
    def time_window(self):
        """A `~uwacan.TimeWindow` describing when the data start and stops."""
        return _core.TimeWindow(start=self.time[0], stop=self.time[-1])

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        """Select a subset of the data over time.

        See `uwacan.TimeWindow.subwindow` for details on the parameters.
        """
        new_window = self.time_window.subwindow(
            time, start=start, stop=stop, center=center, duration=duration, extend=extend
        )
        if isinstance(new_window, _core.TimeWindow):
            start = new_window.start.in_tz("UTC").naive()
            stop = new_window.stop.in_tz("UTC").naive()
            return type(self)(self._data.sel(time=slice(start, stop)))
        else:
            return self._data.sel(time=new_window.in_tz("UTC").naive(), method="nearest")

    def resample(self, time, /, **kwargs):
        """Resample the Track at specific times or rate.

        Parameters
        ----------
        time : float or xr.DataArray
            If an `xr.DataArray` is provided, it represents the new time points to which the data will be resampled.
            If a float is provided, it represents the frequency in Hz at which to resample the data.
        **kwargs : dict
            Additional keyword arguments passed to the `xr.DataArray.interp` method for interpolation.

        Returns
        -------
        type(self)
            A new instance of the same type as ``self``, with the data resampled at the specified time points.
        """
        if not isinstance(time, xr.DataArray):
            n_samples = int(np.floor(self.time_window.duration * time))
            start_time = _core.time_to_np(self.time_window.start)
            offsets = np.arange(n_samples) * 1e9 / time
            time = start_time + offsets.astype("timedelta64[ns]")
        data = self._data.interp(time=time, **kwargs)
        new = type(self)(data)
        return new

    def correct_gps_offset(
        self, forwards=0, portwards=0, to_bow=0, to_stern=0, to_port=0, to_starboard=0, heading=None
    ):
        """Correct positions with respect to ship heading and particulars.

        The ``to_x`` parameters is the distances from the gps antenna to the ship sides.
        The ``forwards`` and ``portwards`` parameters are how much forward and to port the
        new positions should be, relative to the center of the ship.

        Parameters
        ----------
        forwards : numeric, default 0
            How much forwards to shift the positions, in meters
        portwards : numeric, default 0
            How much to port side to shift the positions, in meters
        to_bow : numeric, default 0
            The distance to the bow from the receiver, in meters
        to_stern : numeric, default 0
            The distance to the stern from the receiver, in meters
        to_port : numeric, default 0
            The distance to the port side from the receiver, in meters
        to_starboard : numeric, default 0
            The distance to the starboard side from the receiver, in meters
        heading : array_like, optional
            The headings of the ship. Must be compatible with the coordinates in this Track
            If None, the Track checks for a "heading", then a "course" in the contained data.

        Notes
        -----
        The positions will be shifted in the ``heading`` direction by ``forwards + (to_bow - to_stern) / 2``,
        and towards "port" ``heading - 90`` by ``portwards + (to_port - to_starboard) / 2``.
        Typical usage is to give the receiver position using the ``to_x`` arguments, and the desired
        acoustic reference location with the ``forwards`` and ``portwards`` arguments.
        Inserting correct values for all the ``to_x`` arguments will center the position on the ship middle, so that
        the ``forwards`` and ``portwards`` arguments are relative to the ship center. Alternatively, leave the ``to_x`` arguments
        as the default 0 and only give the desired ``forwards`` and ``portwards`` arguments.
        """
        forwards = forwards + (to_bow - to_stern) / 2
        portwards = portwards + (to_port - to_starboard) / 2
        if heading is None:
            try:
                heading = self.heading
            except AttributeError:
                heading = self.course
        new = self.shift_position(distance=forwards, bearing=heading)
        new = new.shift_position(distance=portwards, bearing=heading - 90)
        return new


def sensor(sensor, /, sensitivity=None, depth=None, position=None, latitude=None, longitude=None):
    """Collect sensor information.

    Typical sensor information is the position, sensitivity, and deployment depth.
    The position can be given as a string, a tuple, or separate longitude and latitudes.
    This factory function will return a `~uwacan.positional.Sensor` or
    `~uwacan.positional.SensorPosition` depending on if a position is provided.

    Parameters
    ----------
    sensor : str
        The label for the sensor. All sensors must have a label.
    sensitivity : float
        The sensor sensitivity, in dB re. V/Q,
        where Q is the desired physical unit.
    depth : float
        Sensor deployment depth.
    position : str or tuple
        A coordinate string, or a tuple with lat, lon information.
    latitude : float
        The latitude the sensor was deployed at.
    longitude : float
        The longitude the sensor was deployed at.

    Returns
    -------
    Sensor or SensorPosition
    """
    if isinstance(sensor, Sensor):
        sensor = sensor._data
    if isinstance(sensor, xr.Dataset):
        sensor = sensor[[key for key, value in sensor.notnull().items() if value]]
        if "latitude" in sensor and "longitude" in sensor:
            obj = SensorPosition(sensor)
        else:
            obj = Sensor(sensor)
    else:
        if position is not None or (latitude is not None and longitude is not None):
            obj = SensorPosition(position, latitude=latitude, longitude=longitude)
        else:
            obj = Sensor(xr.Dataset())
        obj._data.coords["sensor"] = sensor

    if "sensor" not in obj:
        raise ValueError("Cannot have unlabeled sensors")
    if sensitivity is not None:
        obj._data["sensitivity"] = sensitivity
    if depth is not None:
        obj._data["depth"] = depth
    return obj


class Sensor(_core.DatasetWrap):
    """Container for sensor information.

    This class is typically not instantiated directly,
    but through the `~uwacan.sensor` factory function.
    """

    def __repr__(self):
        sens = "" if "sensitivity" not in self._data else f", sensitivity={self['sensitivity']:.2f}"
        depth = "" if "depth" not in self._data else f", depth={self['depth']:.2f}"
        return f"Sensor({self.label}{sens}{depth})"

    @property
    def label(self):
        """The label of this sensor."""
        return self._data["sensor"].item()

    def __and__(self, other):
        if not isinstance(other, (Sensor, xr.Dataset)):
            return NotImplemented
        return SensorArray.concatenate([self, other])

    def with_data(self, **kwargs):
        """Create a copy of this sensor with additional information.

        This method allows you to add new properties to an existing sensor.
        It's particularly useful for adding sensor-specific information
        such as gain or recording channel data that is not part of the core sensor
        class.

        Parameters
        ----------
        **kwargs : dict
            Additional data to add to the sensor. Each keyword argument
            corresponds to a new data variable with one of the following input types:

            - `~xarrayr.DataArray`: Must have a "sensor" dimension matching
              the existing sensor coordinate.
            - dict: Keys should be the sensor names, and the dictionary must
              contain all sensors in the existing data.
            - scalar: A single scalar value that will be assigned to all sensors.
            - array_like: An array with a length matching the "sensor" dimension.

        """
        data = self._data.copy()
        if "sensor" not in data.dims:
            data = data.expand_dims("sensor")
        for key, value in kwargs.items():
            if isinstance(value, xr.DataArray):
                if "sensor" not in value.dims:
                    raise ValueError("Cannot add xarray data without sensor dimension to sensors")
                data[key] = value
            elif isinstance(value, dict):
                data[key] = xr.DataArray(
                    [value[key] for key in data["sensor"].values], coords={"sensor": data["sensor"]}
                )
            elif np.size(value) == 1:
                data[key] = np.squeeze(value)
            elif np.size(value) != data["sensor"].size:
                raise ValueError(f"Cannot assign {np.size(value)} values to {data['sensor'].size} sensors")
            else:
                data[key] = xr.DataArray(value, coords={"sensor": data["sensor"]})
        return type(self)(data.squeeze())


class SensorPosition(Sensor, Position):
    """Container for sensor information, including position.

    This class is typically not instantiated directly,
    but through the `~uwacan.sensor` factory function.
    """

    def __repr__(self):
        sens = "" if "sensitivity" not in self._data else f", sensitivity={self['sensitivity']:.2f}"
        depth = "" if "depth" not in self._data else f", depth={self['depth']:.2f}"
        return f"Sensor({self.label}, latitude={self.latitude:.4f}, longitude={self.longitude:.4f}{sens}{depth})"


class SensorArray(Sensor):
    """Container for sensor information.

    This class is typically not instantiated directly,
    but through the `~uwacan.sensor_array` factory function.

    A collection of sensors can be added with another `Sensor` or
    `SensorArray`.
    """

    @property
    def sensors(self):
        """The stored `Sensor` objects, as a dict."""
        return {label: sensor(data.squeeze()) for label, data in self._data.groupby("sensor", squeeze=False)}

    def __repr__(self):
        return f"SensorArray with {self._data['sensor'].size} sensors"

    @property
    def label(self):
        """The labels of the sensors."""
        return tuple(self._data["sensor"].data)

    @classmethod
    def concatenate(cls, sensors):
        """Collect sensor information from multiple sensors.

        Parameters
        ----------
        sensors : iterable of (Sensor or `xarray.Dataset`)
            The sensors to concatenate, as single or multiple arguments.
            Each item has to be either a `~uwacan.positional.Sensor` or
            an `xarray.Dataset` with sensor information.

        Returns
        -------
        sensors
            One of

            - `~uwacan.positional.SensorArray` if no position information is given,
            - `~uwacan.positional.LocatedSensorArray` if position information is different for the sensors,
            - `~uwacan.positional.ColocatedSensorArray` if the position information is the same for all sensors.

        """
        if isinstance(sensors, Sensor):
            sensors = [sensors.data]
        elif isinstance(sensors, xr.Dataset):
            sensors = [sensors]

        sensors = [item.data if isinstance(item, Sensor) else item for item in sensors]
        sensors = xr.concat(sensors, dim="sensor")
        for key, value in sensors.items():
            if np.all(np.equal(value.values[0], value.values)):
                sensors[key] = value.mean()
        if "latitude" in sensors and "longitude" in sensors:
            if sensors["latitude"].size == 1 and sensors["longitude"].size == 1:
                obj = ColocatedSensorArray(sensors)
            else:
                obj = LocatedSensorArray(sensors)
        else:
            obj = SensorArray(sensors)
        return obj


class LocatedSensorArray(SensorArray, Positions):
    """Container for sensor information, including positions.

    This class is typically not instantiated directly,
    but through the `~uwacan.sensor_array` factory function.
    """

    def __init__(self, data):
        SensorArray.__init__(self, data)


class ColocatedSensorArray(SensorArray, Position):
    """Container for sensor information, with a single position for all.

    This class is typically not instantiated directly,
    but through the `~uwacan.sensor_array` factory function.
    """

    def __init__(self, data):
        SensorArray.__init__(self, data)

    def __repr__(self):
        return f"SensorArray with {self._data['sensor'].size} sensors at ({self.latitude:.4f}, {self.longitude:.4f})"
