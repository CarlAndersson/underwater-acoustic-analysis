import collections.abc
import re
import numpy as np
import xarray as xr
import pendulum
from . import _core


_WGS84_equatorial_radius = 6_378_137.0
_WGS84_polar_radius = 6_356_752.3
_mercator_scale_factor = 0.9996


def nm_to_m(nm):
    """Convert nautical miles to meters"""
    return nm * 1852


def m_to_nm(m):
    """Convert meters to nautical miles"""
    return m / 1852


def mps_to_knots(mps):
    """Convert meters per second to knots"""
    return mps * (3600 / 1852)


def knots_to_mps(knots):
    """Convert knots to meters per second"""
    return knots * (1852 / 3600)


def knots_to_kmph(knots):
    """Convert knots to kilometers per hour"""
    return knots * 1.852


def kmph_to_knots(kmph):
    """Convert kilometers per hour to knots"""
    return kmph / 1.852


def wrap_angle(degrees):
    '''Wrap an angle to (-180, 180].'''
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


_re_rp_2 = (_WGS84_equatorial_radius / _WGS84_polar_radius)**2
def _geodetic_to_geocentric(lat):
    r"""Compute the geocentric latitude from geodetic, in radians.

    The geocentric latitude is the latitude as seen from the center
    of the earth. The geodetic latitude is the angle formed with the
    equatorial plane when drawing the normal at a surface on the earth.

    The conversion for the geocentric latitude :math:`\hat\varphi` is

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
    r"""Computes the earth radius at a given latitude, in radians.

    The formula is

    .. math::
        R( φ) = \sqrt{\frac{(a^2\cos φ)^2+(b^2\sin φ)^2}{(a\cos φ)^2+(b\sin φ)^2}}

    with the geodetic latitude :math:`φ`, and the equatorial and polar earth radii
    :math:`a, b` respectively, see https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii.
    """
    return (
        ((_WGS84_equatorial_radius**2 * np.cos(lat))**2 + (_WGS84_polar_radius**2 * np.sin(lat))**2)
        /
        ((_WGS84_equatorial_radius * np.cos(lat))**2 + (_WGS84_polar_radius * np.sin(lat))**2)
    )**0.5


def _haversine(theta):
    r"""Computes the haversine of an angle, in radians.

    This is the same as

    .. math:: \sin^2(θ/2)
    """
    return np.sin(theta / 2)**2


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
    central_angle = _haversine(lat_2 - lat_1) + (1 - _haversine(lat_1 - lat_2) - _haversine(lat_1 + lat_2)) * _haversine(lon_2 - lon_1)
    d = 2 * r * np.arcsin(central_angle ** 0.5)
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
    new_lon = lon + np.arctan2(np.sin(bearing) * np.sin(dist) * np.cos(lat), np.cos(dist) - np.sin(lat) * np.sin(new_lat))
    new_lat = _geocentric_to_geodetic(new_lat)
    return np.degrees(new_lat), np.degrees(new_lon)


def average_angle(angle, resolution=None):
    """Calculates the average angle from a list of angles and optionally rounds it to a specified resolution.

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
    if '4' in resolution or 'four' in resolution:
        resolution = 4
    elif '8' in resolution or 'eight' in resolution:
        resolution = 8
    elif '16' in resolution or 'sixteen' in resolution:
        resolution = 16
    else:
        raise ValueError(f"Unknown resolution specifier '{resolution}'")

    names = [
        (-180., 'south'),
        (-90., 'west'),
        (0., 'north'),
        (90., 'east'),
        (180., 'south'),
    ]

    if resolution >= 8:
        names.extend([
            (-135., 'southwest'),
            (-45., 'northwest'),
            (45., 'northeast'),
            (135., 'southeast'),
        ])
    if resolution >= 16:
        names.extend([
            (-157.5, 'south-southwest'),
            (-112.5, 'west-southwest'),
            (-67.5, 'west-northwest'),
            (-22.5, 'north-northwest'),
            (22.5, 'north-northeast'),
            (67.5, 'east-northeast'),
            (112.5, 'east-southeast'),
            (157.5, 'south-southeast'),
        ])
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


class _Coordinates(_core.DatasetWrap):
    def __init__(self, coordinates=None, /, latitude=None, longitude=None):
        if coordinates is None:
            coordinates = xr.Dataset(data_vars={"latitude": latitude, "longitude": longitude})
        if isinstance(coordinates, _Coordinates):
            coordinates = coordinates._data.copy()
        super().__init__(coordinates)

    @property
    def coordinates(self):
        return self._data[["latitude", "longitude"]]

    @property
    def latitude(self):
        return self._data['latitude']

    @property
    def longitude(self):
        return self._data['longitude']

    def distance_to(self, other):
        """Calculates the distance to another coordinate."""
        other = self._ensure_latlon(other)
        return distance_to(self.latitude, self.longitude, other.latitude, other.longitude)

    def bearing_to(self, other):
        """Calculates the bearing to another coordinate."""
        other = self._ensure_latlon(other)
        return bearing_to(self.latitude, self.longitude, other.latitude, other.longitude)

    def shift_position(self, distance, bearing):
        """Shifts this position by a distance in a certain bearing"""
        lat, lon = shift_position(self.latitude, self.longitude, distance, bearing)
        return type(self)(latitude=lat, longitude=lon)

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

        Conventions here are :math:`\lambda` as the longitude and :math:`\varphi` as the latitude,
        :math:`x` is easting and :math:`y` is northing.

        .. math::

            \lambda &= \lambda_0 + \frac{x}{R} \\
            \varphi &= 2\arctan\left[\exp \left(\frac{y + y_0}{R}\right)\right] - \frac{\pi}{2}

        The northing offset :math:`y_0` is computed by converting the reference point into
        a mercator projection with the `wgs84_to_local_mercator` function, using (0, 0) as
        the reference coordinates.
        """
        reference_coordinate = Position(reference_coordinate)
        lat, lon = local_mercator_to_wgs84(
            easting, northing,
            reference_coordinate.latitude, reference_coordinate.longitude
        )
        return cls(latitude=lat, longitude=lon, **kwargs)

    def to_local_mercator(self, reference_coordinate):
        r"""Convert wgs84 coordinates into a local mercator projection.

        Conventions here are :math:`\lambda` as the longitude and :math:`\varphi` as the latitude,
        :math:`x` is easting and :math:`y` is northing.

        .. math::
            x &= R(\lambda -\lambda _{0})\\
            y &= R\ln \left[\tan \left(\frac{\pi}{4} + \frac{\varphi - \varphi_0}{2}\right)\right]
        """
        reference_coordinate = Position(reference_coordinate)
        easting, northing = wgs84_to_local_mercator(
            self.latitude, self.longitude,
            reference_coordinate.latitude, reference_coordinate.longitude
        )
        return easting, northing

    def local_length_scale(self):
        """How many nautical miles one longitude minute is

        This gives the apparent length scale for the x-axis in
        mercator projections, i.e., cos(latitude).
        The scaleratio for an x-axis should be set to this value,
        if equal length x- and y-axes are desired, e.g.,
        ```
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
        ```
        """
        # We take the mean so that it works with subclasses with arrays, e.g., Line.
        return np.cos(np.radians(self.latitude.mean().item()))


class Position(_Coordinates):
    @staticmethod
    def parse_coordinates(*args, latitude=None, longitude=None):
        if latitude is not None and longitude is not None:
            return latitude, longitude

        if len(args) == 1:
            arg = args[0]
            try:
                return arg.latitude, arg.longitude
            except AttributeError:
                pass
            try:
                return arg['latitude'], arg['longitude']
            except (KeyError, TypeError):
                pass
            if isinstance(arg, str):
                matches = re.match(
                    r"""((?P<latdeg>[+\-\d.]+)°?)?((?P<latmin>[\d.]+)')?((?P<latsec>[\d.]+)")?(?P<lathemi>[NS])?"""
                    r"""[,]?"""
                    r"""((?P<londeg>[+\-\d.]+)°?)?((?P<lonmin>[\d.]+)')?((?P<lonsec>[\d.]+)")?(?P<lonhemi>[EW])?""",
                    re.sub(r"\s", "", arg)
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
                    raise ValueError(f"Could not parse coordinate string '{arg}', used only {digits_parsed} of {digits_to_parse} digits")

                if matches["lathemi"] == "S":
                    latitude = -abs(latitude)
                if matches["lonhemi"] == "W":
                    longitude = -abs(longitude)

                return latitude, longitude

            else:
                # We should never have just a single argument, try unpacking.
                *args, = arg

        if len(args) == 2:
            latitude, longitude = args
            return latitude, longitude
        elif len(args) == 4:
            (
                latitude_degrees, latitude_minutes,
                longitude_degrees, longitude_minutes
             ) = args
            latitude = latitude_degrees + latitude_minutes / 60
            longitude = longitude_degrees + longitude_minutes / 60
            return latitude, longitude
        elif len(args) == 6:
            (
                latitude_degrees, latitude_minutes, latitude_seconds,
                longitude_degrees, longitude_minutes, longitude_seconds
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
            latitude, longitude = self.parse_coordinates(*args, latitude=longitude, longitude=longitude)
            super().__init__(latitude=latitude, longitude=longitude)

    def __repr__(self):
        return f"{type(self).__name__}({self.latitude.item():.4f}, {self.longitude.item():.4f})"

    def angle_between(self, first, second):
        """Calculates the angle between two positions, as seen from this position."""
        if not isinstance(first, Position):
            first = Position(first)
        if not isinstance(second, Position):
            second = Position(second)
        return angle_between(
            self.latitude, self.longitude,
            first.latitude, first.longitude,
            second.latitude, second.longitude,
        )


class BoundingBox:
    def __init__(self, west, south, east, north):
        self.west = west
        self.south = south
        self.east = east
        self.north = north

    def __repr__(self):
        return f"{type(self).__name__}({self.west}, {self.south}, {self.east}, {self.north})"

    @property
    def north_west(self):
        return Position(self.north, self.west)

    @property
    def north_east(self):
        return Position(self.north, self.east)

    @property
    def south_west(self):
        return Position(self.south, self.west)

    @property
    def south_east(self):
        return Position(self.south, self.east)

    @property
    def center(self):
        return Position(latitude=(self.north + self.south) / 2, longitude=(self.west + self.east) / 2)

    def __contains__(self, position):
        position = Position(position)
        if (self.west <= position.longitude <= self.east) and (self.south <= position.latitude <= self.north):
            return True

    def overlaps(self, other):
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
        center = self.center
        westing, northing = self.north_west.to_local_mercator(center)
        easting, southing = self.south_east.to_local_mercator(center)
        extent = max((northing - southing) / center.local_length_scale(), (easting - westing))
        # This has something to do with the size of a tile in pixels (256),
        # the length of the equator (40_000_000), and then some manual scaling
        # to fix the remainder of issues. Worked nice in plotly 5.18, calling mapbox.
        zoom = np.log2(40_000_000 * pixels / 256 / extent).item() - 1.2
        return zoom


class _CoordinateArray(_Coordinates):
    def __repr__(self):
        return f"{type(self).__name__} with {self.latitude.size} points"

    @property
    def bounding_box(self):
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


class Line(_CoordinateArray):
    @classmethod
    def stack_positions(cls, positions, dim='point', **kwargs):
        """Stacks multiple positions into a line"""
        coordinates = [Position(pos).coordinates for pos in positions]
        coordinates = xr.concat(coordinates, dim=dim)
        return cls(coordinates, **kwargs)

    @classmethod
    def concatenate(cls, lines, dim=None, nan_between_lines=False, **kwargs):
        """Concatenates multiple lines

        If the lines are not connected, it is useful to set `nan_between_lines=True`, which puts
        a nan element between each line. This makes most plotting libraries split the lines in
        visualizations.
        """
        first_line_coords = lines[0].coordinates
        if dim is None:
            if len(first_line_coords.dims) != 1:
                raise ValueError("Cannot guess concatenation dimensions for multi-dimensional line.")
            dim = next(iter(first_line_coords.dims))

        if nan_between_lines:
            nan_data = xr.full_like(first_line_coords.isel({dim: 0}), np.nan).expand_dims(dim)
            lines = sum([[line.coordinates, nan_data] for line in lines], [])
        coordinates = xr.concat(lines, dim=dim)
        return cls(coordinates, **kwargs)

    @classmethod
    def at_position(cls, position, length, bearing, n_points=100, symmetric=False, dim="line"):
        if symmetric:
            n_points += (n_points + 1) % 2
            distance = np.linspace(-length / 2, length / 2, n_points)
        else:
            distance = np.linspace(0, length, n_points)

        distance = xr.DataArray(distance, dims=dim)
        position = Position(position)
        lat, lon = shift_position(position.latitude, position.longitude, distance, bearing)
        return cls(latitude=lat, longitude=lon)


class Track(_CoordinateArray):
    def __init__(self, data, calculate_course=False, calculate_speed=False):
        super().__init__(data)
        if calculate_course:
            self.course
        if calculate_speed:
            self.speed

    @property
    def time(self):
        return self._data["time"]

    @property
    def course(self):
        if "course" in self._data:
            return self._data["course"]
        coords = self.coordinates
        before = coords.shift(time=1).dropna("time")
        after = coords.shift(time=-1).dropna("time")
        interior_course = bearing_to(
            before.latitude, before.longitude,
            after.latitude, after.longitude,
        )
        first_course = bearing_to(
            coords.isel(time=0).latitude, coords.isel(time=1).longitude,
            coords.isel(time=1).latitude, coords.isel(time=1).longitude,
        ).assign_coords(time=coords.time[0])
        last_course = bearing_to(
            coords.isel(time=-2).latitude, coords.isel(time=-2).longitude,
            coords.isel(time=-1).latitude, coords.isel(time=-1).longitude,
        ).assign_coords(time=coords.time[-1])
        course = xr.concat([first_course, interior_course, last_course], dim="time")
        self._data["course"] = course
        return self._data["course"]

    @property
    def speed(self):
        if "speed" in self._data:
            return self._data["speed"]
        coords = self.coordinates
        before = coords.shift(time=1).dropna("time")
        after = coords.shift(time=-1).dropna("time")

        distance_delta = distance_to(
            before.latitude, before.longitude,
            after.latitude, after.longitude
        )
        # We cannot reuse the previous shift here, since the time coordinate is not shifted there
        time_delta = (coords.time.shift(time=-1).dropna("time") - coords.time.shift(time=1).dropna("time")) / np.timedelta64(1, "s")
        interior_speed = distance_delta / time_delta

        first_distance = distance_to(
            coords.isel(time=0).latitude, coords.isel(time=1).longitude,
            coords.isel(time=1).latitude, coords.isel(time=1).longitude,
        )
        first_time = (coords.time[1] - coords.time[0]) / np.timedelta64(1, "s")
        first_speed = (first_distance / first_time).assign_coords(time=coords.time[0])

        last_distance = distance_to(
            coords.isel(time=-2).latitude, coords.isel(time=-2).longitude,
            coords.isel(time=-1).latitude, coords.isel(time=-1).longitude,
        )
        last_time = (coords.time[-1] - coords.time[-2]) / np.timedelta64(1, "s")
        last_speed = (last_distance / last_time).assign_coords(time=coords.time[-1])
        speed = xr.concat([first_speed, interior_speed, last_speed], dim="time")

        self._data["speed"] = speed
        return self._data["speed"]

    def average_course(self, resolution=None):
        return average_angle(self.course, resolution=resolution)

    def closest_point(self, other):
        """Get the point in this track closest to a position."""
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
            track['aspect_angle'] *= -1

        angles = xr.DataArray(angles, coords={'segment': angles})
        center_indices = abs(angles - track.aspect_angle).argmin('time')
        segment_centers = track.isel(time=center_indices)

        # Run a check that we get the windows we want. A sane way might be to check that the
        # first and last windows are closer to their targets than the next window.
        if angles.size > 1:
            actual_first_angle = track.aspect_angle.sel(time=segment_centers.isel(segment=0).time)
            if abs(actual_first_angle - angles.isel(segment=0)) > abs(actual_first_angle - angles.isel(segment=1)):
                raise ValueError(f'Could not find window centered at {angles.isel(segment=0):.1f}⁰, found at most {actual_first_angle:.1f}⁰.')
            actual_last_angle = track.aspect_angle.sel(time=segment_centers.isel(segment=-1).time)
            if abs(actual_last_angle - angles.isel(segment=-1)) > abs(actual_last_angle - angles.isel(segment=-2)):
                raise ValueError(f'Could not find window centered at {angles.isel(segment=-1):.1f}⁰, found at most {actual_last_angle:.1f}⁰.')

        segments = []
        track_angles = track.aspect_angle.data
        track_lat = track.latitude.data
        track_lon = track.longitude.data
        track_time = track.time.data
        for angle, segment_center in segment_centers.groupby('segment', squeeze=False):
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
                        raise ValueError(f'Start of window at {angle}⁰ not found in track. Not sufficiently high angles from window center.')
            if segment_min_duration:
                while abs(segment_center.time - track_time[start_idx]) / np.timedelta64(1, 's') < segment_min_duration / 2:
                    start_idx -= 1
                    if start_idx < 0:
                        raise ValueError(f'Start of window at {angle}⁰ not found in track. Not sufficient time from window center.')
            if segment_min_length:
                while distance_to(segment_center.latitude, segment_center.longitude, track_lat[start_idx], track_lon[start_idx]) < segment_min_length / 2:
                    start_idx -= 1
                    if start_idx < 0:
                        raise ValueError(f'Start of window at {angle}⁰ not found in track. Not sufficient distance from window center.')
            # Finding the end of the window
            stop_idx = center_idx
            if segment_min_angle:
                while abs(segment_center.aspect_angle - track_angles[stop_idx]) < segment_min_angle / 2:
                    stop_idx += 1
                    if stop_idx == track.sizes['time']:
                        raise ValueError(f'End of window at {angle}⁰ not found in track. Not sufficiently high angles from window center.')
            if segment_min_duration:
                while abs(segment_center.time - track_time[stop_idx]) / np.timedelta64(1, 's') < segment_min_duration / 2:
                    stop_idx += 1
                    if stop_idx == track.sizes['time']:
                        raise ValueError(f'End of window at {angle}⁰ not found in track. Not sufficient time from window center.')
            if segment_min_length:
                while distance_to(segment_center.latitude, segment_center.longitude, track_lat[stop_idx], track_lon[stop_idx]) < segment_min_length / 2:
                    stop_idx += 1
                    if stop_idx == track.sizes['time']:
                        raise ValueError(f'End of window at {angle}⁰ not found in track. Not sufficient distance from window center.')

            # Creating the window and saving some attributes
            if start_idx == stop_idx:
                segments.append(segment_center.assign(length=0, angle_span=0, duration=0).reset_coords('time'))
            else:
                segment_start, segment_stop = track.isel(time=start_idx), track.isel(time=stop_idx)
                segments.append(
                    xr.concat([segment_start, segment_center, segment_stop], dim='time')
                    .assign_coords(edge=('time', ['start', 'center', 'stop']))
                    .swap_dims(time='edge')
                    .assign(
                        length=distance_to(segment_start.latitude, segment_start.longitude, segment_stop.latitude, segment_stop.longitude),
                        angle_span=segment_stop.aspect_angle - segment_start.aspect_angle,
                        duration=(segment_stop.time - segment_start.time) / np.timedelta64(1, 's'),
                    )
                    .reset_coords('time')
                )

        if single_segment:
            return segments[0]
        return xr.concat(segments, dim='segment')

    @property
    def time_window(self):
        return _core.TimeWindow(start=self.time[0], stop=self.time[-1])

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        new_window = self.time_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        if isinstance(new_window, _core.TimeWindow):
            start = new_window.start.in_tz('UTC').naive()
            stop = new_window.stop.in_tz('UTC').naive()
            return type(self)(self._data.sel(time=slice(start, stop)))
        else:
            return self._data.sel(time=new_window.in_tz('UTC').naive(), method='nearest')

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
            A new instance of the same type as `self`, with the data resampled at the specified time points.
        """
        if not isinstance(time, xr.DataArray):
            n_samples = int(np.floor(self.time_window.duration * time))
            start_time = _core.time_to_np(self.time_window.start)
            offsets = np.arange(n_samples) * 1e9 / time
            time = start_time + offsets.astype("timedelta64[ns]")
        data = self._data.interp(
            time=time,
            **kwargs
        )
        new = type(self)(data)
        return new


def Sensor(sensor, /, position=None, sensitivity=None, depth=None, latitude=None, longitude=None):
    """Stores sensor information.

    Typical sensor information is the position, sensitivity, and deployment depth.
    The position can be given as a string, a tuple, or separate longitude and latitudes.
    If a position is provided, the instance will have all methods from the `Position` class.

    Parameters
    ----------
    sensor : str
        The label for the sensor. All sensors must have a label.
    position : str or tuple
        A coordinate string, or a tuple with lat, lon information.
    sensitivity : float
        The sensor sensitivity, in dB re. V/Q,
        where Q is the desired physical unit.
    depth : float
        Sensor deployment depth.
    """
    if isinstance(sensor, _Sensor):
         sensor = sensor._data
    if isinstance(sensor, xr.Dataset):
        sensor = sensor[[key for key, value in sensor.notnull().items() if value]]
        if "latitude" in sensor and "longitude" in sensor:
            obj = _SensorPosition(sensor)
        else:
            obj = _Sensor(sensor)
    else:
        if position is not None or (latitude is not None and longitude is not None):
            obj = _SensorPosition(position, latitude=latitude, longitude=longitude)
        else:
            obj = _Sensor(xr.Dataset())
        obj._data.coords["sensor"] = sensor

    if "sensor" not in obj:
        raise ValueError("Cannot have unlabeled sensors")
    if sensitivity is not None:
        obj._data["sensitivity"] = sensitivity
    if depth is not None:
        obj._data["depth"] = depth
    return obj


class _Sensor(_core.DatasetWrap):
    def __repr__(self):
        sens = "" if "sensitivity" not in self._data else f", sensitivity={self['sensitivity']:.2f}"
        depth = "" if "depth" not in self._data else f", depth={self['depth']:.2f}"
        return f"Sensor({self.label}{sens}{depth})"

    @property
    def label(self):
        return self._data["sensor"].item()

    def with_data(self, **kwargs):
        data = self._data.copy()
        if "sensor" not in data.dims:
            data = data.expand_dims("sensor")
        for key, value in kwargs.items():
            if isinstance(value, xr.DataArray):
                if "sensor" not in value.dims:
                    raise ValueError("Cannot add xarray data without sensor dimension to sensors")
                data[key] = value
            elif isinstance(value, dict):
                data[key] = xr.DataArray([value[key] for key in data["sensor"].values], coords={"sensor": data["sensor"]})
            elif np.size(value) == 1:
                data[key] = np.squeeze(value)
            elif np.size(value) != data["sensor"].size:
                raise ValueError(f"Cannot assign {np.size(value)} values to {data["sensor"].size} sensors")
            else:
                data[key] = xr.DataArray(value, coords={"sensor": data["sensor"]})
        return type(self)(data.squeeze())


class _SensorPosition(_Sensor, Position):
    def __repr__(self):
        sens = "" if "sensitivity" not in self._data else f", sensitivity={self['sensitivity']:.2f}"
        depth = "" if "depth" not in self._data else f", depth={self['depth']:.2f}"
        return f"Sensor({self.label}, latitude={self.latitude:.4f}, longitude={self.longitude:.4f}{sens}{depth})"


def SensorArray(*sensors, **kwargs):
    """Collects sensor information from multiple sensors.

    This accepts two types of calls: positional sensors or keywords with dicts.
    The positional format is `SensorArray(sensor_1, sensor_2, ...)`
    where each sensor is a `Sensor`.
    The other format is keyword arguments with sensor labels as the keys, and a dictionary
    with the sensor information as the value, e.g.,

    ```
    SensorArray(
        soundtrap_1={'position': (58.25, 11.14), 'sensitivity': -182},
        soundtrap_2={'position': (58.26, 11.15), 'sensitivity': -183},
    )
    ```
    Note that labels that are not valid arguments can still be created using dict unpacking
    ```
    SensorArray(**{
        'SoundTrap 1': {'position': (58.25, 11.14), 'sensitivity': -182},
        'SoundTrap 2': {'position': (58.26, 11.15), 'sensitivity': -183},
    })
    ```
    see `Sensor` for more details on the possible information.
    """
    if kwargs:
        sensors = sensors + tuple(
            Sensor(label, **values)
            for label, values in kwargs.items()
        )
    sensors = [sensor._data if isinstance(sensor, _Sensor) else sensor for sensor in sensors]
    sensors = xr.concat(sensors, dim='sensor')
    for key, value in sensors.items():
        if np.ptp(value.values) == 0:
            sensors[key] = value.mean()
    if ("latitude" in sensors and "longitude" in sensors):
        if (sensors["latitude"].size == 1 and sensors["longitude"].size == 1):
            obj = _ColocatedSensorArray(sensors)
        else:
            obj = _LocatedSensorArray(sensors)
    else:
        obj = _SensorArray(sensors)
    return obj


class _SensorArray(_Sensor):

    @property
    def sensors(self):
        return {label: Sensor(data.squeeze()) for label, data in self._data.groupby("sensor", squeeze=False)}

    def __repr__(self):
        return f"SensorArray with {self._data['sensor'].size} sensors"

    @property
    def label(self):
        return tuple(self._data["sensor"].data)

    def __add__(self, other):
        if isinstance(other, _SensorArray):
            return SensorArray(*self.sensors.values(), *other.sensors.values())
        if not isinstance(other, _Sensor):
            other = Sensor(other)
        return SensorArray(*self.sensors.values(), other)


class _LocatedSensorArray(_SensorArray, _Coordinates):
    def __init__(self, data):
        _SensorArray.__init__(self, data)


class _ColocatedSensorArray(_SensorArray, Position):
    def __init__(self, data):
        _SensorArray.__init__(self, data)

    def __repr__(self):
        return f"SensorArray with {self._data['sensor'].size} sensors at ({self.latitude:.4f}, {self.longitude:.4f})"
