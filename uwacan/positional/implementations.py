import numpy as np

WGS84_equatorial_radius = 6_378_137.0
WGS84_polar_radius = 6_356_752.3
mercator_scale_factor = 0.9996

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
    radius = WGS84_equatorial_radius * mercator_scale_factor

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
    radius = WGS84_equatorial_radius * mercator_scale_factor
    local_longitude = np.radians(longitude - reference_longitude)
    local_latitude = np.radians(latitude - reference_latitude)
    easting = radius * local_longitude
    northing = radius * np.log(np.tan(np.pi / 4 + local_latitude / 2))
    return easting, northing

def wrap_angle(degrees):
    '''Wrap an angle to (-180, 180].'''
    return 180 - np.mod(180 - degrees, 360)


_re_rp_2 = (WGS84_equatorial_radius / WGS84_polar_radius)**2
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
        ((WGS84_equatorial_radius**2 * np.cos(lat))**2 + (WGS84_polar_radius**2 * np.sin(lat))**2)
        /
        ((WGS84_equatorial_radius * np.cos(lat))**2 + (WGS84_polar_radius * np.sin(lat))**2)
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
