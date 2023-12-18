"""Handles positional tracks.

This manages logs over positions of measurement objects via e.g. GPS.
Some of the operations include smoothing data, calculating distances,
and reading log files.
"""

import numpy as np
from geographiclib.geodesic import Geodesic
import pendulum
import xarray as xr
import os
import re
geod = Geodesic.WGS84


one_knot = 1.94384
"""One m/s in knots, i.e., this has the units of knots/(m/s).
Multiply with this value to go from m/s to knots,
divide by this value to go from knots to m/s."""


def time_to_np(input):
    if isinstance(input, np.datetime64):
        return input
    if not isinstance(input, pendulum.DateTime):
        input = time_to_datetime(input)
    return np.datetime64(input.in_tz('UTC').naive())


def time_to_datetime(input, fmt=None, tz=None):
    """Sanitize datetimes to the same internal format.

    This is not really an outwards-facing function. The main use-case is
    to make sure that we have `pendulum.DateTime` objects to work with
    internally.
    It's recommended that users use nice datetimes instead of strings,
    but sometimes a user will pass a string somewhere and then we'll try to
    parse it.
    """
    try:
        return pendulum.instance(input)
    except ValueError as err:
        if 'instance() only accepts datetime objects.' in str(err):
            pass
        else:
            raise

    if isinstance(input, xr.DataArray):
        if input.size == 1:
            input = input.values
        else:
            raise ValueError('Cannot convert multiple values at once.')

    if fmt is not None:
        return pendulum.from_format(input, fmt=fmt, tz=tz)

    if isinstance(input, np.datetime64):
        input = input.astype('timedelta64') / np.timedelta64(1, 's')  # Gets the time as a timestamp, will parse nicely below.

    try:
        return pendulum.from_timestamp(input)
    except TypeError as err:
        if 'object cannot be interpreted as an integer' in str(err):
            pass
        else:
            raise
    return pendulum.parse(input)


class TimeWindow:
    def __init__(self, start=None, stop=None, center=None, duration=None):
        if start is not None:
            start = time_to_datetime(start)
        if stop is not None:
            stop = time_to_datetime(stop)
        if center is not None:
            center = time_to_datetime(center)

        if None not in (start, stop):
            _start = start
            _stop = stop
            start = stop = None
        elif None not in (center, duration):
            _start = center - pendulum.duration(seconds=duration / 2)
            _stop = center + pendulum.duration(seconds=duration / 2)
            center = duration = None
        elif None not in (start, duration):
            _start = start
            _stop = start + pendulum.duration(seconds=duration)
            start = duration = None
        elif None not in (stop, duration):
            _stop = stop
            _start = stop - pendulum.duration(seconds=duration)
            stop = duration = None
        elif None not in (start, center):
            _start = start
            _stop = start + (center - start) / 2
            start = center = None
        elif None not in (stop, center):
            _stop = stop
            _start = stop - (stop - center) / 2
            stop = center = None
        else:
            raise TypeError('Needs two of the input arguments to determine time window.')

        if (start, stop, center, duration) != (None, None, None, None):
            raise TypeError('Cannot input more than two input arguments to a time window!')

        self._window = pendulum.period(_start, _stop)

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        if time is None:
            # Period specified with keyword arguments, convert to period.
            if (start, stop, center, duration).count(None) == 3:
                # Only one argument which has to be start or stop, fill the other from self.
                if start is not None:
                    window = type(self)(start=start, stop=self.stop)
                elif stop is not None:
                    window = type(self)(start=self.start, stop=stop)
                else:
                    raise TypeError('Cannot create subwindow from arguments')
            elif duration is not None and True in (start, stop, center):
                if start is True:
                    window = type(self)(start=self.start, duration=duration)
                elif stop is True:
                    window = type(self)(stop=self.stop, duration=duration)
                elif center is True:
                    window = type(self)(center=self.center, duration=duration)
                else:
                    raise TypeError('Cannot create subwindow from arguments')
            else:
                # The same types explicit arguments as the normal constructor
                window = type(self)(start=start, stop=stop, center=center, duration=duration)
        elif isinstance(time, type(self)):
            window = time
        elif isinstance(time, pendulum.Period):
            window = type(self)(start=time.start, stop=time.end)
        elif isinstance(time, xr.Dataset):
            window = type(self)(start=time.time.min(), stop=time.time.max())
        else:
            # It's not a period, so it shold be a single datetime. Parse or convert, check valitidy.
            time = time_to_datetime(time)
            if time not in self:
                raise ValueError("Received time outside of contained window")
            return time

        if window not in self:
            raise ValueError("Requested subwindow is outside contained time window")
        return window

    def __repr__(self):
        return f'TimeWindow(start={self.start}, stop={self.stop})'

    @property
    def start(self):
        return self._window.start

    @property
    def stop(self):
        return self._window.end

    @property
    def center(self):
        return self.start.add(seconds=self._window.total_seconds() / 2)

    @property
    def duration(self):
        return self._window.total_seconds()

    def __contains__(self, other):
        if isinstance(other, type(self)):
            other = other._window
        if isinstance(other, pendulum.Period):
            return other.start in self._window and other.end in self._window
        return other in self._window


def position(*args, **kwargs):
    """Stacks latitude and longitude in a xr.Dataset

    This function supports multiple variants of calling signature:

    `position(dataset)`
        This returns a dataset with at least `latitude` and `longitude`
        The input dataset can be any object with latitude and longitude properties.
    `position(latitude=lat, longitude=lon)`
    `position(lat, lon)`
        These uses latitude and longitude in degrees with decimals.
        E.g., (57.6931022, 11.974318) -> 57.6931022°N 11.974318°E
        This format can be used as either positional or keyword arguments.
    `position(lat, lat_min, lon, lon_min)`
        This uses degrees and decimal minutes format for the coordinates.
        E.g., (57, 41.586132, 11, 58.45908) -> 57° 41.58613200'N 11° 58.45908000'E
    `position(lat, lat_min, lat_sec, lon, lon_min, lon_sec)`
        This uses degrees and minutes, decimal seconds format for the coordinates.
        E.g., (57, 41, 35.17, 11, 58, 27.54) -> 57° 41' 35.17"N 11° 58' 27.54"E
    `position(..., t)`
    `position(..., time=t)`
        A last optional positional or keyword argument can be used to supply a time
        for the position as well.
    """
    if len(args) == 1:
        arg = args[0]
        args = tuple()
        if isinstance(arg, xr.Dataset):
            if 'latitude' not in arg or 'longitude' not in arg:
                raise ValueError('latitude and longitude apparently not in position dataset')
            return arg
        else:
            if hasattr(arg, 'latitude') and hasattr(arg, 'latitude'):
                if hasattr(arg, 'time'):
                    args = (arg.latitude, arg.longitude, arg.time)
                else:
                    args = (arg.latitude, arg.longitude)
            else:
                # We should never have just a single argument, try unpacking.
                *args, = arg

    latitude = kwargs.pop('latitude', None)
    longitude = kwargs.pop('longitude', None)
    time = kwargs.pop('time', None)

    if len(args) % 2:
        if time is not None:
            raise TypeError("position got multiple values for argument 'time'")
        *args, time = args

    if len(args) != 0:
        if latitude is not None:
            raise TypeError("position got multiple values for argument 'latitude'")
        if longitude is not None:
            raise TypeError("position got multiple values for argument 'longitude'")

        if len(args) == 2:
            latitude, longitude = args
        elif len(args) == 4:
            (
                latitude_degrees, latitude_minutes,
                longitude_degrees, longitude_minutes
             ) = args
            latitude = latitude_degrees + latitude_minutes / 60
            longitude = longitude_degrees + longitude_minutes / 60
        elif len(args) == 6:
            (
                latitude_degrees, latitude_minutes, latitude_seconds,
                longitude_degrees, longitude_minutes, longitude_seconds
             ) = args
            latitude = latitude_degrees + latitude_minutes / 60 + latitude_seconds / 3600
            longitude = longitude_degrees + longitude_minutes / 60 + longitude_seconds / 3600
        else:
            raise TypeError(f"Undefined number of non-time arguments for position {len(args)} was given, expects 2, 4, or 6.")

    dataset = xr.Dataset({'latitude': latitude, 'longitude': longitude})
    if time is not None:
        # TODO: convert the time into a numpy datetime!
        dataset['time'] = time
    return dataset


def format_coordinates(*args, format='minutes', precision=None, **kwargs):
    pos = position(*args, **kwargs)
    latitude = np.atleast_1d(pos.latitude.values)
    longitude = np.atleast_1d(pos.longitude.values)
    def ns(lat):
        return 'N' if lat > 0 else 'S'
    def ew(lon):
        return 'E' if lon > 0 else 'W'

    if format.lower()[:3] == 'deg':
        if precision is None:
            precision = 6

        def format(lat, lon):
            lat = f"{abs(lat):.{precision}f}°{ns(lat)}"
            lon = f"{abs(lon):.{precision}f}°{ew(lon)}"
            return lat + " " + lon

    elif format.lower()[:3] == 'min':
        if precision is None:
            precision = 4

        def format(lat, lon):
            latdeg, latmin = np.divmod(abs(lat) * 60, 60)
            londeg, lonmin = np.divmod(abs(lon) * 60, 60)
            lat = f"{abs(latdeg):.0f}°{latmin:.{precision}f}'{ns(lat)}"
            lon = f"{abs(londeg):.0f}°{lonmin:.{precision}f}'{ew(lon)}"
            return lat + " " + lon
    elif format.lower()[:3] == 'sec':
        format = 'sec'
        if precision is None:
            precision = 2

        def format(lat, lon):
            latdeg, latmin = np.divmod(abs(lat) * 60, 60)
            londeg, lonmin = np.divmod(abs(lon) * 60, 60)
            latmin, latsec = np.divmod(latmin * 60, 60)
            lonmin, lonsec = np.divmod(lonmin * 60, 60)
            lat = f"""{abs(latdeg):.0f}°{latmin:.0f}'{latsec:.{precision}f}"{ns(lat)}"""
            lon = f"""{abs(londeg):.0f}°{latmin:.0f}'{lonsec:.{precision}f}"{ew(lon)}"""
            return lat + " " + lon

    formatted = [
        format(lat, lon)
        for lat, lon in zip(latitude, longitude)
    ]
    if len(formatted) == 1:
        return formatted[0]
    return formatted


def distance_between(first, second):
    """Calculate the distance between two coordinates."""
    def func(lat_1, lon_1, lat_2, lon_2):
        return geod.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=geod.DISTANCE)['s12']
    return xr.apply_ufunc(
        func,
        first.latitude, first.longitude,
        second.latitude, second.longitude,
        vectorize=True,
        join='inner',
    )


def wrap_angle(angle):
    '''Wrap an angle to (-180, 180].'''
    return 180 - np.mod(180 - angle, 360)


def bearing_to(first, second):
    """Calculate the heading from one coordinate to another."""
    def func(lat_1, lon_1, lat_2, lon_2):
        return geod.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=geod.AZIMUTH)['azi1']
    return xr.apply_ufunc(
        func,
        first.latitude, first.longitude,
        second.latitude, second.longitude,
        vectorize=True,
        join='inner',
    )


def average_heading(heading, resolution=None):
    complex_heading = np.exp(1j * np.radians(heading))
    heading = wrap_angle(np.degrees(np.angle(complex_heading.mean())))
    if resolution is None:
        return heading

    if not isinstance(resolution, str):
        return wrap_angle(np.round(heading / 360 * resolution) * 360 / resolution)

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
    name = min([(abs(deg - heading), name) for deg, name in names], key=lambda x: x[0])[1]
    return name.capitalize()


def angle_between(center, first, second):
    """Calculate the angle between two coordinates, as seen from a center vertex."""
    first_heading = bearing_to(center, first)
    second_heading = bearing_to(center, second)
    return wrap_angle(second_heading - first_heading)


def shift_position(pos, distance, bearing):
    def func(lat, lon, head, dist):
        out = geod.Direct(lat, lon, head, dist, outmask=geod.LATITUDE | geod.LONGITUDE)
        return out['lat2'], out['lon2']
    lat, lon = xr.apply_ufunc(
        func,
        pos.latitude,
        pos.longitude,
        bearing,
        distance,
        vectorize=True,
        output_core_dims=[[], []]
    )
    return position(lat, lon)


def calculate_course(positions, inplace=False):
    course = bearing_to(positions, positions.shift(time=-1).dropna('time'))
    if inplace:
        positions['course'] = course
        return positions
    return course


def circle_at(center, radius, n_points=72):
    angles = np.linspace(0, 360, n_points + 1)
    # positions = []
    latitudes = np.zeros(angles.size)
    longitudes = np.zeros(angles.size)
    for idx, angle in angles:
        out = geod.Direct(center.latitude, center.longitude, angle, radius)
        latitudes[idx] = out['lat2']
        longitudes[idx] = out['lon2']
    return position(latitude=latitudes, longitude=longitudes)


def closest_point(reference, track):
    distances = distance_between(reference, track)
    return track.assign(distance=distances).isel(distances.argmin(...))


def aspect_segments(
    reference,
    track,
    angles,
    segment_min_length=None,
    segment_min_angle=None,
    segment_min_duration=None,
):
    """Get time segments corresponding to specific aspect angles.

    Parameters
    ----------
    track : xarray.Dataset w. latitude and longitude
        Track from which to analyze the segments.
    angles : array_like
        The aspect angles to find. This is a value in degrees relative to the closest point to
        the track from the reference point.
    segment_min_length : numeric, optional
        The minimum length of each segment, in meters.
    segment_min_angle : numeric, optional
        The minimum length of each segment, seen as an angle from the reference point.
    segment_min_duration : numeric, optional
        The minimum duration of each window, in seconds.
    """
    track = track[['latitude', 'longitude']]  # Speeds up some computations since we're not managing unnecessary data
    cpa = closest_point(reference, track)  # If the path if to long this will crunch a shit-ton of data...

    try:
        iter(angles)
    except TypeError:
        single_segment = True
        angles = [angles]
    else:
        single_segment = False

    angles = np.sort(angles)
    track = track.assign(aspect_angle=angle_between(reference, cpa, track))
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
            raise ValueError(f'Could not find window centered at {angles.isel(segment=0)}⁰, found at most {actual_first_angle}⁰.')
        actual_last_angle = track.aspect_angle.sel(time=segment_centers.isel(segment=-1).time)
        if abs(actual_last_angle - angles.isel(segment=-1)) > abs(actual_first_angle - angles.isel(segment=-2)):
            raise ValueError(f'Could not find window centered at {angles.isel(segment=-1)}⁰, found at most {actual_last_angle}⁰.')

    segments = []
    for angle, segment_center in segment_centers.groupby('segment'):
        # Finding the start of the window
        # The inner loops here are somewhat slow, likely due to indexing into the xr.Dataset all the time
        # At the time of writing (2023-12-14), there seems to be no way to iterate over a dataset in reverse order.
        # The `groupby` method can be used to iterate forwards, which solves finding the end of the segment,
        # but calling `track.sel(time=slice(t, None, -1)).groupby('time')` still iterates in the forward order.
        center_idx = int(np.abs(track.time - segment_center.time).argmin())
        start_idx = center_idx
        if segment_min_angle:
            while abs(segment_center.aspect_angle - track.isel(time=start_idx).aspect_angle) < segment_min_angle / 2:
                start_idx -= 1
                if start_idx < 0:
                    raise ValueError(f'Start of window at {angle}⁰ not found in track. Not sufficiently high angles from window center.')
        if segment_min_duration:
            while abs(segment_center.time - track.time.isel(time=start_idx)) / np.timedelta64(1, 's') < segment_min_duration / 2:
                start_idx -= 1
                if start_idx < 0:
                    raise ValueError(f'Start of window at {angle}⁰ not found in track. Not sufficient time from window center.')
        if segment_min_length:
            while distance_between(segment_center, track.isel(time=start_idx)) < segment_min_length / 2:
                start_idx -= 1
                if start_idx < 0:
                    raise ValueError(f'Start of window at {angle}⁰ not found in track. Not sufficient distance from window center.')
        # Finding the end of the window
        stop_idx = center_idx
        if segment_min_angle:
            while abs(segment_center.aspect_angle - track.isel(time=stop_idx).aspect_angle) < segment_min_angle / 2:
                stop_idx += 1
                if stop_idx == track.sizes['time']:
                    raise ValueError(f'End of window at {angle}⁰ not found in track. Not sufficiently high angles from window center.')
        if segment_min_duration:
            while abs(segment_center.time - track.time.isel(time=stop_idx)) / np.timedelta64(1, 's') < segment_min_duration / 2:
                stop_idx += 1
                if stop_idx == track.sizes['time']:
                    raise ValueError(f'End of window at {angle}⁰ not found in track. Not sufficient time from window center.')
        if segment_min_length:
            while distance_between(segment_center, track.isel(time=stop_idx)) < segment_min_length / 2:
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
                    length=distance_between(segment_start, segment_stop),
                    angle_span=segment_stop.aspect_angle - segment_start.aspect_angle,
                    duration=(segment_stop.time - segment_start.time) / np.timedelta64(1, 's'),
                )
                .reset_coords('time')
            )

    if single_segment:
        return segments[0]
    return xr.concat(segments, dim='segment')


def blueflow(path, renames=None):
    import pandas
    ext = os.path.splitext(path)[1]
    if ext == '.xlsx':
        data = pandas.read_excel(path)
    elif ext == '.csv':
        data = pandas.read_csv(path)
    else:
        raise ValueError(f"Unknown fileformat for blueflow file '{path}'. Only xlsx and csv supported.")
    data = data.to_xarray()
    names = {}
    exp = r'([^\(\)\[\]]*) [\[\(]([^\(\)\[\]]*)[\]\)]'
    for key in list(data):
        name, unit = re.match(exp, key).groups()
        names[key] = name.strip()
        data[key].attrs['unit'] = unit
    data = data.rename(names)

    renames = {
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Timestamp': 'time',
        'Time': 'time',
        'Tidpunkt': 'time',
        'Latitud': 'latitude',
        'Longitud': 'longitude',
    } | (renames or {})

    renames = {key: value for key, value in renames.items() if key in data}
    data = data.rename(renames).set_coords('time').swap_dims(index='time').drop('index')
    if not np.issubdtype(data.time.dtype, np.datetime64):
        data['time'] = xr.apply_ufunc(np.datetime64, data.time, vectorize=True, keep_attrs=True)
    return data


def correct_gps_offset(positions, heading, forwards=0, portwards=0, to_bow=0, to_stern=0, to_port=0, to_starboard=0, inplace=False):
    """Correct positions with respect to ship heading.

    The positions will be shifted in the `heading` direction by `forwards + (to_bow - to_stern) / 2`,
    and towards "port" `heading - 90` by `portwards + (to_port - to_starboard) / 2`.
    Typical usage is to give the receiver position using the `to_x` arguments, and the desired
    acoustic reference location with the `forwards` and `portwards` arguments.
    Inserting correct values for all the `to_x` arguments will center the position on the ship middle, so that
    the `forwards` and `portwards` arguments are relative to the ship center. Alternatively, leave the `to_x` arguments
    as the default 0 and only give the desired `forwards` and `portwards` arguments.

    Parameters
    ----------
    positions : xarray.Dataset w. latitude and longitude
        The positions to modify
    heading : array like
        The headings of the ship. Must be compatible with the positions Dataset
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
    inplace : boolean, default False
        If this is true, the corrected positions will be assigned the the input position dataset,
        which is returned. If false, a new dataset with only latitude and longitude is returned.
    """
    forwards = forwards + (to_bow - to_stern) / 2
    portwards = portwards + (to_port - to_starboard) / 2
    front_back_fixed = shift_position(positions, forwards, heading)
    sideways_fixed = shift_position(front_back_fixed, portwards, heading - 90)
    if inplace:
        if isinstance(positions, xr.Dataset):
            positions['latitude'] = sideways_fixed.latitude
            positions['longitude'] = sideways_fixed.longitude
        else:
            positions.latitude = sideways_fixed.latitude
            positions.longitude = sideways_fixed.longitude
        return positions
    else:
        return sideways_fixed
