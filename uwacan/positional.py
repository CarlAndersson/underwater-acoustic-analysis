"""Handles positional tracks.

This manages logs over positions of measurement objects via e.g. GPS.
Some of the operations include smoothing data, calculating distances,
and reading log files.
"""

import abc
import numpy as np
import scipy.interpolate
import scipy.signal
from geographiclib.geodesic import Geodesic
from . import _core
import datetime
import dateutil
import bisect
geod = Geodesic.WGS84


one_knot = 1.94384


def wrap_angle(angle):
    '''Wraps an angle to (-180, 180].'''
    return 180 - np.mod(180 - angle, 360)


class Position:
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            arg = args[0]
            args = tuple()
            if isinstance(arg, Position):
                if arg.timestamp is None:
                    args = (arg.latitude, arg.longitude)
                else:
                    args = (arg.latitude, arg.longitude, arg.timestamp)
            else:
                try:
                    kwargs = dict(**arg, **kwargs)
                except TypeError as err:
                    if 'argument after ** must be a mapping' not in str(err):
                        raise
                    else:
                        *args, = arg
                        args = tuple(args)

        latitude = kwargs.pop('latitude', None)
        longitude = kwargs.pop('longitude', None)
        timestamp = kwargs.pop('timestamp', None)

        if len(args) % 2:
            if timestamp is not None:
                raise TypeError("Position got multiple values for argument 'timestamp'")
            *args, timestamp = args

        if len(args) != 0:
            if latitude is not None:
                raise TypeError("Position got multiple values for argument 'latitude'")
            if longitude is not None:
                raise TypeError("Position got multiple values for argument 'longitude'")

            if len(args) == 2:
                latitude, longitude = args
            elif len(args) == 4:
                latitude_degrees, latitude_minutes, longitude_degrees, longitude_minutes = args
                latitude = latitude_degrees + latitude_minutes / 60
                longitude = longitude_degrees + longitude_minutes / 60
            elif len(args) == 6:
                latitude_degrees, latitude_minutes, latitude_seconds, longitude_degrees, longitude_minutes, longitude_seconds = args
                latitude = latitude_degrees + latitude_minutes / 60 + latitude_seconds / 3600
                longitude = longitude_degrees + longitude_minutes / 60 + longitude_seconds / 3600
            else:
                raise TypeError(f"Undefined number of non-time arguments for Position {len(args)} was given, expects 2, 4, or 6.")

        self._latitude = latitude
        self._longitude = longitude
        self._timestamp = timestamp

    def __repr__(self):
        lat = f'latitude={self.latitude}'
        lon = f', longitude={self.longitude}'
        time = f', timestamp={self.timestamp}' if self.timestamp is not None else ''
        cls = self.__class__.__name__
        return cls + '(' + lat + lon + time + ')'

    def __eq__(self, other):
        return (
            type(other) == type(self)
            and self.latitude == other.latitude
            and self.longitude == other.longitude
        )

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def minutes(self):
        latdeg, latmin = divmod(self.latitude * 60, 60)
        londeg, lonmin = divmod(self.longitude * 60, 60)
        return f"{latdeg:.0f}° {latmin}', {londeg:.0f}° {lonmin}'"

    @property
    def seconds(self):
        latdeg, latmin = divmod(self.latitude * 60, 60)
        londeg, lonmin = divmod(self.longitude * 60, 60)
        latmin, latsec = divmod(latmin * 60, 60)
        lonmin, lonsec = divmod(latmin * 60, 60)
        return f'''{latdeg:.0f}° {latmin:.0f}' {latsec:.2f}", {londeg:.0f}° {lonmin:.0f}' {lonsec:.2f}"'''

    @property
    def latmin(self):
        return self._latitude * 60 % 60

    @property
    def latsec(self):
        return self._latitude * 3600 % 60

    @property
    def lonmin(self):
        return self._lonitude * 60 % 60

    @property
    def lonsec(self):
        return self._lonitude * 3600 % 60

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def coordinates(self):
        return self.latitude, self.longitude

    def copy(self, deep=False):
        obj = type(self).__new__(type(self))
        obj._latitude = self._latitude
        obj._longitude = self._longitude
        obj._timestamp = self._timestamp
        return obj

    def distance_to(self, other):
        try:
            iter(other)
        except TypeError as err:
            if str(err).endswith('object is not iterable'):
                other = [other]
            else:
                raise

        distances = [
            geod.Inverse(
                self.latitude,
                self.longitude,
                pos.latitude,
                pos.longitude,
                outmask=geod.DISTANCE,
            )['s12']
            for pos in other
        ]
        if len(distances) == 1:
            return distances[0]
        return np.asarray(distances)

    def heading_to(self, other):
        try:
            iter(other)
        except TypeError as err:
            if str(err).endswith('object is not iterable'):
                other = [other]
            else:
                raise

        headings = [
            geod.Inverse(
                self.latitude,
                self.longitude,
                pos.latitude,
                pos.longitude,
                outmask=geod.AZIMUTH,
            )['azi1']
            for pos in other
        ]
        if len(headings) == 1:
            return headings[0]
        return np.asarray(headings)

    def angle_between(self, first_position, second_position):
        first_azimuth = geod.Inverse(
            self.latitude, self.longitude,
            first_position.latitude, first_position.longitude,
            outmask=geod.AZIMUTH
        )['azi1']
        second_azimuth = geod.Inverse(
            self.latitude, self.longitude,
            second_position.latitude, second_position.longitude,
            outmask=geod.AZIMUTH
        )['azi1']
        angular_difference = second_azimuth - first_azimuth
        return wrap_angle(angular_difference)

    def circle(self, radius, n_points=72):
        angles = np.linspace(0, 360, n_points + 1)
        positions = []
        for angle in angles:
            out = geod.Direct(self.latitude, self.longitude, angle, radius)
            positions.append(type(self)(latitude=out['lat2'], longitude=out['lon2']))
        return positions

    def offset_position(self, distance, heading):
        out = geod.Direct(self.latitude, self.longitude, heading, distance)
        return type(self)(latitude=out['lat2'], longitude=out['lon2'])


class Track(abc.ABC):
    """A track of positions measured over time.

    Parameters
    ----------
    latitude : array_like
        The measured latitudes
    longitude : array_like
        The measured longitudes
    time : `TimedSequence`
        Specification of the times where the position was measured.
    """

    def copy(self, deep=False):
        obj = type(self).__new__(type(self))
        return obj

    @abc.abstractmethod
    def __len__(self):
        ...

    @abc.abstractmethod
    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        ...

    @property
    @abc.abstractmethod
    def timestamps(self):
        """List of timestamps as `datetime` objects for each position."""
        ...

    @property
    @abc.abstractmethod
    def relative_time(self):
        """Array of time in the track relative to the start of the track, in seconds."""
        ...

    @property
    @abc.abstractmethod
    def time_period(self):
        """Time window that the track covers."""
        ...

    @property
    @abc.abstractmethod
    def latitude(self):
        """Latitudes of the track, in degrees."""
        ...

    @property
    @abc.abstractmethod
    def longitude(self):
        """Longitudes of the track, in degrees."""
        ...

    @property
    def coordinates(self):
        return np.stack([self.latitude, self.longitude], axis=0)

    @property
    @abc.abstractmethod
    def speed(self):
        """Speed in the track, in meters per second"""
        ...

    @property
    @abc.abstractmethod
    def heading(self):
        """Heading in the track, in degrees"""
        ...

    @property
    def mean(self):
        """Mean position of the track."""
        lat = np.mean(self.latitude)
        lon = np.mean(self.longitude)
        return Position(latitude=lat, longitude=lon)

    def mean_heading(self, resolution=None):
        complex_heading = np.exp(1j * np.radians(self.heading))
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
        name = min([(abs(deg - heading), name) for  deg, name in names], key=lambda x: x[0])[1]
        return name.capitalize()

    @property
    def boundaries(self):
        """Boundaries of the track.

        (lat_min, lat_max, lon_min, lon_max)
        """
        min_lat = np.min(self.latitude)
        min_lon = np.min(self.longitude)
        max_lat = np.max(self.latitude)
        max_lon = np.max(self.longitude)
        return min_lat, max_lat, min_lon, max_lon

    def distance_to(self, other):
        if isinstance(other, Track):
            raise TypeError('Cannot calculate distances between two tracks')
        if isinstance(other, Position):
            return other.distance_to(self)
        try:
            lat, lon = other
        except TypeError as err:
            if str(err).startswith('cannot unpack non-iterable'):
                raise TypeError(f'Cannot calculate distance between track and {other}')
            else:
                raise
        return Position(latitude=lat, longitude=lon).distance_to(self)

    def closest_point(self, other):
        if not isinstance(other, Position):
            try:
                lat, lon = other
            except TypeError as err:
                if str(err).startswith('cannot unpack non-iterable'):
                    raise TypeError(f'Cannot calculate distance between track and {other}')
                else:
                    raise
            other = Position(latitude=lat, longitude=lon)
        distances = other.distance_to(self)
        idx = np.argmin(distances)
        distance = distances[idx]
        position = self[idx]
        position.distance = distance
        return position

    def aspect_windows(self, reference_point, angles, window_min_length=None, window_min_angle=None, window_min_duration=None):
        """Get time windows corresponding to specific aspect angles

        Parameters
        ----------
        reference_point : Position
            The position from where the angles are calculated.
        angles : array_like
            The aspect angles to find. This is a value in degrees relative to the closest point to
            the track from the reference point.
        window_min_length : numeric, optional
            The minimum length of each window, in meters.
        window_min_angle : numeric, optional
            The minimum length of each window, seen as an angle from the reference point.
            If neither of `window_min_length` or `window_min_angle` is given, the `window_min_angle`
            defaults to `resolution`.
        window_min_duration : numeric, optional
            The minimum duration of each window, in seconds.
        """
        cpa = self.closest_point(reference_point)  # If the path if to long this will crunch a shit-ton of data...
        cpa.angle = 0

        try:
            iter(angles)
        except TypeError:
            angles = [angles]

        angles = np.sort(angles)
        pre_cpa_angles = angles[angles < 0]
        post_cpa_angles = angles[angles > 0]

        pre_cpa_window_centers = [cpa]
        post_cpa_window_centers = [cpa]

        for angle in reversed(pre_cpa_angles):
            for point in reversed(self.time_subperiod(stop=pre_cpa_window_centers[-1].timestamp)):
                if abs(reference_point.angle_between(cpa, point)) >= -angle:
                    point.angle = angle
                    pre_cpa_window_centers.append(point)
                    break
            else:
                raise ValueError(f'Could not find window centered at {angle} degrees, found at most {-abs(reference_point.angle_between(cpa, point))}. Include additional early track data.')

        for angle in post_cpa_angles:
            for point in self.time_subperiod(start=post_cpa_window_centers[-1].timestamp):
                if abs(reference_point.angle_between(cpa, point)) >= angle:
                    point.angle = angle
                    post_cpa_window_centers.append(point)
                    break
            else:
                raise ValueError(f'Could not find window centered at {angle} degrees, found at most {abs(reference_point.angle_between(cpa, point))}. Include additional late track data.')

        # Merge the list of window centers
        # The pre cpa list is reversed to have them in the correct order
        # Both lists have the cpa in them, so it's removed.
        if 0 in angles:
            window_centers = pre_cpa_window_centers[:0:-1] + [cpa] + post_cpa_window_centers[1:]
        else:
            window_centers = pre_cpa_window_centers[:0:-1] + post_cpa_window_centers[1:]

        windows = []
        for center in window_centers:
            meets_angle_criteria = window_min_angle is None
            meets_length_criteria = window_min_length is None
            meets_time_criteria = window_min_duration is None
            for point in reversed(self.time_subperiod(stop=center.timestamp)):
                if not meets_angle_criteria:
                    # Calculate angle and check if it's fine
                    meets_angle_criteria = abs(reference_point.angle_between(center, point)) >= window_min_angle / 2
                if not meets_length_criteria:
                    # Calculate length and check if it's fine
                    meets_length_criteria = center.distance_to(point) >= window_min_length / 2
                if not meets_time_criteria:
                    meets_time_criteria = (center.timestamp - point.timestamp).total_seconds() >= window_min_duration / 2
                if meets_length_criteria and meets_angle_criteria and meets_time_criteria:
                    window_start = point
                    break
            else:
                msg = f'Could not find starting point for window at {center.angle} degrees. Include more early track data.'
                if not meets_angle_criteria:
                    msg += f' Highest angle found in track is {abs(reference_point.angle_between(center, point))} degrees, {window_min_angle/2} was requested.'
                if not meets_length_criteria:
                    msg += f' Furthest distance found in track is {center.distance_to(point)}, {window_min_length/2} was requested.'
                if not meets_time_criteria:
                    msg += f' Earliest point found in track is {(center.timestamp - point.timestamp).total_seconds():.2f} before window center, {window_min_duration/2} was requested.'
                raise ValueError(msg)

            meets_angle_criteria = window_min_angle is None
            meets_length_criteria = window_min_length is None
            meets_time_criteria = window_min_duration is None
            for point in self.time_subperiod(start=center.timestamp):
                if not meets_angle_criteria:
                    # Calculate angle and check if it's fine
                    meets_angle_criteria = abs(reference_point.angle_between(center, point)) >= window_min_angle / 2
                if not meets_length_criteria:
                    # Calculate length and check if it's fine
                    meets_length_criteria = center.distance_to(point) >= window_min_length / 2
                if not meets_time_criteria:
                    meets_time_criteria = (point.timestamp - center.timestamp).total_seconds() >= window_min_duration / 2
                if meets_length_criteria and meets_angle_criteria and meets_time_criteria:
                    window_stop = point
                    break
            else:
                msg = f'Could not find stopping point for window at {center.angle} degrees. Include more late track data.'
                if not meets_angle_criteria:
                    msg += f' Highest angle found in track is {abs(reference_point.angle_between(center, point))} degrees, {window_min_angle/2} was requested.'
                if not meets_length_criteria:
                    msg += f' Furthest distance found in track is {center.distance_to(point)}, {window_min_length/2} was requested.'
                if not meets_time_criteria:
                    msg += f' Latest point found in track is {(point.timestamp - center.timestamp).total_seconds():.2f} after window center, {window_min_duration/2} was requested.'
                raise ValueError(msg)
            window = _core.TimePeriod(start=window_start.timestamp, stop=window_stop.timestamp)
            window.angle = center.angle
            windows.append(window)

        if len(windows) == 1:
            return windows[0]
        return windows


    def resample(self, sampletime, order='linear'):
        """Resample a position track.

        Parameters
        ----------
        sampletime : numerical
            The desired time between samples of the track.
        order : integer or string, default "linear"
            Sets the polynomial order of the interpolation.
            See `kind` argument of `scipy.interpolate.interp1d`
        """
        return ResampledTrack(
            sampletime=sampletime,
            time=self.relative_time,
            start_time=self.time_window.start,
            order=order,
            latitude=self.latitude,
            longitude=self.longitude,
        )


class TimestampedTrack(Track):
    def __init__(self, timestamps):
        self._timestamps = np.asarray(timestamps)

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def time_period(self):
        return _core.TimePeriod(start=self.timestamps[0], stop=self.timestamps[-1])

    @abc.abstractmethod
    def __getitem__(self, key):
        ...

    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        time = self.time_period.subperiod(time, start=start, stop=stop, center=center, duration=duration)
        if isinstance(time, _core.TimePeriod):
            start = bisect.bisect_left(self._timestamps, time.start)
            stop = bisect.bisect_right(self._timestamps, time.stop)
            return self[start:stop]

        # If it's not a period, it's a single datetime. Get the correct index and return said value.
        idx = bisect.bisect_right(self._timestamps, time)
        if idx == 0:
            return idx
        if idx == len(self):
            return idx - 1
        # Closest value interpolation.
        right_distance = (self._timestamps[idx] - time).total_seconds()
        left_distance = (time - self._timestamps[idx - 1]).total_seconds()
        if left_distance < right_distance:
            idx -= 1
        return self[idx]

    @property
    def relative_time(self):
        return np.asarray([
            (stamp - self.timestamps[0]).total_seconds()
            for stamp in self.timestamps
        ])

    def __len__(self):
        return self._timestamps.size

    def copy(self, deep=False):
        obj = super().copy()
        obj._timestamps = self._timestamps
        return obj


class ReferencedTrack(Track):
    def __init__(self, times, reference):
        self._times = times
        self._reference = reference

    def copy(self, deep=False):
        obj = super().copy(deep=deep)
        obj._times = self._times
        obj._reference = self._reference

    def __len__(self):
        return len(self._times)

    @property
    def relative_time(self):
        return self._times - self._times[0]

    @property
    def timestamps(self):
        return [
            self._reference + datetime.timedelta(seconds=t)
            for t in self._times
        ]

    @property
    def time_period(self):
        return _core.TimePeriod(
            start=self._reference + datetime.timedelta(seconds=self._times[0]),
            stop=self._reference + datetime.timedelta(seconds=self._times[-1]),
        )

    @abc.abstractmethod
    def __getitem__(self, key):
        ...

    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        time = self.time_period.subperiod(time, start=start, stop=stop, center=center, duration=duration)
        if isinstance(time, _core.TimePeriod):
            start = (time.start - self._reference).total_seconds()
            start = bisect.bisect_left(self._times, start)
            stop = (time.stop - self._reference).total_seconds()
            stop = bisect.bisect_right(self._times, stop)
            return self[start:stop]

        # If it's not a period, it's a single datetime. Get the correct index and return said value.
        time = (time - self._reference).total_seconds()
        idx = bisect.bisect_right(self._times, time)
        if idx == 0:
            return idx
        if idx == len(self):
            return idx - 1
        # Closest value interpolation.
        right_distance = self._times[idx] - time
        left_distance = time - self._times[idx - 1]
        if left_distance < right_distance:
            idx -= 1
        return self[idx]


class SampledTrack(Track):
    def __init__(self, sampletime, start, num_samples):
        self._sampletime = sampletime
        self._start = start
        self._num_samples = num_samples

    def copy(self, deep=False):
        obj = super().copy(deep=deep)
        obj._sampletime = self._sampletime
        obj._start = self._start
        obj._num_samples = self._num_samples
        return obj

    def __len__(self):
        return self._num_samples

    @property
    def timestamps(self):
        return [
            self._start + datetime.timedelta(seconds=idx * self._sampletime)
            for idx in range(self._num_samples)
        ]

    @property
    def relative_time(self):
        return np.arange(self._num_samples) * self._sampletime

    @property
    def time_period(self):
        return _core.TimePeriod(
            start=self._start,
            duration=self._num_samples * self._sampletime
        )

    @abc.abstractmethod
    def __getitem__(self, key):
        ...

    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        time = self.time_period.subperiod(time, start=start, stop=stop, center=center, duration=duration)

        if isinstance(time, _core.TimePeriod):
            start = (time.start - self._start).total_seconds()
            start = np.math.ceil(start / self._sampletime)
            stop = (time.stop - self._start).total_seconds()
            stop = np.math.floor(stop / self._sampletime)
            self[start:stop]

        # If it's not a period, it's a single datetime. Get the correct index and return said value.
        time = (time - self._start).total_seconds()
        # Closest value interpolation.
        idx = round(time * self._sampletime)
        return self[idx]


class TimestampedPositionTrack(TimestampedTrack):
    def __init__(self, timestamps, latitude, longitude, *args, **kwargs):
        super().__init__(timestamps=timestamps, *args, **kwargs)
        self._latitude = np.asarray(latitude)
        self._longitude = np.asarray(longitude)

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def heading(self):
        raise NotImplementedError()

    @property
    def speed(self):
        raise NotImplementedError()

    def copy(self, deep=False):
        obj = super().copy(deep=deep)
        obj._latitude = self._latitude
        obj._longitude = self._longitude
        return obj

    def __getitem__(self, key):
        latitude = self.latitude[key]
        longitude = self.longitude[key]
        timestamps = self.timestamps[key]

        if not isinstance(key, slice):
            return Position(
                latitude=latitude,
                longitude=longitude,
                timestamp=timestamps,
            )

        obj = self.copy(deep=False)
        obj._latitude = latitude
        obj._longitude = longitude
        obj._timestamps = timestamps
        return obj


class GPXTrack(TimestampedPositionTrack):
    def __init__(self, path):
        import gpxpy
        file = open(path, 'r')
        contents = gpxpy.parse(file)
        latitudes = []
        longitudes = []
        times = []
        for point in contents.get_points_data():
            latitudes.append(point.point.latitude)
            longitudes.append(point.point.longitude)
            times.append(point.point.time)
        super().__init__(timestamps=times, latitude=latitudes, longitude=longitudes)


class Blueflow(TimestampedTrack):
    def __init__(self, path):
        import pandas
        self.data = pandas.read_excel(path)
        super().__init__(timestamps=self.data['Timestamp [UTC]'].dt.tz_localize(dateutil.tz.UTC).dt.to_pydatetime())

    def copy(self, deep=False):
        obj = super().copy(deep=deep)
        obj.data = self.data
        return obj

    @property
    def latitude(self):
        latitude = self.data['Latitude [deg]']
        try:
            return latitude.to_numpy()
        except AttributeError:
            return latitude

    @property
    def longitude(self):
        longitude = self.data['Longitude [deg]']
        try:
            return longitude.to_numpy()
        except AttributeError:
            return longitude

    @property
    def heading(self):
        heading = self.data['Heading [deg]']
        try:
            return heading.to_numpy()
        except AttributeError:
            return heading

    @property
    def speed(self):
        speed = self.data['Speed over ground [kts]'] / one_knot
        try:
            return speed.to_numpy()
        except AttributeError:
            return speed

    def __getitem__(self, key):
        obj = self.copy(deep=False)
        obj.data = self.data.iloc[key]
        obj._timestamps = self.timestamps[key]
        if not isinstance(key, slice):
            return Position(
                latitude=obj.latitude,
                longitude=obj.longitude,
                timestamp=obj.timestamps,
            )
        return obj


class ResampledTrack(Track):
    def __init__(self, sampletime, time, latitude, longitude, start_time=None, order='linear', **kwargs):
        self.sampletime = sampletime
        self.order = order
        self.start_time = start_time
        self._unique_times, self._time_indices, self._duplicate_counts = np.unique(time, return_inverse=True, return_counts=True)
        n_samples = np.math.floor((self._unique_times[-1] - self._unique_times[0]) / self.sampletime) + 1
        self.time = np.arange(n_samples) * self.sampletime + self._unique_times[0]

        self.latitude = self._interpolate(latitude)
        self.longitude = self._interpolate(longitude)
        self._extra_data = list(kwargs.keys())
        for key, data in kwargs.items():
            setattr(self, key, self._interpolate(data))

    def _interpolate(self, data):
        if self._unique_times.size < self._time_indices.size:
            data = np.bincount(self._time_indices, data) / self._duplicate_counts
        interpolator = scipy.interpolate.interp1d(self._unique_times, data, kind=self.order)
        return interpolator(self.time)

    def smooth(self, time_constant, smoothing_method='median'):
        return SmoothTrack(
            time_constant=time_constant,
            sampletime=self.sampletime,
            smoothing_method=smoothing_method,
            start_time=self.start_time,
            latitude=self.latitude,
            longitude=self.longitude,
            **{key: getattr(self, key) for key in self._extra_data}
        )


class SmoothTrack(Track):
    def __init__(self, time_constant, sampletime, latitude, longitude, start_time=None, smoothing_method='median', **kwargs):
        self.time_constant = time_constant
        self.sampletime = sampletime
        self.start_time = start_time
        self.smoothing_method = smoothing_method

        self.latitude = self._smooth(latitude)
        self.longitude = self._smooth(longitude)
        for key, data in kwargs.items():
            setattr(self, key, self._smooth(data))

    def _smooth(self, data):
        if self.smoothing_method.lower() == 'median':
            smoother =  self._median_filter
        elif callable(self.smoothing_method):
            smoother = self.smoothing_method
        else:
            raise ValueError(f'Unknown smoothing method {self.smoothing_method}')
        return smoother(data=data, time_constant=self.time_constant, sampletime=self.sampletime)

    @staticmethod
    def _median_filter(data, time_constant, sampletime):
        import scipy.ndimage
        kernel = round(time_constant / sampletime)
        kernel = (1,) * (np.ndim(data) - 1) + (kernel, )
        return scipy.ndimage.percentile_filter(data, 50, kernel)
