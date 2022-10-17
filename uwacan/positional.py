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
# from . import timestamps
import datetime
import dateutil
import bisect
geod = Geodesic.WGS84


one_knot = 1.94384


def parse_timestamp(stamp):
    return dateutil.parser.parse(stamp)
    stamp = ''.join(c for c in stamp if c in '1234567890')
    num_chars = len(stamp)
    year = int(stamp[0:4])
    month = int(stamp[4:6]) if num_chars > 4 else 1
    day = int(stamp[6:8]) if num_chars > 6 else 1
    hour = int(stamp[8:10]) if num_chars > 8 else 0
    minute = int(stamp[10:12]) if num_chars > 10 else 0
    second = int(stamp[12:14]) if num_chars > 12 else 0
    microsecond = int(stamp[14:18].ljust(6, '0')) if num_chars > 12 else 0
    return datetime.datetime(year, month, day, hour, minute, second, microsecond)


def wrap_angle(angle):
    '''Wraps an angle to (-180, 180].'''
    return 180 - np.mod(180 - angle, 360)


class TimeWindow:
    def __init__(self, start=None, stop=None, center=None, duration=None):
        if isinstance(start, str):
            start = parse_timestamp(start)
        if isinstance(stop, str):
            stop = parse_timestamp(stop)
        if isinstance(center, str):
            center = parse_timestamp(center)

        if None not in (start, stop):
            self._start = start
            self._stop = stop
            start = stop = None
        elif None not in (center, duration):
            self._start = center - datetime.timedelta(seconds=duration / 2)
            self._stop = center + datetime.timedelta(seconds=duration / 2)
            center = duration = None
        elif None not in (start, duration):
            self._start = start
            self._stop = start + datetime.timedelta(seconds=duration)
            start = duration = None
        elif None not in (stop, duration):
            self._stop = stop
            self._start = stop - datetime.timedelta(seconds=duration)
            stop = duration = None
        elif None not in (start, center):
            self._start = start
            self._stop = start + (center - start) / 2
            start = center = None
        elif None not in (stop, center):
            self._stop = stop
            self._start = stop - (stop - center) / 2
            stop = center = None

        if (start, stop, center, duration) != (None, None, None, None):
            raise ValueError('Cannot input more than two input arguments to a time window!')

    def __repr__(self):
        return f'TimeWindow(start={self.start}, stop={self.stop})'

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def duration(self):
        return (self.stop - self.start).total_seconds()

    @property
    def center(self):
        return self.start + datetime.timedelta(seconds=self.duration / 2)

    def __contains__(self, other):
        if isinstance(other, TimeWindow):
            return (other.start >= self.start) and (other.stop <= self.stop)
        if isinstance(other, Position):
            return self.start <= other.timestamp <= self.stop
        if isinstance(other, datetime.datetime):
            return self.start <= other <= self.stop
        raise TypeError(f'Cannot check if {other.__class__.__name__} is within a time window')


class Position:
    def __init__(self, *args, **kwargs):
        latitude = kwargs.pop('latitude', None)
        longitude = kwargs.pop('longitude', None)
        timestamp = kwargs.pop('timestamp', None)

        if len(args) % 2:
            if timestamp is not None:
                raise TypeError("Position got multiple values for argument 'timestamp'")
            *args, timestamp = args

        if len(args) != 0:
            if latitude is not None:
                raise TypeError("Position got multiple values for argument 'longitude'")
            if longitude is not None:
                raise TypeError("Position got multiple values for argument 'latitude'")

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

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

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
    def time_window(self):
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

    def time_range(self, start=None, stop=None, center=None, duration=None):
        """Restrict the time range.

        This gets a window to the same time signal over a specified time period.
        The time period can be specified with any combination of two of the input
        parameters. The times can be specified either as `datetime` objects,
        or as strings on following format: `YYYYMMDDhhmmssffffff`.
        The microsecond part can be optionally omitted, and any non-digit characters
        are removed. Some examples include
        - 20220525165717
        - 2022-05-25_16-57-17
        - 2022/05/25 16:57:17.123456

        Parameters
        ----------
        start : datetime or string
            The start of the time window
        stop : datetime or string
            The end of the time window
        center : datetime or string
            The center of the time window
        duration : numeric
            The total duration of the time window, in seconds.
        """
        time_window = TimeWindow(
            start=start,
            stop=stop,
            center=center,
            duration=duration,
        )
        return self[time_window]


    @abc.abstractmethod
    def __getitem__(self, key):
        """Restrict the time range.

        This gets the same signal but restricted to a time range specified
        with a TimeWindow object.

        Parameters
        ----------
        window : `TimeWindow`
            The time window to restrict to.
        """
        ...

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
            for point in reversed(self[:pre_cpa_window_centers[-1].timestamp]):
                if abs(reference_point.angle_between(cpa, point)) >= -angle:
                    point.angle = angle
                    pre_cpa_window_centers.append(point)
                    break
            else:
                raise ValueError(f'Could not find window centered at {angle} degrees, found at most {-abs(reference_point.angle_between(cpa, point))}. Include additional early track data.')

        for angle in post_cpa_angles:
            for point in self[post_cpa_window_centers[-1].timestamp:]:
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
            for point in reversed(self[:center.timestamp]):
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
            for point in self[center.timestamp:]:
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

            windows.append(TimeWindow(start=window_start.timestamp, stop=window_stop.timestamp))

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
    def time_window(self):
        return TimeWindow(start=self.timestamps[0], stop=self.timestamps[-1])

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

    @abc.abstractmethod
    def __getitem__(self, key):
        if isinstance(key, TimeWindow):
            key = slice(key.start, key.stop)

        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if isinstance(start, datetime.datetime):
                start = bisect.bisect_left(self._timestamps, start)
            if isinstance(stop, datetime.datetime):
                stop = bisect.bisect_right(self._timestamps, stop)
            return slice(start, stop)

        if isinstance(key, datetime.datetime):
            idx = bisect.bisect_right(self._timestamps, key)
            if idx == 0:
                return idx
            if idx == len(self):
                return idx - 1
            right_distance = (self._timestamps[idx] - key).total_seconds()
            left_distance = (key - self._timestamps[idx - 1]).total_seconds()
            if left_distance < right_distance:
                idx -= 1
            return idx

        return key


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
    def time_window(self):
        return TimeWindow(
            start=self._reference + datetime.timedelta(seconds=self._times[0]),
            stop=self._reference + datetime.timedelta(seconds=self._times[-1]),
        )

    @abc.abstractmethod
    def __getitem__(self, key):
        if isinstance(key, TimeWindow):
            key = slice(key.start, key.stop)

        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if isinstance(start, datetime.datetime):
                start = (start - self._reference).total_seconds()
                start = bisect.bisect_left(self._times, start)
            if isinstance(stop, datetime.datetime):
                stop = (stop - self._reference).total_seconds()
                stop = bisect.bisect_right(self._times, stop)
            return slice(start, stop)

        if isinstance(key, datetime.datetime):
            key = (key - self._reference).total_seconds()
            idx = bisect.bisect_right(self._times, key)
            if idx == 0:
                return idx
            if idx == len(self):
                return idx - 1
            right_distance = self._times[idx] - key
            left_distance = key - self._times[idx - 1]
            if left_distance < right_distance:
                idx -= 1
            return idx

        return key


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
        return np.ararnge(self._num_samples) * self._sampletime

    @property
    @abc.abstractmethod
    def time_window(self):
        return TimeWindow(
            start=self._start,
            duration=self._num_samples * self._sampletime
        )

    @abc.abstractmethod
    def __getitem__(self, key):
        if isinstance(key, TimeWindow):
            key = slice(key.start, key.stop)

        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if isinstance(start, datetime.datetime):
                start = (start - self._start).total_seconds()
                start = np.math.ceil(start / self._sampletime)
            if isinstance(stop, datetime.datetime):
                stop = (stop - self._start).total_seconds()
                stop = np.math.floor(stop / self._sampletime)
            return slice(start, stop)

        if isinstance(key, datetime.datetime):
            key = (key - self._start).total_seconds()
            idx = round(key * self._sampletime)
            return idx

        return key


class GPXTrack(TimestampedTrack):
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
        super().__init__(timestamps=times)
        self._latitude = np.asarray(latitudes)
        self._longitude = np.asarray(longitudes)

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
        key = super().__getitem__(key)
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
        key = super().__getitem__(key)
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
