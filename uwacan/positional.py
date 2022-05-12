"""Handles positional tracks.

This manages logs over positions of measurement objects via e.g. GPS.
Some of the operations include smoothing data, calculating distances,
and reading log files.
"""

import numpy as np
import scipy.interpolate
import scipy.signal
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84


def _callable_property(enclosing_class, name):
    def wrap_class(wrapper_class):
        attribute = f'_{enclosing_class.__name__}_{wrapper_class.__name__}_{name}'

        def getter(self):
            try:
                return getattr(self, attribute)
            except AttributeError:
                setattr(self, attribute, wrapper_class(self))
                return getattr(self, attribute)

        prop = property(fget=getter, doc=wrapper_class.__call__.__doc__)
        setattr(enclosing_class, name, prop)

        return wrapper_class
    return wrap_class


class Position:
    def __init__(self, latitude, longitude):
        self._latitude = latitude
        self._longitude = longitude

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    def distance_to(self, other):
        # TODO: Make sure that this broadcasts properly!
        # I expect that the geodesic doesn't broadcast, so you might need to loop. Look at np.nditer or np.verctorize
        ...


class Track(Position):
    def __init__(self, latitude, longitude, time):
        super().__init__(latitude=latitude, longitude=longitude)
        self._time = time

    @property
    def time(self):
        return self._time

    @property
    def speed(self):
        ...

    @property
    def heading(self):
        ...

    def time_range(self, start, stop, interpolate_boundaries=True):
        # TODO: do we want an indexing way to do this as well?
        ...


@_callable_property(Track, 'resample')
class ResampledTrack(Track):
    def __init__(self, track):
        self._track = track

    def __call__(self, sampletime, start=None, stop=None, order='linear'):
        """Resample a position track.

        Parameters
        ----------
        sampletime : numerical
            The desired time between samples of the track.
        start : numerical, optional
            The time in the track where the resampling should start.
            Default to use the entire track.
        stop : numerical, optional
            The time in the track where the resampling should stop.
            Default to use the entire track.
        order : integer or string, default "linear"
            Sets the polynomial order of the interpolation.
            See `kind` argument of `scipy.interpolate.interp1d`
        """
        # Handling uniqueness of samples.
        unique_times, time_indices, duplicate_counts = np.unique(self._track.time, return_inverse=True, return_counts=True)
        latitude = self._track.latitude
        longitude = self._track.longitude
        if unique_times.size < time_indices.size:
            # We have duplicate timestamps
            # bincount will count how often each value in `time_indices` occurs
            # i.e. how many times the same time exists in the time vector.
            # These values are weighted by the input values in latitude and longitude,
            # given as the second argument.
            # Finally the mean is calculated by dividing by the number of duplicates.
            latitude = np.bincount(time_indices, latitude) / duplicate_counts
            longitude = np.bincount(time_indices, longitude) / duplicate_counts

        # Handling subsampling of the domain
        if start is not None:
            # Get the index which is just to the right of the start value, then go left one.
            # This is to include at one additional sample if the requested time is not actually in the times.
            start_idx = max(np.searchsorted(unique_times, start, 'right') - 1, 0)
        else:
            start_idx = 0
            start = unique_times[0]

        if stop is not None:
            # Get the index which is just to the left of the start value, then go right one.
            # This is to include at one additional sample if the requested time is not actually in the times.
            stop_idx = min(np.searchsorted(unique_times, stop, 'left') + 1, len(unique_times))
        else:
            stop_idx = len(unique_times)
            stop = unique_times[-1]
        unique_times = unique_times[start_idx:stop_idx]
        latitude = latitude[start_idx:stop_idx]
        longitude = longitude[start_idx:stop_idx]

        # Performing the interpolation
        n_samples = np.math.floor((stop - start) / sampletime) + 1
        time = np.arange(n_samples) * sampletime + start
        latitude = scipy.interpolate.interp1d(unique_times, latitude, kind=order, bounds_error=False, fill_value=(latitude[0], latitude[-1]))
        longitude = scipy.interpolate.interp1d(unique_times, longitude, kind=order, bounds_error=False, fill_value=(longitude[0], longitude[-1]))
        latitude = latitude(time)
        longitude = longitude(time)

        # Store the new data
        self._latitude = latitude
        self._longitude = longitude
        self._time = time
        self.sampletime = sampletime
        return self


@_callable_property(Track, 'smooth')
class SmoothTrack(Track):
    def __init__(self, track):
        self._track = track

    @property
    def sampletime(self):
        try:
            return self._track.sampletime
        except AttributeError:
            raise TypeError('Cannot smooth a track which has no fixed sampling time!')

    def __call__(self, time_constant, smoothing_method='median'):
        """Smooths a position track.

        Parameters
        ----------
        time_constant : numerical
            Controlls the strength of the smoothing. Is in units of seconds.
        smoothing_method : string, default "median"
            Sets the smoothing method
        """
        self.time_constant = time_constant
        self.smoothing_method = smoothing_method

        latitude = self._track.latitude
        longitude = self._track.longitude

        if smoothing_method.lower() == 'median':
            kernel = round(self.time_constant / self.sampletime)
            kernel += (kernel + 1) % 2  # Median filter kernel has to be odd!

            smooth_latitude = scipy.signal.medfilt(latitude, kernel)
            smooth_longitude = scipy.signal.medfilt(longitude, kernel)

            # Recalculate the edges with a shrinking kernel instead of zero-padding
            kernel_offset = kernel // 2 + 1
            for idx in range(kernel // 2):
                smooth_latitude[idx] = np.median(latitude[:idx + kernel_offset])
                smooth_longitude[idx] = np.median(longitude[:idx + kernel_offset])
                smooth_latitude[-1 - idx] = np.median(latitude[-idx - kernel_offset:])
                smooth_longitude[-1 - idx] = np.median(longitude[idx - kernel_offset:])
        else:
            raise ValueError(f'Unknown smoothing method {smoothing_method}')

        self._latitude = smooth_latitude
        self._longitude = smooth_longitude
        self._time = self._track.time
        return self
