"""Some shared core functionality in the package.

This contains mostly wrappers around `xarray` objects, and some very basic functions.
A few of these are publicly available in the main package namespace. They should be
accessed from there if used externally, but from here if used internally.


Classes and functions only exposed here
---------------------------------------
.. autosummary::
    :toctree: generated

    time_to_np
    time_to_datetime
    TimeWindow
    xrwrap
    DataArrayWrap
    DatasetWrap

Classes and functions exposed in the main package namespace
-----------------------------------------------------------
.. autosummary::

    uwacan.TimeData
    uwacan.FrequencyData
    uwacan.TimeFrequencyData
    uwacan.Transit
    uwacan.dB

"""


import numpy as np
import xarray as xr
import collections.abc
import pendulum

__all__ = [
    "TimeWindow",
    "dB",
    "TimeData",
    "FrequencyData",
    "TimeFrequencyData",
    "Transit",
]


def time_to_np(input):
    """Convert a time to `numpy.datetime64`."""
    if isinstance(input, np.datetime64):
        return input
    if not isinstance(input, pendulum.DateTime):
        input = time_to_datetime(input)
    return np.datetime64(input.in_tz('UTC').naive())


def time_to_datetime(input, fmt=None, tz="UTC"):
    """Convert datetimes to the same internal format.

    This function takes a few types of input and tries to convert
    the input to a pendulum.DateTime.
    - Any datetime-like input will be converted directly.
    - np.datetime64 and Unix timestamps are treated similarly.
    - Strings are parsed with ``fmt`` if given, otherwise a few different common formats are tried.

    Parameters
    ----------
    input : datetime-like, string, or numeric.
        The input data specifying the time.
    fmt : string, optional
        Optional format detailing how to parse input strings. See `pendulum.from_format`.
    tz : string, default "UTC"
        The timezone of the input time for parsing, and the output time zone.
        Unix timestamps have no timezone, and np.datetime64 only supports UTC.

    Returns
    -------
    time : pendulum.DateTime
        The converted time.
    """
    try:
        return pendulum.instance(input, tz=tz)
    except AttributeError as err:
        if "object has no attribute 'tzinfo'" in str(err):
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
        if tz != "UTC":
            raise ValueError("Numpy datetime64 values should always be stored in UTC")
        input = input.astype('timedelta64') / np.timedelta64(1, 's')  # Gets the time as a timestamp, will parse nicely below.

    try:
        return pendulum.from_timestamp(input, tz=tz)
    except TypeError as err:
        if 'object cannot be interpreted as an integer' in str(err):
            pass
        else:
            raise
    return pendulum.parse(input, tz=tz)


class TimeWindow:
    """Describes a start and stop point in time.

    Give two and only two of the four basic parameters.
    Less than two will not fully define a window, while
    more than two will overdetermine the window.

    Parameters
    ----------
    start : time_like
        A window that starts at this time
    stop : time_like
        A window stat stops at this time
    center : time_like
        A window centered at this time
    duration : float
        A window with this duration, in seconds
    extend : float
        Extend the window defined by two of the four above
        with this much at each end, in seconds.
    """

    def __init__(self, start=None, stop=None, center=None, duration=None, extend=None):
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

        if extend is not None:
            _start = _start.subtract(seconds=extend)
            _stop = _stop.add(seconds=extend)

        self._window = pendulum.interval(_start, _stop)

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        """Select a smaller window of time.

        Parameters
        ----------
        time : time_window_like
            An object that will be used to extract start and stop times.
        start : time_like
            A new window that starts at this time.
            Give ``True`` to use the start of the existing window.
        stop : time_like
            A new window stat stops at this time
            Give ``True`` to use the stop of the existing window.
        center : time_like
            A new window centered at this time
            Give ``True`` to use the center of the existing window.
        duration : float
            A new window with this duration, in seconds
        extend : float
            Extend the new window defined by two of the four above
            with this much at each end, in seconds.

        Notes
        -----
        This takes the same basic inputs as `TimeWindow`, defining a window
        with two out of four of ``start``, ``stop``, ``center``, and ``duration``.
        Additionally, one of ``start``, ``stop``, ``center`` can be given as ``True``
        instead of an actual time to use the times of the existing window.
        If only one of ``start`` and ``stop`` is given, the other one is filled from
        the existing window.

        If a single positional argument is given, it should be time_window_like,
        i.e., have a defined start and stop time, which will then be used.
        This can be one of `TimeWindow`, `pendulum.Interval`, and `xarray.Dataset`.
        If it is a dataset, it must have a time attribute, and its minimum and maximum
        will be used as the start and stop for the new window.
        """
        if time is None:
            # Period specified with keyword arguments, convert to period.
            if (start, stop, center, duration).count(None) == 3:
                # Only one argument which has to be start or stop, fill the other from self.
                if start is not None:
                    window = type(self)(start=start, stop=self.stop, extend=extend)
                elif stop is not None:
                    window = type(self)(start=self.start, stop=stop, extend=extend)
                else:
                    raise TypeError('Cannot create subwindow from arguments')
            elif duration is not None and True in (start, stop, center):
                if start is True:
                    window = type(self)(start=self.start, duration=duration, extend=extend)
                elif stop is True:
                    window = type(self)(stop=self.stop, duration=duration, extend=extend)
                elif center is True:
                    window = type(self)(center=self.center, duration=duration, extend=extend)
                else:
                    raise TypeError('Cannot create subwindow from arguments')
            else:
                # The same types explicit arguments as the normal constructor
                window = type(self)(start=start, stop=stop, center=center, duration=duration, extend=extend)
        elif isinstance(time, type(self)):
            window = time
        elif isinstance(time, pendulum.Interval):
            window = type(self)(start=time.start, stop=time.end, extend=extend)
        elif isinstance(time, xr.Dataset):
            window = type(self)(start=time.time.min(), stop=time.time.max(), extend=extend)
        else:
            # It's not a period, so it should be a single datetime. Parse or convert, check validity.
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
        """The start of this window."""
        return self._window.start

    @property
    def stop(self):
        """The stop of this window."""
        return self._window.end

    @property
    def center(self):
        """The center of this window."""
        return self.start.add(seconds=self._window.total_seconds() / 2)

    @property
    def duration(self):
        """The duration of this window, in seconds."""
        return self._window.total_seconds()

    def __contains__(self, other):
        if isinstance(other, type(self)):
            other = other._window
        if isinstance(other, pendulum.Interval):
            return other.start in self._window and other.end in self._window
        return other in self._window


class xrwrap:
    """Wrapper around `xarray` objects.

    This base class exists to delegate work to our internal
    `xarray` objects.
    """

    def __init__(self, data):
        if data is None:
            return
        if isinstance(data, xrwrap):
            data = data.data
        self._data = data

    @property
    def data(self):
        """The contained data."""
        return self._data

    def _transfer_attributes(self, other):
        """Copy attributes form self to other.

        This is useful to when creating a new copy of the same instance
        but with new data. The intent is for subclasses to extend
        this function to preserve attributes of the class that
        are not stored within the data variable.
        Note that this does not create a new instance of the class,
        so ``other`` should already be instantiated with data.
        The typical scheme to create a new instance from a new data structure
        is::

            new = type(self)(data)
            self._transfer_attributes(new)
            return new

        """
        pass

    @classmethod
    def _select_wrapper(cls, data):
        """Select an appropriate wrapper for `xarrayr.DataArray`.

        This classmethod inspects the DataArray and returns
        a wrapper class for it. The base implementation is
        to return the class on which this method was called.
        Subclasses can extend this to choose another wrapper
        as appropriate.
        """
        return cls

    def __array_wrap__(self, data, context=None, transfer_attributes=True):
        """Wrap output data in in a new object.

        This takes data from some processing and wraps it back into a
        suitable class. If no suitable class was found, the data is
        returned as is.
        """
        cls = self._select_wrapper(data)
        if cls is None:
            return data
        new = cls(data)
        if transfer_attributes:
            self._transfer_attributes(new)
        return new

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, drop_allnan=True, **indexers_kwargs):
        """Select a subset of the data from the coordinate labels.

        The selection is easiest done with keywords, e.g. ``obj.sel(sensor="Colmar 1")``
        to select a specific sensor. For numerical coordinates, ``method="nearest"`` can
        be quite useful. Use a slice to select a range of values, e.g.,
        ``obj.sel(frequency=slice(10, 100))``.

        For more details, see `xarray.DataArray.sel` and `xarray.Dataset.sel`.
        """
        new = self.data.sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs
        )
        if drop_allnan:
            new = new.where(~new.isnull(), drop=True)
        return self.__array_wrap__(new)

    def isel(self, indexers=None, drop=False, missing_dims='raise', drop_allnan=True, **indexers_kwargs):
        """Select a subset of the data from the coordinate indices.

        The selection is easiest done with keywords, e.g. ``obj.sel(sensor=0)``
        to select the zeroth sensor. Use a slice to select a range of values, e.g.,
        ``obj.sel(frequency=slice(10, 100))``.

        For more details, see `xarray.DataArray.isel` and `xarray.Dataset.isel`.
        """
        new = self.data.isel(
            indexers=indexers,
            drop=drop,
            missing_dims=missing_dims,
            **indexers_kwargs
        )
        if drop_allnan:
            new = new.where(~new.isnull(), drop=True)
        return self.__array_wrap__(new)

    @property
    def coords(self):
        """The coordinate (dimension) arrays for this data.

        Refer to `xarray.DataArray.coords` and `xarray.Dataset.coords`.
        """
        return self.data.coords

    @property
    def dims(self):
        """The dimensions of this data.

        Refer to `xarray.DataArray.dims` and `xarray.Dataset.dims`.
        """
        return self.data.dims

    def groupby(self, group):
        for label, group in self.data.groupby(group, squeeze=False):
            yield label, self.__array_wrap__(group.squeeze())


class DataArrayWrap(xrwrap, np.lib.mixins.NDArrayOperatorsMixin):
    """Wrapper around `xarray.DataArray`.

    This base class exists to wrap functionality in `xarray.DataArray`,
    and numerical operations from `numpy`.
    """

    _coords_set_by_init = set()

    def __init__(self, data, dims=(), coords=None):
        if data is None:
            return
        if isinstance(data, DataArrayWrap):
            data = data.data
        if not isinstance(data, xr.DataArray):
            if isinstance(dims, str):
                dims = [dims]
            if dims is None:
                dims = ()
            if np.ndim(data) != np.size(dims):
                raise ValueError(f"Dimension names '{dims}' for {type(self).__name__} does not match data with {np.ndim(data)} dimensions")
            data = xr.DataArray(data, dims=dims)
        if coords is not None:
            data = data.assign_coords(**{name: coord for (name, coord) in coords.items() if name not in self._coords_set_by_init})
        self._data = data

    def __array__(self, dtype=None):
        """Casts this object into a `numpy.ndarray`."""
        return self.data.__array__(dtype=dtype)

    @staticmethod
    def _implements_np_func(np_func):
        """Tag implementations of `numpy` functions.

        We use the ``__array_function__`` interface to implement many
        `numpy` functions. This decorator will only tag an implementation
        function with which `numpy` function it implements.
        """
        def decorator(func):
            func._implements_np_func = np_func
            return func
        return decorator

    def __init_subclass__(cls):
        """Set up the `numpy` implementations for a class.

        This will run when a subclass is defined, and
        check if there are any methods in it that are tagged
        with a numpy implementation. All those implementations
        will be stored in a class-level dictionary.
        """
        implementations = {}
        for name, value in cls.__dict__.items():
            if callable(value) and hasattr(value, "_implements_np_func"):
                implementations[value._implements_np_func] = value
        cls._np_func_implementations = implementations

    def __array_function__(self, func, types, args, kwargs):
        """Interfaces with numpy functions.

        This will run when general numpy functions are used on objects
        of this class. We have stored tagged implementations in class
        dictionaries, so we can check if there is an explicit implementation.
        We have no actual method which does this, so we go through the ``mro``
        manually.

        If no explicit implementation is found, we try replacing all wrappers
        with their `xarray.DataArray` objects, and call the function on them
        instead.
        """
        for cls in self.__class__.mro():
            if hasattr(cls, "_np_func_implementations"):
                if func in cls._np_func_implementations:
                    func = cls._np_func_implementations[func]
                    break
        else:
            # We couldn't find an explicit implementation.
            # Try replacing all _DataWrapper with their data and calling the function.
            args = (arg.data if isinstance(arg, DataArrayWrap) else arg for arg in args)
            out = func(*args, **kwargs)
            if not isinstance(out, xr.DataArray):
                try:
                    out = self.data.__array_wrap__(out)
                except:
                    # We cannot wrap this in an xarray, then we cannot wrap in our own wrapper.
                    return out
            out = self.__array_wrap__(out)
            return out
        return func(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Interfaces with `numpy.ufunc`.

        Many functions in numpy are ufuncs. `xarray.DataArray.__array_ufunc__` will
        do the heavy lifting here.
        """
        inputs = (arg.data if isinstance(arg, DataArrayWrap) else arg for arg in inputs)
        return self.__array_wrap__(self.data.__array_ufunc__(ufunc, method, *inputs, **kwargs))

    @_implements_np_func(np.mean)
    def mean(self, dim=..., **kwargs):
        """Average of this data, along some dimension.

        See `xarray.DataArray.mean` for more details.
        """
        return self.__array_wrap__(self.data.mean(dim, **kwargs))

    @_implements_np_func(np.sum)
    def sum(self, dim=..., **kwargs):
        """Sum of this data, along some dimension.

        See `xarray.DataArray.sum` for more details.
        """
        return self.__array_wrap__(self.data.sum(dim, **kwargs))

    @_implements_np_func(np.std)
    def std(self, dim=..., **kwargs):
        """Standard deviation of this data, along some dimension.

        See `xarray.DataArray.std` for more details.
        """  # noqa: D401
        return self.__array_wrap__(self.data.std(dim, **kwargs))

    @_implements_np_func(np.max)
    def max(self, dim=..., **kwargs):
        """Maximum of this data, along some dimension.

        See `xarray.DataArray.max` for more details.
        """
        return self.__array_wrap__(self.data.max(dim, **kwargs))

    @_implements_np_func(np.min)
    def min(self, dim=..., **kwargs):
        """Minimum of this data, along some dimension.

        See `xarray.DataArray.min` for more details.
        """
        return self.__array_wrap__(self.data.min(dim, **kwargs))

    def apply(self, func, *args, **kwargs):
        """Apply some function to the contained data.

        This calls the supplied function with the `xarray.DataArray`
        in this object, then wraps the output in a similar container again.
        """
        data = func(self.data, *args, **kwargs)
        return self.__array_wrap__(data)

    def reduce(self, func, dim, **kwargs):
        """Apply a reduction function along some dimension in this data.

        See `xarray.DataArray.reduce` for more details.
        """
        data = self.data.reduce(func=func, dim=dim, **kwargs)
        return self.__array_wrap__(data)

DataArrayWrap.__init_subclass__()


class DatasetWrap(xrwrap, collections.abc.MutableMapping):
    """Wraps `xarray.Dataset` objects.

    This wraps a dataset by passing indexing to the underlying dataset
    indexing, and mimics the `xarray` attribute access by passing
    attribute access to indexing if the attribute exists in the dataset.
    Using a MutableMapping from collections enables lots of dict-style
    iteration.
    """

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError as e:
            raise KeyError(*e.args) from None

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._data.variables))


class TimeData(DataArrayWrap):
    """Handing data which varies over time.

    This class is mainly used to wrap time-signals of sampled sounds.
    As such, the time data is assumed to be sampled at a constant samplerate.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the time data.
    start_time : time_like, optional
        The start time for the first sample in the signal.
        This should ideally be a proper time type, but it will be parsed if it is a string.
        Defaults to "now" if not given.
    samplerate : float, optional
        The samplerate for this data, in Hz.
        If the ``data`` is a `numpy.ndarray`, this has to be given.
        If the ``data`` is a `xarray.DataArray` which already has a time coordinate,
        this can be omitted.
    dims : str or [str], default="time"
        The dimensions of the data. Must have the same length as the number of dimensions in the data.
        Only used for `numpy` inputs.
    coords : `xarray.DataArray.coords`
        Additional coordinates for this data.
    """

    _coords_set_by_init = {"time"}

    def __init__(self, data, start_time=None, samplerate=None, dims="time", coords=None, **kwargs):
        super().__init__(
            data,
            dims=dims,
            coords=coords,
            **kwargs
        )

        if samplerate is not None:
            if start_time is None:
                if "time" in self.data.coords:
                    start_time = self.data.time[0].item()
                else:
                    start_time = "now"
            n_samples = self.data.sizes["time"]
            start_time = time_to_np(start_time)
            offsets = np.arange(n_samples) * 1e9 / samplerate
            time = start_time + offsets.astype("timedelta64[ns]")
            self.data.coords["time"] = ("time", time, {"rate": samplerate})

    @classmethod
    def _select_wrapper(cls, data):
        if "time" in data.coords:
            return super()._select_wrapper(data)
        # This is not time data any more, just return the plain xr.DataArray
        return None

    @property
    def time(self):
        """Time coordinates for this data."""
        return self.data.time

    @property
    def samplerate(self):
        """Samplerate for the time coordinates."""
        return self.data.time.rate

    @property
    def time_window(self):
        """A `TimeWindow` describing when the data start and stops."""
        # Calculating duration from number and rate means the stop points to the sample after the last,
        # which is more intuitive when considering signal durations etc.
        return TimeWindow(
            start=self.data.time.data[0],
            duration=self.data.sizes['time'] / self.samplerate,
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        """Select a subset of the data over time.

        See `TimeWindow.subwindow` for details on the parameters.
        """
        original_window = self.time_window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        if isinstance(new_window, TimeWindow):
            start = (new_window.start - original_window.start).total_seconds()
            stop = (new_window.stop - original_window.start).total_seconds()
            # Indices assumed to be seconds from start
            start = int(np.floor(start * self.samplerate))
            stop = int(np.ceil(stop * self.samplerate))
            idx = slice(start, stop)
        else:
            idx = (new_window - original_window.start).total_seconds()
            idx = round(idx * self.samplerate)

        selected_data = self.data.isel(time=idx)
        new = type(self)(selected_data)
        self._transfer_attributes(new)
        return new

    def listen(self, downsampling=1, upsampling=None, headroom=6, **kwargs):
        """Play back this time data over speakers.

        This tries to play the time data as audio using the `sounddevice` package.
        The audio will be centered at 0 and normalized before playback.

        Parameters
        ----------
        downsampling : float, optional
            Artificially uses a lower samplerate in playback to slow
            down the audio, lowering the pitch of the content.
        upsampling : int, optional
            Decimates the data by selecting every Nth sample, speeding
            up the audio and raising the pitch of the content.
        headroom : float, default 6
            How much headroom to leave in the normalization, in dB.
        **kwargs : dict, optional
            Remaining keyword arguments are passed to `sounddevice.play`.
            The most useful arguments are ``blocking=True``, and ``device``.
        """
        import sounddevice as sd
        sd.stop()
        data = self.data
        if upsampling:
            data = data[::upsampling]
        scaled = data - data.mean()
        scaled = scaled / np.max(np.abs(scaled)) * 10 ** (-headroom / 20)
        sd.play(scaled, samplerate=round(self.samplerate / downsampling), **kwargs)


class FrequencyData(DataArrayWrap):
    """Handing data which varies over frequency.

    This class is mainly used to wrap spectra of sampled sounds.
    Typically, this is represented as power spectral densities,
    but other data types are also possible.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the frequency data.
    frequency : array_like, optional
        The frequencies corresponding to the data. Mandatory if ``data`` is a `numpy.ndarray`.
    bandwidth : array_like, optional
        The bandwidth of each data point. Can be an array with per-frequency
        bandwidth or a single value valid for all frequencies.
    dims : str or [str], default="frequency"
        The dimensions of the data. Must have the same length as the number of dimensions in the data.
        Only used for `numpy` inputs.
    coords : `xarray.DataArray.coords`
        Additional coordinates for this data.
    """

    _coords_set_by_init = {"frequency", "bandwidth"}

    def __init__(self, data, frequency=None, bandwidth=None, dims="frequency", coords=None, **kwargs):
        super().__init__(
            data,
            dims=dims,
            coords=coords,
            **kwargs
        )
        if frequency is not None:
            self.data.coords["frequency"] = frequency
        if bandwidth is not None:
            bandwidth = np.broadcast_to(bandwidth, np.shape(frequency))
            self.data.coords["bandwidth"] = ("frequency", bandwidth)

    @classmethod
    def _select_wrapper(cls, data):
        if "frequency" in data.coords:
            return super()._select_wrapper(data)
        # This is not frequency data any more, just return the plain xr.DataArray
        return None

    @property
    def frequency(self):
        """"The frequencies for the data."""
        return self.data.frequency

    def estimate_bandwidth(self):
        """Estimate the bandwidth of the frequency vector.

        This tries to determine if the frequency vector is linearly
        or logarithmically spaced, then uses either linear or logarithmic
        bandwidth estimation.

        Returns
        -------
        bandwidth : `xarray.DataArray`
            The estimated bandwidth.
        """
        frequency = np.asarray(self.frequency)
        # Check if the frequency array seems linearly or logarithmically spaced
        if frequency[0] == 0:
            diff = frequency[2:] - frequency[1:-1]
            frac = frequency[2:] / frequency[1:-1]
        else:
            diff = frequency[1:] - frequency[:-1]
            frac = frequency[1:] / frequency[:-1]
        diff_err = np.std(diff) / np.mean(diff)
        frac_err = np.std(frac) / np.mean(frac)
        # Note: if there are three values and the first is zero, the std is 0 for both.
        # The equals option makes us end up in the linear frequency case.
        if diff_err <= frac_err:
            # Central differences, with forwards and backwards at the ends
            central = (frequency[2:] - frequency[:-2]) / 2
            first = frequency[1] - frequency[0]
            last = frequency[-1] - frequency[-2]
        else:
            # upper edge is at sqrt(f_{l+1} * f_l), lower edge is at sqrt(f_{l-1} * f_l)
            # the difference simplifies as below.
            central = (frequency[2:]**0.5 - frequency[:-2]**0.5) * frequency[1:-1]**0.5
            # extrapolating to one bin below lowest and one above highest using constant ratio
            # the expression above then simplifies to the expressions below
            first = (frequency[1] - frequency[0]) * (frequency[0] / frequency[1])**0.5
            last = (frequency[-1] - frequency[-2]) * (frequency[-1] / frequency[-2])**0.5
        bandwidth = np.concatenate([[first], central, [last]])
        return xr.DataArray(bandwidth, coords={'frequency': self.frequency})


class TimeFrequencyData(TimeData, FrequencyData):
    """Handing data which varies over time and frequency.

    This class is mainly used to wrap spectrogram-like data.
    There are no processing implemented in this class, but
    subclasses are free to add processing methods, custom
    initialization, or instantiation in class methods.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the time-frequency data.
    start_time : time_like, optional
        The start time for the first sample in the signal.
        This should ideally be a proper time type, but it will be parsed if it is a string.
        Defaults to "now" if not given.
    samplerate : float, optional
        The samplerate for this data, in Hz. This is not the samplerate of the underlying time signal,
        but time steps of the time axis on the time-frequency data.
        If the `data` is a `numpy.ndarray`, this has to be given.
        If the `data` is a `xarray.DataArray` which already has a time coordinate,
        this can be omitted.
    frequency : array_like, optional
        The frequencies corresponding to the data. Mandatory if `data` is a `numpy.ndarray`.
    bandwidth : array_like, optional
        The bandwidth of each data point. Can be an array with per-frequency
        bandwidth or a single value valid for all frequencies.
    dims : str or [str], optional
        The dimensions of the data. Must have the same length as the number of dimensions in the data.
        Mandatory used for `numpy` inputs, not used for `xarray` inputs.
    coords : `xarray.DataArray.coords`
        Additional coordinates for this data.
    """

    _coords_set_by_init = {"time", "frequency", "bandwidth"}
    def __init__(self, data, start_time=None, samplerate=None, frequency=None, bandwidth=None, dims=None, coords=None, **kwargs):
        super().__init__(
            data,
            dims=dims,
            coords=coords,
            start_time=start_time, samplerate=samplerate,
            frequency=frequency, bandwidth=bandwidth,
            **kwargs
        )

    @classmethod
    def _select_wrapper(cls, data):
        if "frequency" not in data.coords:
            # It's not frequency-data, but it might be time data
            return TimeData._select_wrapper(data)
        if "time" not in data.coords:
            # It's not time-data, but it might be frequency data
            return FrequencyData._select_wrapper(data)
        return super()._select_wrapper(data)


class Transit:
    """Container for recorded ship transits.

    This class is responsible for bundling recordings and position tracks.
    Note that this class does not take `TimeData` as the input.
    The track and recording will be restricted to a time window which
    both of them covers.

    Attributes
    ----------
    recording : `recordings.Recording`
        A recording of the ship transit.
    track : `positional.Track`
        The position track of the ship.
    """

    def __init__(self, recording, track):
        start = max(recording.time_window.start, track.time_window.start)
        stop = min(recording.time_window.stop, track.time_window.stop)

        self.recording = recording.subwindow(start=start, stop=stop)
        self.track = track.subwindow(start=start, stop=stop)

    @property
    def time_window(self):
        """A `TimeWindow` describing when the data start and stops."""
        rec_window = self.recording.time_window
        track_window = self.track.time_window
        return TimeWindow(start=max(rec_window.start, track_window.start), stop=min(rec_window.stop, track_window.stop))

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        """Select a subset of the data over time.

        See `TimeWindow.subwindow` for details on the parameters.
        """
        subwindow = self.time_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        rec = self.recording.subwindow(subwindow)
        track = self.track.subwindow(subwindow)
        return type(self)(recording=rec, track=track)


def dB(x, power=True, safe_zeros=True, ref=1):
    """Calculate the decibel of an input value.

    Parameters
    ----------
    x : numeric
        The value to take the decibel of
    power : boolean, default True
        Specifies if the input is a power-scale quantity or a root-power quantity.
        For power-scale quantities, the output is 10 log(x), for root-power quantities the output is 20 log(x).
        If there are negative values in a power-scale input, the handling can be controlled as follows:
        - `power='imag'`: return imaginary values
        - `power='nan'`: return nan where power < 0
        - `power=True`: as `nan`, but raises a warning.
    safe_zeros : boolean, default True
        If this option is on, all zero values in the input will be replaced with the smallest non-zero value.
    ref : numeric
        The reference unit for the decibel. Note that this should be in the same unit as the `x` input,
        e.g., if `x` is a power, the `ref` value might need squaring.
    """
    if isinstance(x, DataArrayWrap):
        return x.apply(dB, power=power, safe_zeros=safe_zeros, ref=ref)

    if safe_zeros and np.size(x) > 1:
        nonzero = x != 0
        min_value = np.nanmin(abs(xr.where(nonzero, x, np.nan)))
        x = xr.where(nonzero, x, min_value)
    if power:
        if np.any(x < 0):
            if power == 'imag':
                return 10 * np.log10(x + 0j)
            if power == 'nan':
                return 10 * np.log10(xr.where(x > 0, x, np.nan))
        return 10 * np.log10(x / ref)
    else:
        return 20 * np.log10(np.abs(x) / ref)
