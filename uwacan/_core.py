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
    if isinstance(input, np.datetime64):
        return input
    if not isinstance(input, pendulum.DateTime):
        input = time_to_datetime(input)
    return np.datetime64(input.in_tz('UTC').naive())


def time_to_datetime(input, fmt=None, tz="UTC"):
    """Converts datetimes to the same internal format.

    This function takes a few types of input and tries to convert
    the input to a pendulum.DateTime.
    - Any datetime-like input will be converted directly.
    - np.datetime64 and Unix timestamps are treated similarly.
    - Strings are parsed with `fmt` if given, otherwise a few different common formats are tried.

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
        if isinstance(other, pendulum.Interval):
            return other.start in self._window and other.end in self._window
        return other in self._window


class xrwrap:
    def __init__(self, data):
        if data is None:
            return
        if isinstance(data, xrwrap):
            data = data.data
        self._data = data

    @property
    def data(self):
        return self._data

    def _transfer_attributes(self, other):
        """Copy attributes form self to other

        This is useful to when creating a new copy of the same instance
        but with new data. The intent is for subclasses to extend
        this function to preserve attributes of the class that
        are not stored within the data variable.
        Note that this does not create a new instance of the class,
        so `other` should already be instantiated with data.
        The typical scheme to create a new instance from a new datastructure
        is
        ```
        new = type(self)(data)
        self._transfer_attributes(new)
        return new
        ```
        """
        pass

    @classmethod
    def _select_wrapper(cls, data):
        """Select an appropriate wrapper for xr.DataArray

        This classmethod inspects the DataArray and returns
        a wrapper class for it. The base implementation is
        to return the class on which this method was called.
        Subclasses can extend this to choose another wrapper
        as appropriate.
        """
        return cls

    def __array_wrap__(self, data, context=None, transfer_attributes=True):
        cls = self._select_wrapper(data)
        if cls is None:
            return data
        new = cls(data)
        if transfer_attributes:
            self._transfer_attributes(new)
        return new

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, drop_allnan=True, **indexers_kwargs):
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
        return self.data.coords

    @property
    def dims(self):
        return self.data.dims


class DataArrayWrap(xrwrap, np.lib.mixins.NDArrayOperatorsMixin):
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
        return self.data.__array__(dtype=dtype)

    @staticmethod
    def _implements_np_func(np_func):
        def decorator(func):
            func._implements_np_func = np_func
            return func
        return decorator

    def __init_subclass__(cls) -> None:
        implementations = {}
        for name, value in cls.__dict__.items():
            if callable(value) and hasattr(value, "_implements_np_func"):
                implementations[value._implements_np_func] = value
        cls._np_func_implementations = implementations

    def __array_function__(self, func, types, args, kwargs):
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
                out = self.data.__array_wrap__(out)
            out = self.__array_wrap__(out)
            return out
        return func(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = (arg.data if isinstance(arg, DataArrayWrap) else arg for arg in inputs)
        return self.__array_wrap__(self.data.__array_ufunc__(ufunc, method, *inputs, **kwargs))

    @_implements_np_func(np.mean)
    def mean(self, dim=..., **kwargs):
        return self.__array_wrap__(self.data.mean(dim, **kwargs))

    @_implements_np_func(np.sum)
    def sum(self, dim=..., **kwargs):
        return self.__array_wrap__(self.data.sum(dim, **kwargs))

    @_implements_np_func(np.std)
    def std(self, dim=..., **kwargs):
        return self.__array_wrap__(self.data.std(dim, **kwargs))

    @_implements_np_func(np.max)
    def max(self, dim=..., **kwargs):
        return self.__array_wrap__(self.data.max(dim, **kwargs))

    @_implements_np_func(np.min)
    def min(self, dim=..., **kwargs):
        return self.__array_wrap__(self.data.min(dim, **kwargs))

    def apply(self, func, *args, **kwargs):
        data = func(self.data, *args, **kwargs)
        return self.__array_wrap__(data)

    def reduce(self, func, dim, **kwargs):
        data = self.data.reduce(func=func, dim=dim, **kwargs)
        return self.__array_wrap__(data)

DataArrayWrap.__init_subclass__()


class DatasetWrap(xrwrap, collections.abc.MutableMapping):
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
        return self.data.time

    @property
    def samplerate(self):
        return self.data.time.rate

    @property
    def time_window(self):
        # Calculating duration from number and rate means the stop points to the sample after the last,
        # which is more intuitive when considering signal durations etc.
        return TimeWindow(
            start=self.data.time.data[0],
            duration=self.data.sizes['time'] / self.samplerate,
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        original_window = self.time_window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        if isinstance(new_window, TimeWindow):
            start = (new_window.start - original_window.start).total_seconds()
            stop = (new_window.stop - original_window.start).total_seconds()
            # Indices assumed to be seconds from start
            start = np.math.floor(start * self.samplerate)
            stop = np.math.ceil(stop * self.samplerate)
            idx = slice(start, stop)
        else:
            idx = (new_window - original_window.start).total_seconds()
            idx = round(idx * self.samplerate)

        selected_data = self.data.isel(time=idx)
        new = type(self)(selected_data)
        self._transfer_attributes(new)
        return new

    def listen(self, downsampling=1, upsampling=None, headroom=6, **kwargs):
        import sounddevice as sd
        sd.stop()
        data = self.data
        if upsampling:
            data = data[::upsampling]
        scaled = data - data.mean()
        scaled = scaled / np.max(np.abs(scaled)) * 10 ** (-headroom / 20)
        sd.play(scaled, samplerate=round(self.samplerate / downsampling), **kwargs)


class FrequencyData(DataArrayWrap):
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
        return self.data.frequency

    def estimate_bandwidth(self):
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
    def __init__(self, recording, track):
        start = max(recording.time_window.start, track.time_window.start)
        stop = min(recording.time_window.stop, track.time_window.stop)

        self.recording = recording.subwindow(start=start, stop=stop)
        self.track = track.subwindow(start=start, stop=stop)

    @property
    def time_window(self):
        rec_window = self.recording.time_window
        track_window = self.track.time_window
        return TimeWindow(start=max(rec_window.start, track_window.start), stop=min(rec_window.stop, track_window.stop))

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        subwindow = self.time_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        rec = self.recording.subwindow(subwindow)
        track = self.track.subwindow(subwindow)
        return type(self)(recording=rec, track=track)


def dB(x, power=True, safe_zeros=True, ref=1):
    '''Calculate the decibel of an input value

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
    '''
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
