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
    time_frame_settings
    TimeWindow
    xrwrap
    DataArrayWrap
    DatasetWrap
    Roller
    TimeDataRoller

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
import whenever

__all__ = [
    "TimeWindow",
    "dB",
    "TimeData",
    "FrequencyData",
    "TimeFrequencyData",
    "Transit",
]


def time_to_np(input, **kwargs):
    """Convert a time to `numpy.datetime64`."""
    if isinstance(input, np.datetime64):
        return input
    if not isinstance(input, whenever.Instant):
        input = time_to_datetime(input, **kwargs)
    return np.datetime64(input.timestamp_nanos(), "ns")


def time_to_datetime(input, fmt="RFC 3339", tz="UTC"):
    """Convert datetimes to the same internal format.

    This function takes a few types of input and tries to convert
    the input to a `whenever.Instant`.
    - Any datetime-like input will be converted directly.
    - np.datetime64 and Unix timestamps are treated similarly.
    - Strings are parsed with ``fmt`` if given, otherwise a few different common formats are tried.

    Parameters
    ----------
    input : datetime-like, string, or numeric.
        The input data specifying the time.
    fmt : string, optional
        This can be one of the standard formats ``"RFC 3339"``, ``"RFC 2822"``, or ``"ISO 8601"``, handled by `whenever <https://whenever.readthedocs.io/en/latest/overview.html#formatting-and-parsing>`_.
        Alternatively, a custom parsing specifier can be supplied, using `strptime format codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
    tz : string, default "UTC"
        The assumed timezone for the input, if it does not contain one.
        Unix timestamps have no timezone, and np.datetime64 only supports UTC.

    Returns
    -------
    time : whenever.Instant
        The converted time.
    """
    if isinstance(input, str):
        if input == "now":
            return whenever.Instant.now()
        if fmt == "RFC 3339":
            return whenever.OffsetDateTime.parse_rfc3339(input).instant()
        if fmt == "RFC 2822":
            return whenever.OffsetDateTime.parse_rfc2822(input).instant()
        if "ISO" in fmt:
            return whenever.OffsetDateTime.parse_common_iso(input).instant()
        if "z" in fmt:
            return whenever.OffsetDateTime.strptime(input, fmt).instant()
        if tz == "UTC":
            return whenever.LocalDateTime.strptime(input, fmt).assume_utc()
        if "/" in tz:
            return whenever.LocalDateTime.strptime(input, fmt).assume_tz(tz).instant()
        if ":" in tz:
            if tz[0] == "+":
                sign = 1
                tz = tz[1:]
            elif tz[0] == "-":
                sign = -1
                tz = tz[1:]
            else:
                sign = 1
            hours, minutes = tz.split(":")
            tz = sign * whenever.TimeDelta(hours=int(hours), minutes=int(minutes))
            return whenever.LocalDateTime.strptime(input, fmt).assume_fixed_offset(tz).instant()

    if isinstance(input, whenever.Instant):
        return input
    if isinstance(input, (whenever.OffsetDateTime, whenever.ZonedDateTime)):
        return input.instant()
    if isinstance(input, whenever.LocalDateTime):
        return input.assume_utc()

    if hasattr(input, "timestamp"):
        if callable(input.timestamp):
            input = input.timestamp()
        else:
            input = input.timestamp
    if isinstance(input, xr.DataArray):
        input = input.data
    if isinstance(input, np.datetime64) or (
        isinstance(input, np.ndarray) and np.issubdtype(input.dtype, np.datetime64)
    ):
        if tz != "UTC":
            raise ValueError("Numpy datetime64 values should always be stored in UTC")
        input = float(input.astype("timedelta64") / np.timedelta64(1, "s"))
    return whenever.Instant.from_timestamp(input)


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
            _start = center.subtract(seconds=duration / 2)
            _stop = center.add(seconds=duration / 2)
            center = duration = None
        elif None not in (start, duration):
            _start = start
            _stop = start.add(seconds=duration)
            start = duration = None
        elif None not in (stop, duration):
            _stop = stop
            _start = stop.subtract(seconds=duration)
            stop = duration = None
        elif None not in (start, center):
            _start = start
            _stop = start + (center - start) * 2
            start = center = None
        elif None not in (stop, center):
            _stop = stop
            _start = stop - (stop - center) * 2
            stop = center = None
        else:
            raise TypeError("Needs two of the input arguments to determine time window.")

        if (start, stop, center, duration) != (None, None, None, None):
            raise TypeError("Cannot input more than two input arguments to a time window!")

        if extend is not None:
            _start = _start.subtract(seconds=extend)
            _stop = _stop.add(seconds=extend)

        self._start = _start
        self._stop = _stop

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
        This can be one of `TimeWindow`, and `xarray.Dataset`.
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
                    raise TypeError("Cannot create subwindow from arguments")
            elif duration is not None and True in (start, stop, center):
                if start is True:
                    window = type(self)(start=self.start, duration=duration, extend=extend)
                elif stop is True:
                    window = type(self)(stop=self.stop, duration=duration, extend=extend)
                elif center is True:
                    window = type(self)(center=self.center, duration=duration, extend=extend)
                else:
                    raise TypeError("Cannot create subwindow from arguments")
            else:
                # The same types explicit arguments as the normal constructor
                window = type(self)(start=start, stop=stop, center=center, duration=duration, extend=extend)
        elif isinstance(time, type(self)):
            window = time
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
        return f"TimeWindow(start={self.start.format_rfc3339()}, stop={self.stop.format_rfc3339()})"

    @property
    def start(self):
        """The start of this window."""
        return self._start

    @property
    def stop(self):
        """The stop of this window."""
        return self._stop

    @property
    def center(self):
        """The center of this window."""
        return self.start.add(seconds=self.duration / 2)

    @property
    def duration(self):
        """The duration of this window, in seconds."""
        return (self.stop - self.start).in_seconds()

    def __contains__(self, other):
        if isinstance(other, type(self)):
            return other.start in self and other.stop in self
        return self.start <= time_to_datetime(other) <= self.stop


def compute_class(data_class):
    """Enable a computation class to delay evaluation.

    This decorator should be used on classes that are callable with
    some data to compute some result. The compute class (the one decorated)
    should be callable and return instances of the data class.
    The compute class should be instantiable without the ``data`` argument,
    creating a processing instance used for later evaluation in the ``__call__``
    method.
    The compute class should also have a ``_should_process(data, *args, **kwargs)`` classmethod,
    determining if the data should be processed or not.This method is only called when
    data was supplied when instantiating.
    If ``_should_process`` returns False, the data is used to create an instance of the
    data class.
    If ``_should_process`` returns True, the other arguments are used to make a processing instance,
    which is called with the data.

    The documentation (and autocomplete) of the class initialization will be taken from
    the decorated class. This means that the decorated class init method should have
    arguments that (if possible) are compatible with both initializers.

    Examples
    --------
    class Data:
        def __init__(self, data):
            self.data = data

        def __repr__(self):
            return f"{self.__class__.__name__}({self.data})"

    @data_processor_class(Data)
    class Processor:
        @classmethod
        def _should_process(cls, data, parameter=None):
            return isinstance(data, str)

        def __init__(self, *, parameter):
            self.parameter = parameter

        def __call__(self, data):
            return Data(data + self.parameter)

        def __repr__(self):
            return f"{self.__class__.__name__}({self.parameter})"

    print(Processor(parameter="param"))
    >>> Processor(param)
    print(Processor("foo", parameter="param"))
    >>> Data(fooparam)
    """
    import functools
    import inspect

    def wrapper(compute_class):
        @functools.wraps(compute_class, updated=())
        class Wrapped(compute_class, data_class):
            def __new__(cls, data=None, *args, **kwargs):
                if data is None:
                    return compute_class(*args, **kwargs)
                if cls._should_process(data, *args, **kwargs):
                    processor = compute_class(*args, **kwargs)
                    return processor(data)
                return data_class(data, *args, **kwargs)

        Wrapped.__signature__ = inspect.signature(compute_class)
        return Wrapped

    return wrapper


class xrwrap:
    """Wrapper around `xarray` objects.

    This base class exists to delegate work to our internal
    `xarray` objects.
    """

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """Instantiate the class from a dataset.

        This classmethod is mainly used to choose the correct class
        to instantiate, depending on the data.
        """
        return cls(dataset, **kwargs)

    def __init__(self, data, attrs=None):
        if isinstance(data, xrwrap):
            data = data.data
        self._data = data
        if attrs:
            self._data.attrs.update(attrs)

    @property
    def attrs(self):
        """Attributes stored in the data."""
        return self.data.attrs

    @property
    def data(self):
        """The contained data."""
        return self._data

    @classmethod
    def _select_wrapper(cls, data):
        """Select an appropriate wrapper for `xarray.DataArray`.

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
            new.attrs.update(self.attrs)
        return new

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, drop_allnan=True, **indexers_kwargs):
        """Select a subset of the data from the coordinate labels.

        The selection is easiest done with keywords, e.g. ``obj.sel(sensor="Colmar 1")``
        to select a specific sensor. For numerical coordinates, ``method="nearest"`` can
        be quite useful. Use a slice to select a range of values, e.g.,
        ``obj.sel(frequency=slice(10, 100))``.

        For more details, see `xarray.DataArray.sel` and `xarray.Dataset.sel`.
        """
        new = self.data.sel(indexers=indexers, method=method, tolerance=tolerance, drop=drop, **indexers_kwargs)
        if drop_allnan:
            new = new.where(~new.isnull(), drop=True)
        return self.__array_wrap__(new)

    def isel(self, indexers=None, drop=False, missing_dims="raise", drop_allnan=True, **indexers_kwargs):
        """Select a subset of the data from the coordinate indices.

        The selection is easiest done with keywords, e.g. ``obj.sel(sensor=0)``
        to select the zeroth sensor. Use a slice to select a range of values, e.g.,
        ``obj.sel(frequency=slice(10, 100))``.

        For more details, see `xarray.DataArray.isel` and `xarray.Dataset.isel`.
        """
        new = self.data.isel(indexers=indexers, drop=drop, missing_dims=missing_dims, **indexers_kwargs)
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

    def _figure_template(self, **kwargs):
        """Create default figure layout for this data."""
        import plotly.graph_objects as go
        import plotly.io as pio

        template = go.layout.Template()
        template.update(pio.templates[pio.templates.default])
        return template

    def make_figure(self, **kwargs):
        """Create a plotly figure, styled for this data.

        Some useful keyword arguments:

        - ``xaxis_title`` and ``yaxis_title`` controls axis titles for the figure.
        - ``height`` and ``width`` sets the figure size in pixels.
        - ``title`` adds a top level title.
        """
        import plotly.graph_objects as go

        fig = go.Figure(layout=dict(template=self._figure_template(**kwargs), **kwargs))
        return fig

    def save(self, path, append_dim=None, **kwargs):
        """Save the data to a Zarr file at the specified path.

        The class name is stored as an attribute, so it can be used later to reconstruct the object.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path where the data will be saved. The directory will be
            created if it doesn't exist.
        append_dim : str, optional
            A dimension which should be used to append to existing data in the path.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `xarray.Dataset.to_zarr` method, which
            is responsible for saving the data to the Zarr format.
        """
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        module = self.__class__.__module__
        name = self.__class__.__qualname__
        if module is not None and module != "__builtin__":
            name = module + "." + name

        data = self.data.assign_attrs(__uwacan_class__=name)

        if append_dim:
            if append_dim not in data.dims:
                data = data.expand_dims(append_dim)
            if not path.exists():
                # Cannot append if there is no data to append to, so we just write as normal
                append_dim = None
        if isinstance(data, xr.Dataset):
            for var in data.dara_vars.values():
                if np.issubdtype(var.dtype, np.datetime64):
                    var.encoding = {"units": "nanoseconds since 1970-01-01"}
        for coord in data.coords.values():
            if np.issubdtype(coord.dtype, np.datetime64):
                coord.encoding = {"units": "nanoseconds since 1970-01-01"}

        data.to_zarr(path, append_dim=append_dim, **kwargs)

    @classmethod
    def load(cls, path, lookup_class=True):
        """Load data from a Zarr file and optionally restore the original class.

        This method loads data from a Zarr file and attempts to reconstruct the
        original class that was used to save the data. The class information is
        stored in the `__uwacan_class__` attribute of the dataset. If the class
        is found, the method dynamically loads and instantiates it.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path from which to load the data.
        lookup_class : bool, default=True
            If True attempts to restore the original class from the
            metadata stored in the Zarr file. If False, the called class is used
            to load the data.

        Returns
        -------
        obj : cls or ``__uwacan_class__``
            An instance of the class used to save the data (if found in the Zarr
            file's metadata), or an instance of the called class.

        Notes
        -----
        - The Zarr file should have the `__uwacan_class__` attribute in the
          dataset's metadata to allow class reconstruction.
        - If the class cannot be found, or if there is an error during dynamic
          import, the method falls back to using the current class `cls` to
          instantiate the object.

        """
        data = xr.open_zarr(path)
        if "__xarray_dataarray_variable__" in data:
            data = data["__xarray_dataarray_variable__"]
            data.name = None
        if lookup_class and "__uwacan_class__" in data.attrs:
            import importlib

            module_name, class_name = data.attrs["__uwacan_class__"].rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except:
                pass
        return cls(data)


class DataArrayWrap(xrwrap, np.lib.mixins.NDArrayOperatorsMixin):
    """Wrapper around `xarray.DataArray`.

    This base class exists to wrap functionality in `xarray.DataArray`,
    and numerical operations from `numpy`.
    """

    _coords_set_by_init = set()

    def __init__(self, data, dims=(), coords=None, **kwargs):
        if isinstance(data, DataArrayWrap):
            data = data.data
        if not isinstance(data, xr.DataArray):
            if isinstance(dims, str):
                dims = [dims]
            if dims is None:
                dims = ()
            if np.ndim(data) != np.size(dims):
                raise ValueError(
                    f"Dimension names '{dims}' for {type(self).__name__} does not match data with {np.ndim(data)} dimensions"
                )
            data = xr.DataArray(data, dims=dims)
        if coords is not None:
            data = data.assign_coords(
                **{name: coord for (name, coord) in coords.items() if name not in self._coords_set_by_init}
            )
        super().__init__(data, **kwargs)

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


class DatasetWrap(xrwrap):
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

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def get(self, key, default=None):
        return self.data.get(key, default=default)

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._data.variables))


class Roller:
    """Base class for rolling windows."""

    @property
    def num_frames(self):
        """The number of frames in this rolling output."""
        return self.settings["num_frames"]

    @property
    def shape(self):
        """The shape of the output frames."""
        ...

    @property
    def dims(self):
        """The dimensions of the output frames."""
        ...

    @property
    def coords(self):
        """The coords of the output frames."""
        ...

    def numpy_frames(self):
        """Generate numpy frames with data."""
        yield

    def __iter__(self):
        """Generate rolling frames of the contained data, of the same type."""
        yield


class TimeData(DataArrayWrap):
    """Handing data which varies over time.

    This class is mainly used to wrap time-signals of sampled sounds.
    As such, the time data is assumed to be sampled at a constant samplerate.

    Parameters
    ----------
    data : array_like
        A `numpy.ndarray` or a `xarray.DataArray` with the time data.
    time : array_like, optional
        A `numpy.ndarray` with ``dtype=datetime64[ns]`` containing time stamps for the samples.
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
    attrs : dict, optional
        Additional attributes to store with this data.
    """

    _coords_set_by_init = {"time"}

    def __init__(
        self, data, time=None, start_time=None, samplerate=None, dims="time", coords=None, attrs=None, **kwargs
    ):
        super().__init__(data, dims=dims, coords=coords, attrs=attrs, **kwargs)

        if samplerate is None and time is not None:
            samplerate = np.timedelta64(1, "s") / np.mean(np.diff(time[:1000]))
        elif time is None and samplerate is not None:
            if start_time is None:
                if "time" in self.data.coords:
                    start_time = self.data.time[0].item()
                else:
                    start_time = "now"
            n_samples = self.data.sizes["time"]
            start_time = time_to_np(start_time)
            offsets = np.arange(n_samples) * 1e9 / samplerate
            time = start_time + offsets.astype("timedelta64[ns]")

        if time is not None:
            if not isinstance(time, xr.DataArray):
                time = xr.DataArray(time, dims="time")
            if samplerate is not None:
                time.attrs["rate"] = samplerate
            self.data.coords["time"] = time

    @classmethod
    def _select_wrapper(cls, data):
        if "time" in data.coords:
            return super()._select_wrapper(data)
        # This is not time data any more, just return the plain xr.DataArray
        return None

    @property
    def samplerate(self):
        return self.data.time.rate

    @property
    def time(self):
        """Time coordinates for this data."""
        return self.data.time

    @property
    def time_window(self):
        """A `TimeWindow` describing when the data start and stops."""
        # Calculating duration from number and rate means the stop points to the sample after the last,
        # which is more intuitive when considering signal durations etc.
        return TimeWindow(
            start=self.data.time.data[0],
            duration=self.data.sizes["time"] / self.samplerate,
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        """Select a subset of the data over time.

        See `TimeWindow.subwindow` for details on the parameters.
        """
        original_window = self.time_window
        new_window = original_window.subwindow(
            time, start=start, stop=stop, center=center, duration=duration, extend=extend
        )
        if isinstance(new_window, TimeWindow):
            start = (new_window.start - original_window.start).in_seconds()
            stop = (new_window.stop - original_window.start).in_seconds()
            # Indices assumed to be seconds from start
            start = int(np.floor(start * self.samplerate))
            stop = int(np.ceil(stop * self.samplerate))
            idx = slice(start, stop)
        else:
            idx = (new_window - original_window.start).in_seconds()
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

    def rolling(self, duration=None, step=None, overlap=None):
        """Generate rolling windows of this data.

        Parameters
        ----------
        duration : float
            The duration of each frame, in seconds.
        step : float
            The step between consecutive frames, in seconds.
        overlap : float
            The overlap between consecutive frames, as a fraction of the duration.

        Returns
        -------
        roller : `TimeDataRoller`
            A roller object to roll over the data.
        """
        return TimeDataRoller(self, duration=duration, step=step, overlap=overlap)


class TimeDataRoller(Roller):
    """Rolling windows of time data.

    Parameters
    ----------
    obj : TimeData
        The time data wrapper to roll over.
    duration : float
        The duration of each frame, in seconds.
    step : float
        The step between consecutive frames, in seconds.
    overlap : float
        The overlap between consecutive frames, as a fraction of the duration.
    """

    def __init__(self, obj, duration=None, step=None, overlap=0):
        self.obj = obj
        self.settings = time_frame_settings(
            duration=duration,
            step=step,
            overlap=overlap,
            signal_length=self.obj.time_window.duration,
            samplerate=self.obj.samplerate,
        )

        self._slices = (
            slice(start_idx, start_idx + self.settings["samples_per_frame"])
            for start_idx in range(
                0, self.settings["num_frames"] * self.settings["sample_step"], self.settings["sample_step"]
            )
        )

    @property
    def coords(self):
        coords = dict(self.obj.coords)
        coords["time"] = coords["time"][: self.settings["samples_per_frame"]]
        return coords

    @property
    def dims(self):
        dims = list(self.obj.dims)
        dims.remove("time")
        return tuple(["time"] + dims)

    @property
    def shape(self):
        shape = list(self.obj.data.shape)
        del shape[self.obj.dims.index("time")]
        shape = [self.settings["samples_per_frame"]] + shape
        return tuple(shape)

    def numpy_frames(self):
        np_data = self.obj.data.transpose("time", ...).data
        for s in self._slices:
            yield np_data[s]

    def time_data(self):
        """Generate rolling frames of the contained data, as `TimeData`."""
        for s in self._slices:
            yield self.obj.isel(time=s)

    def __iter__(self):
        yield from self.time_data()


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
    attrs : dict, optional
        Additional attributes to store with this data.
    """

    _coords_set_by_init = {"frequency", "bandwidth"}

    def __init__(self, data, frequency=None, bandwidth=None, dims="frequency", coords=None, attrs=None, **kwargs):
        super().__init__(data, dims=dims, coords=coords, attrs=attrs, **kwargs)
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
        """The frequencies for the data."""
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
            central = (frequency[2:] ** 0.5 - frequency[:-2] ** 0.5) * frequency[1:-1] ** 0.5
            # extrapolating to one bin below lowest and one above highest using constant ratio
            # the expression above then simplifies to the expressions below
            first = (frequency[1] - frequency[0]) * (frequency[0] / frequency[1]) ** 0.5
            last = (frequency[-1] - frequency[-2]) * (frequency[-1] / frequency[-2]) ** 0.5
        bandwidth = np.concatenate([[first], central, [last]])
        return xr.DataArray(bandwidth, coords={"frequency": self.frequency})

    def _figure_template(self, **kwargs):
        template = super()._figure_template(**kwargs)
        template.layout.update(
            xaxis=dict(
                type="log",
                title="Frequency in Hz",
            )
        )
        return template

    def plot(self, **kwargs):
        """Make a scatter trace of this data.

        Some useful keyword arguments:

        - ``name`` sets the legendentry of this trace.
        - ``line_color`` and ``marker_color`` sets the line and marker colors.
        - ``mode`` sets the mode, typically one of ``"lines"``, ``"markers"``, or ``"markers+lines"``.

        """
        import plotly.graph_objects as go

        if self.dims != ("frequency",):
            raise ValueError(
                f"Cannot simply scatter frequency data with dimensions '{self.dims}'. "
                "Use the `.groupby(dim)` method to loop over non-frequency dimensions."
            )

        return go.Scatter(
            x=self.frequency,
            y=self.data,
            **kwargs,
        )


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
    attrs : dict, optional
        Additional attributes to store with this data.
    """

    _coords_set_by_init = {"time", "frequency", "bandwidth"}

    def __init__(
        self,
        data,
        start_time=None,
        samplerate=None,
        frequency=None,
        bandwidth=None,
        dims=None,
        coords=None,
        attrs=None,
        **kwargs,
    ):
        super().__init__(
            data,
            dims=dims,
            coords=coords,
            attrs=attrs,
            start_time=start_time,
            samplerate=samplerate,
            frequency=frequency,
            bandwidth=bandwidth,
            **kwargs,
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

    def rolling(self, duration=None, step=None, overlap=None):
        """Generate rolling windows of this data.

        By default, a single time instance of the data will be
        generated in each frame.

        Parameters
        ----------
        duration : float
            The duration of each frame, in seconds.
        step : float
            The step between consecutive frames, in seconds.
        overlap : float
            The overlap between consecutive frames, as a fraction of the duration.

        Returns
        -------
        roller : `TimeDataRoller`
            A roller object to roll over the data.
        """
        if (duration, step) == (None, None):
            step = 1 / self.samplerate
            overlap = 0
        return TimeDataRoller(self, duration=duration, step=step, overlap=overlap)

    def _figure_template(self, **kwargs):
        template = super()._figure_template(**kwargs)
        template.layout.update(
            xaxis=dict(
                title_text="Time",
                type=None,  # Use default which is `linear`` for "normal" data and `date` for timestamp data
            ),
            yaxis=dict(
                title_text="Frequency in Hz",
                type="log",
            ),
        )
        template.data.update(
            heatmap=[
                dict(
                    colorscale="viridis",
                    colorbar_title_side="right",
                    hovertemplate="%{x}<br>%{y:.5s}Hz<br>%{z}<extra></extra>",
                )
            ]
        )
        return template

    def plot(self, **kwargs):
        """Make a heatmap trace of this data.

        Some useful keyword arguments:

        - ``name`` sets the legendentry of this trace.
        - ``colorscale`` chooses the colorscale, e.g., ``"viridis"``, ``"delta"``, ``"twilight"``.
        - ``zmin`` and ``zmax`` sets the color range.

        """
        import plotly.graph_objects as go

        if set(self.dims) != {"frequency", "time"}:
            raise ValueError(
                f"Cannot make heatmap of  time-frequency data with dimensions '{self.dims}'. "
                "Use the `.groupby(dim)` method to loop over extra dimensions."
            )

        trace = go.Heatmap(
            x=self.time,
            y=self.frequency,
            z=self.data.transpose("frequency", "time"),
        )
        return trace.update(**kwargs)


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
        subwindow = self.time_window.subwindow(
            time, start=start, stop=stop, center=center, duration=duration, extend=extend
        )
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
        min_value = np.nanmin(abs(xr.where(nonzero, x, np.nan, keep_attrs=True)))
        x = xr.where(nonzero, x, min_value, keep_attrs=True)
    if power:
        if np.any(x < 0):
            if power == "imag":
                return 10 * np.log10(x + 0j)
            if power == "nan":
                return 10 * np.log10(xr.where(x > 0, x, np.nan, keep_attrs=True))
        return 10 * np.log10(x / ref)
    else:
        return 20 * np.log10(np.abs(x) / ref)


def time_frame_settings(
    duration=None,
    step=None,
    overlap=None,
    resolution=None,
    num_frames=None,
    signal_length=None,
    samplerate=None,
):
    """Calculate time frame overlap settings from various input parameters.

    Parameters
    ----------
    duration : float
        How long each frame is, in seconds
    step : float
        The time between frame starts, in seconds
    overlap : float
        How much overlap there is between the frames, as a fraction of the duration.
        If this is negative the frames will have extra space between them.
    resolution : float
        Desired frequency resolution in Hz. Equals ``1/duration``
    num_frames : int
        The total number of frames in the signal
    signal_length : float, optional
        The total length of the signal, in seconds
    samplerate : float, optional
        The samplerate of the signal, in Hz.
        Only used to compute sample versions of the outputs.

    Returns
    -------
    settings : dict
        The settings dict has keys:
            ``"duration"``, ``"step"``, ``"overlap"``, ``"resolution"``
        and if ``signal_length`` was given:
            ``"num_frames"``, ``"signal_length"``
        and if ``samplerate`` was given:
            ``"samples_per_frame"``, ``"sample_step"``, ``"sample_overlap"``
        and if both ``samplerate`` and ``signal_length`` was given:
            ``"sample_total"``

    Raises
    ------
    ValueError
        If the inputs are not sufficient to determine the frame,
        or if priorities cannot be disambiguated.

    Notes
    -----
    The parameters will be used in the following priority:

    1. ``signal_length``, ``num_frames``
    2. ``step``, ``duration``
    3. ``resolution`` (only if duration not given)
    4. ``overlap`` (default 0)

    Each frame ``idx=[0, ..., num_frames - 1]`` has::

        start = idx * step
        stop = idx * step + duration

    The last frame thus ends at ``(num_frames - 1) step + duration``.
    The overlap relations are::

        duration = step / (1 - overlap)
        step = duration (1 - overlap)
        overlap = 1 - step / duration

    This gives us the following total list of priorities:

    1) ``signal_length`` and ``num_frames`` are given
        a) ``step`` or ``duration`` (not both!) given
        b) ``resolution`` is given
        c) ``overlap`` is given
    2) ``step`` and ``duration`` given
    3) ``step`` given
        a) ``resolution`` is given
        b) ``overlap`` is given
    4) ``duration`` given (``resolution`` is ignored, ``overlap`` is used)
    5) ``resolution`` given

    For cases 2-5, ``num_frames`` is calculated if ``signal_length`` is given,
    and a new truncated ``signal_length`` is returned.

    If a samplerate was given, the number of samples in an output is also computed.
    This is done with::

        samples_per_frame = ceil(duration * samplerate)
        sample_step = floor(step * samplerate)
        sample_overlap = samples_per_frame - sample_step

    We use ceil for the duration since it is often chosen to obtain a minimum frequency resolution.
    We use floor for the step to make sure we fit all the frames in the total length.
    Note that this means that the sample overlap might not equal
    ``round(duration * overlap * samplerate)``, due to how the rounding is done.
    It can at most be two samples larger than the rounded value.
    """
    if None not in (num_frames, signal_length):
        if None not in (duration, step):
            raise ValueError("Overdetermined time frames")
        elif step is not None:
            # We have the step, calculate the duration
            duration = signal_length - (num_frames - 1) * step
            overlap = 1 - step / duration
        elif duration is not None:
            # We have the duration, calculate the step
            step = (signal_length - duration) / (num_frames - 1)
            overlap = 1 - step / duration
        elif resolution is not None:
            duration = 1 / resolution
            step = (signal_length - duration) / (num_frames - 1)
            overlap = 1 - step / duration
        else:
            overlap = overlap or 0
            duration = signal_length / (num_frames + overlap - num_frames * overlap)
            step = duration * (1 - overlap)
    elif None not in (step, duration):
        overlap = 1 - step / duration
    elif step is not None:
        if resolution is not None:
            duration = 1 / resolution
            overlap = 1 - step / duration
        else:
            overlap = overlap or 0
            duration = step / (1 - overlap)
    elif duration is not None:
        overlap = overlap or 0
        step = duration * (1 - overlap)
    elif resolution is not None:
        duration = 1 / resolution
        overlap = overlap or 0
        step = duration * (1 - overlap)
    else:
        raise ValueError(
            "Must give at least one of (`step`, `duration`, `resolution`) or the pair of `signal_length` and `num_frames`."
        )

    settings = {
        "duration": duration,
        "step": step,
        "overlap": overlap,
        "resolution": 1 / duration,
    }
    if signal_length is not None:
        num_frames = num_frames or int(np.floor((signal_length - duration) / step + 1))
        settings["num_frames"] = num_frames
        settings["signal_length"] = (num_frames - 1) * step + duration
    if samplerate is not None:
        settings["samples_per_frame"] = int(np.ceil(samplerate * duration))
        settings["sample_step"] = int(np.floor(samplerate * step))
        settings["sample_overlap"] = settings["samples_per_frame"] - settings["sample_step"]
        if "num_frames" in settings:
            settings["sample_total"] = (settings["num_frames"] - 1) * settings["sample_step"] + settings["samples_per_frame"]  # fmt: skip
    return settings


def concatenate(items, dim=None, nan_between_items=False, sort=False, cls=None, **kwargs):
    """Concatenates a list of items along a specified dimension.

    Parameters
    ----------
    items : list of `xarray.Dataset` or `xarray.DataArray` or `xrwrap`
        A list of items to be concatenated. The items can be `xr.Dataset`,
        `xr.DataArray`, or instances of a subclass of `xrwrap`.
        Most `uwacan` data wrappers are compatible.
    dim : str or None, optional
        The dimension along which to concatenate. If `None`, the function will
        attempt to infer the dimension from the first item. If the items have
        more than one dimension, a `ValueError` will be raised.
        The concatenation dimension can be an existing dim or a new dim.
    nan_between_items : bool, optional, default=False
        If `True`, inserts NaN values between concatenated items along the specified dimension.
        This is useful for visualization purposes, as it makes most plotting libraries split the lines.
    sort : bool, optional, default=False
        If `True`, sorts the concatenated items by the specified dimension.
    cls : class or callable, optional
        The class or callable to use for wrapping the concatenated result. If `None`,
        the function will try to infer the class from the first item. If `cls` is a subclass
        of `xrwrap` and has a `from_dataset` method, it will use that method to wrap
        the result.
    **kwargs : dict
        Additional keyword arguments passed to the class or callable specified by `cls` when
        wrapping the concatenated result.

    Returns
    -------
    object
        The concatenated object, either as an `xr.Dataset`, `xr.DataArray`, or as an
        instance of the specified `cls` (or inferred class).
    """
    if dim is None:
        if len(items[0].dims) != 1:
            raise ValueError("Cannot guess concatenation dimensions for multi-dimensional items.")
        dim = next(iter(items[0].dims))

    if cls is None:
        for item in items:
            if isinstance(item, xrwrap):
                cls = type(item)
                break

    if isinstance(cls, type) and issubclass(cls, xrwrap) and hasattr(cls, "from_dataset"):
        cls = cls.from_dataset

    items = [item.data if isinstance(item, xrwrap) else item for item in items]
    if nan_between_items:
        items = [x for item in items for x in [item, xr.full_like(item.isel({dim: -1}), np.nan).expand_dims(dim)]][:-1]
    items = xr.concat(items, dim=dim)
    if sort:
        items = items.sortby(dim)
    if cls:
        items = cls(items, **kwargs)
    return items
