"""Various analysis protocols and standards for recorded underwater noise from ships."""
import functools
import numpy as np
from . import positional, propagation, _tools
import scipy.signal
import xarray as xr


class _DataWrapper(np.lib.mixins.NDArrayOperatorsMixin):
    _coords_set_by_init = set()

    def __init__(self, data, dims=(), coords=None):
        if data is None:
            return
        if isinstance(data, _DataWrapper):
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

    @property
    def data(self):
        return self._data

    def __array__(self, dtype=None):
        return self.data.__array__(dtype=dtype)

    def __array_wrap__(self, out_arr, context=None):
        cls = self._select_wrapper(out_arr)
        if cls is None:
            # We have decided not to wrap this out_arr.
            return out_arr
        new = cls(out_arr)
        self._transfer_attributes(new)
        return new

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
            args = (arg.data if isinstance(arg, _DataWrapper) else arg for arg in args)
            out = func(*args, **kwargs)
            if not isinstance(out, xr.DataArray):
                out = self.data.__array_wrap__(out)
            out = self.__array_wrap__(out)
            return out
        return func(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = (arg.data if isinstance(arg, _DataWrapper) else arg for arg in inputs)
        return self.__array_wrap__(self.data.__array_ufunc__(ufunc, method, *inputs, **kwargs))

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

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        return self.__array_wrap__(self.data.sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs
        ))

    @property
    def coords(self):
        return self.data.coords


_DataWrapper.__init_subclass__()

class TimeData(_DataWrapper):
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
            start_time = positional.time_to_np(start_time)
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
        return positional.TimeWindow(
            start=self.data.time.data[0],
            duration=self.data.sizes['time'] / self.samplerate,
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        original_window = self.time_window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        if isinstance(new_window, positional.TimeWindow):
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


class FrequencyData(_DataWrapper):
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
        return positional.TimeWindow(start=max(rec_window.start, track_window.start), stop=min(rec_window.stop, track_window.stop))

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        subwindow = self.time_window.subwindow(time, start=start, stop=stop, center=center, duration=duration)
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
    if isinstance(x, _DataWrapper):
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


class Spectrogram(TimeFrequencyData):
    """Calculates spectrograms

    The processing is done in stft frames determined by `frame_duration`, `frame_step`
    `frame_overlap`, and `frequency_resolution`. At least one of "duration", "step",
    or "resolution" has to be given, see `time_frame_settings` for further details.

    Parameters
    ----------
    data : TimeData, optional
        The time data from which to calculate the spectrogram.
        Omit this to create a callable object for later evaluation.
    frame_duration : float
        The duration of each stft frame, in seconds.
    frame_step : float
        The time step between stft frames, in seconds.
    frame_overlap : float, default 0.5
        The overlap factor between stft frames. A negative value leaves
        gaps between frames.
    frequency_resolution : float
        A frequency resolution to aim for. Only used if `frame_duration` is not given.
    fft_window : str, default="hann"
        The shape of the window used for the stft.
    """
    def __init__(
            self,
            data=None,
            frame_duration=None,
            frame_step=None,
            frame_overlap=0.5,
            frequency_resolution=None,
            fft_window="hann",
            **kwargs
        ):
        super().__init__(data, **kwargs)
        try:
            self.frame_settings = time_frame_settings(
                duration=frame_duration,
                step=frame_step,
                resolution=frequency_resolution,
                overlap=frame_overlap,
            ) | {"window": fft_window}
        except ValueError:
            self.frame_settings = None

        if isinstance(data, TimeData):
            # __call__ will create an object that we only use the ._data from
            self._data = self(data).data

    def __call__(self, time_data):
        if isinstance(time_data, type(self)):
            return time_data
        xr_data = time_data.data
        frame_samples = round(self.frame_settings["duration"] * time_data.samplerate)
        overlap_samples = round(self.frame_settings["duration"] * self.frame_settings["overlap"] * time_data.samplerate)
        f, t, Sxx = scipy.signal.spectrogram(
            x=xr_data.data,  # Gets the numpy array
            fs=time_data.samplerate,
            window=self.frame_settings["window"],
            nperseg=frame_samples,
            noverlap=overlap_samples,
            axis=xr_data.dims.index('time'),
        )
        dims = list(xr_data.dims)
        dims[dims.index('time')] = 'frequency'
        dims.append('time')
        new = type(self)(
            data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
            # The array returned from the spectrogram function is the real part of the original stft, reshaped.
            # This means that the array takes twice the memory (the imaginary part is still around),
            # and it's not contiguous which slows down filtering a lot.
            samplerate=time_data.samplerate / (frame_samples - overlap_samples),
            start_time=time_data.time_window.start.add(seconds=t[0]),
            frequency=f,
            bandwidth=time_data.samplerate / frame_samples,
            dims=tuple(dims),
            coords=xr_data.coords,
        )
        self._transfer_attributes(new)
        return new

    def _transfer_attributes(self, obj):
        super()._transfer_attributes(obj)
        obj.frame_settings = self.frame_settings

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            raise AttributeError(
                f"'{type(self.__name__)}' object has no stored data. "
                "This instance is likely a callable processing object, not a data container."
            )


class NthDecadeSpectrogram(TimeFrequencyData):
    """Calculates Nth-decade spectrograms.

    The processing is done in stft frames determined by `frame_duration`, `frame_step`
    `frame_overlap`, and `hybrid_resolution`. At least one of "duration", "step",
    or "resolution" has to be given, see `time_frame_settings` for further details.
    At least one of `lower_bound` and `hybrid_resolution` has to be given.
    Note that the `frame_duration` and `frame_step` can be auto-chosen from the overlap
    and required frequency resolution, either from `hybrid_resolution` or `lower_bound`.

    Parameters
    ----------
    data : TimeData or Spectrogram, optional
        The time data from which to calculate the spectrogram.
        Omit this to create a callable object for later evaluation.
    frame_duration : float
        The duration of each stft frame, in seconds.
    frame_step : float
        The time step between stft frames, in seconds.
    frame_overlap : float, default 0.5
        The overlap factor between stft frames. A negative value leaves
        gaps between frames.
    lower_bound : float
        The lowest frequency to include in the processing.
    upper_bound : float
        The highest frequency to include in the processing.
    hybrid_resolution : float
        A frequency resolution to aim for. Only used if `frame_duration` is not given.
    scaling : str, default="power spectral density"
        The scaling to use for the output.
        - "power spectral density" scales the output as a power spectral density.
        - "power spectrum" scales the output as the total power in each band.
    fft_window : str, default="hann"
        The shape of the window used for the stft.

    Raises
    ------
    ValueError
        If the processing settings are not compatible, e.g.,
        - frequency bands with bandwidth smaller than the frame duration allows,
        - no lower bound and no hybrid resolution.
    """
    def __init__(
        self,
        data=None,
        bands_per_decade=None,
        frame_step=None,
        frame_duration=None,
        frame_overlap=0.5,
        lower_bound=None,
        upper_bound=None,
        hybrid_resolution=None,
        fft_window="hann",
        **kwargs
    ):
        super().__init__(data, **kwargs)
        try:
            # We cannot use any form of `hybrid_resolution in (True, False, None)`
            # since they use `==`` and not `is` and `1 == True` >> True
            if not (
                hybrid_resolution is True
                or hybrid_resolution is False
                or hybrid_resolution is None
            ):
                resolution = hybrid_resolution
            elif lower_bound is not None and bands_per_decade is not None:
                resolution = lower_bound * (10 ** (0.5 / bands_per_decade) - 10 ** (-0.5 / bands_per_decade))
            else:
                resolution = None
            self.frame_settings = time_frame_settings(
                duration=frame_duration,
                step=frame_step,
                resolution=resolution,
                overlap=frame_overlap,
            ) | {"window": fft_window}
        except ValueError:
            self.frame_settings = None

        self.bands_per_decade = bands_per_decade
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.hybrid_resolution = hybrid_resolution

        if isinstance(data, TimeData):
            # __call__ will create an object that we only use the .data from
            self._data = self(data).data

    def __call__(self, data):
        if isinstance(data, type(self)):
            return data
        if not isinstance(data, Spectrogram):
            data = Spectrogram(
                data,
                frame_step=self.frame_settings["step"],
                frame_duration=self.frame_settings["duration"],
                frame_overlap=self.frame_settings["overlap"],
                frequency_resolution=self.frame_settings["resolution"],
            )

        if self.hybrid_resolution:
            if self.hybrid_resolution is True:
                hybrid_resolution = 1 / data.frame_settings["duration"]
            else:
                hybrid_resolution = self.hybrid_resolution
            if hybrid_resolution * data.frame_settings["duration"] < 1:
                raise ValueError(
                    f'Hybrid filterbank with resolution of {hybrid_resolution:.2f} Hz '
                    f'cannot be calculated from temporal windows of {data.frame_settings["duration"]:.2f} s.'
                )
        else:
            hybrid_resolution = False
            lowest_bandwidth = self.lower_bound * (10 ** (0.5 / self.bands_per_decade) - 10 ** (-0.5 / self.bands_per_decade))
            if lowest_bandwidth * data.frame_settings["duration"] < 1:
                raise ValueError(
                    f'{self.bands_per_decade}th-decade filter band at {self.lower_bound:.2f} Hz with bandwidth of {lowest_bandwidth:.2f} Hz '
                    f'cannot be calculated from temporal windows of {data.frame_settings["duration"]:.2f} s.'
                )

        spec_xr = data.data.transpose("frequency", ...)
        # Get frequency centers for the new bands
        bands_per_decade = self.bands_per_decade
        log_band_scaling = 10 ** (0.5 / bands_per_decade)
        lower_bound = self.lower_bound or 0  # Prevent None or False.
        upper_bound = self.upper_bound or spec_xr.frequency.data[-1] / log_band_scaling
        if hybrid_resolution:
            # The frequency at which the logspaced bands cover at least one linspaced band
            minimum_bandwidth_frequency = hybrid_resolution / (log_band_scaling - 1 / log_band_scaling)
            first_log_idx = np.math.ceil(bands_per_decade * np.log10(minimum_bandwidth_frequency / 1e3))
            last_linear_idx = np.math.floor(minimum_bandwidth_frequency / hybrid_resolution)

            # Since the logspaced bands have pre-determined centers, we can't just start them after the linspaced bands.
            # Often, the bands will overlap at the minimum bandwidth frequency, so we look for the first band
            # that does not overlap, i.e., the upper edge of the last linspaced band is below the lower edge of the first
            # logspaced band
            while (last_linear_idx + 0.5) * hybrid_resolution > 1e3 * 10 ** ((first_log_idx - 0.5) / bands_per_decade):
                # Condition is "upper edge of last linear band is higher than lower edge of first logarithmic band"
                last_linear_idx += 1
                first_log_idx += 1

            if last_linear_idx * hybrid_resolution > upper_bound:
                last_linear_idx = np.math.floor(upper_bound / hybrid_resolution)
            first_linear_idx = np.math.ceil(lower_bound / hybrid_resolution)
        else:
            first_linear_idx = last_linear_idx = 0
            first_log_idx = np.round(bands_per_decade * np.log10(lower_bound / 1e3))

        last_log_idx = round(bands_per_decade * np.log10(upper_bound / 1e3))

        lin_centers = np.arange(first_linear_idx, last_linear_idx) * hybrid_resolution
        lin_lowers = lin_centers - 0.5 * hybrid_resolution
        lin_uppers = lin_centers + 0.5 * hybrid_resolution

        log_centers = 1e3 * 10 ** (np.arange(first_log_idx, last_log_idx + 1) / bands_per_decade)
        log_lowers = log_centers / log_band_scaling
        log_uppers = log_centers * log_band_scaling

        centers = np.concatenate([lin_centers, log_centers])
        lowers = np.concatenate([lin_lowers, log_lowers])
        uppers = np.concatenate([lin_uppers, log_uppers])

        # Aggregate input spectrum into bands
        spec_np = spec_xr.data
        banded_data = np.full(centers.shape + spec_np.shape[1:], np.nan)
        spectral_resolution = 1 / data.frame_settings["duration"]
        for idx, (l, u) in enumerate(zip(lowers, uppers)):
            l_idx = int(np.floor(l / spectral_resolution + 0.5))  # (l_idx - 0.5) * Δf = l
            u_idx = int(np.ceil(u / spectral_resolution - 0.5))  # (u_idx + 0.5) * Δf = u
            l_idx = max(l_idx, 0)
            u_idx = min(u_idx, spec_np.shape[0] - 1)

            if l_idx == u_idx:
                # This can only happen if both frequencies l and u are within the same fft bin.
                # Since we don't allow the fft bins to be larger than the output bins, we thus have the exact same band.
                banded_data[idx] = spec_np[l_idx]
            else:
                # weight edge bins by "(whole bin - what is not in the new band) / whole bin"
                # lower fft bin edge l_e = (l_idx - 0.5) * Δf
                # w_l = (Δf - (l - l_e)) / Δf = l_idx + 0.5 - l / Δf
                first_weight = l_idx + 0.5 - l / spectral_resolution
                # upper fft bin edge u_e = (u_idx + 0.5) * Δf
                # w_u = (Δf - (u_e - u)) / Δf = 0.5 - u_idx + u / Δf
                last_weight = u / spectral_resolution - u_idx + 0.5
                # Sum the components fully within the output bin `[l_idx + 1:u_idx]`, and weighted components partially in the band.
                this_band = (
                    spec_np[l_idx + 1 : u_idx].sum(axis=0)
                    + spec_np[l_idx] * first_weight
                    + spec_np[u_idx] * last_weight
                )
                banded_data[idx] = this_band * (spectral_resolution / (u - l))  # Rescale the power density.
        banded = type(self)(
            data=banded_data,
            start_time=spec_xr.time[0],
            samplerate=spec_xr.time.rate,
            frequency=centers,
            bandwidth=uppers - lowers,
            dims=spec_xr.dims,
            coords=spec_xr.coords,
        )
        self._transfer_attributes(banded)
        return banded

    def _transfer_attributes(self, obj):
        super()._transfer_attributes(obj)
        obj.frame_settings = self.frame_settings

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no stored data. "
                "This instance is likely a callable processing object, not a data container."
            ) from None


class ShipLevel:
    """Calculates and stores measured ship levels."""
    @classmethod
    def analyze_transits(
        cls,
        *transits,
        filterbank=None,
        propagation_model=None,
        background_noise=None,
        transit_min_angle=None,
        transit_min_duration=None,
        transit_min_lengh=None,
    ):
        if filterbank is None:
            filterbank = NthDecadeSpectrogram(
                bands_per_decade=10,
                lower_bound=20,
                upper_bound=20_000,
                frame_step=1
            )

        if background_noise is None:
            def background_noise(received_power, **kwargs):
                return received_power

        if propagation_model is None:
            propagation_model = propagation.MlogR(m=20)

        if isinstance(propagation_model, propagation.PropagationModel):
            propagation_model = propagation_model.compensate_propagation

        results = []
        for transit in transits:
            if (transit_min_angle, transit_min_duration, transit_min_lengh) == (None, None, None):
                cpa_time = transit.track.closest_point(transit.recording.sensor)["time"].data
            else:
                segment = transit.track.aspect_segments(
                    reference=transit.recording.sensor,
                    angles=0,
                    segment_min_duration=transit_min_duration,
                    segment_min_angle=transit_min_angle,
                    segment_min_length=transit_min_lengh,
                )
                cpa_time = segment.time.sel(edge="center").data
                transit = transit.subwindow(segment)

            direction = transit.track.average_course('eight')
            time_data = transit.recording.time_data()
            received_power = filterbank(time_data)

            received_power = background_noise(received_power)
            track = transit.track.resample(received_power.time)
            source_power = propagation_model(received_power=received_power, receiver=transit.recording.sensor, source=track)
            transit_time = (received_power.data["time"] - cpa_time) / np.timedelta64(1, "s")
            closest_to_cpa = np.abs(transit_time).argmin("time").item()
            segment = xr.DataArray(np.arange(transit_time.time.size) - closest_to_cpa, coords={"time": received_power.time})
            transit_results = xr.Dataset(
                data_vars=dict(
                    source_power=source_power.data,
                    latitude=track.latitude,
                    longitude=track.longitude,
                    transit_time=transit_time,
                ),
                coords=dict(
                    segment=segment,
                    direction=direction,
                )
            )
            transit_results["received_power"] = received_power.data
            if hasattr(received_power, "snr"):
                transit_results["snr"] = received_power.snr.data
            results.append(transit_results.swap_dims(time="segment").reset_coords("time"))
        results = xr.concat(results, "transit")
        results.coords["transit"] = np.arange(results.sizes["transit"]) + 1
        return cls(results)

    def __init__(self, data):
       self._data = data

    @property
    def source_power(self):
        return FrequencyData(self._data["source_power"])

    @property
    def source_level(self):
        return dB(self.source_power, power=True)

    @property
    def received_power(self):
        return FrequencyData(self._data["received_power"])

    @property
    def received_level(self):
        return dB(self.received_power, power=True)

    def mean(self, dims, **kwargs):
        return type(self)(self._data.mean(dims, **kwargs))

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        new = self._data.sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs
        )
        new = new.where(~new.transit_time.isnull(), drop=True)
        return type(self)(new)

    @property
    def snr(self):
        return FrequencyData(self._data["snr"])

    def meets_snr_threshold(self, threshold):
        snr = self._data["snr"]
        meets_threshold = xr.where(
            snr.isnull(),
            np.nan,
            snr > threshold,
        )
        return FrequencyData(meets_threshold)


def bureau_veritas_source_spectrum(
    passages,
    propagation_model=None,
    background_noise=None,
    filterbank=None,
    aspect_angles=tuple(range(-45, 46, 5)),
    aspect_segment_length=100,
    aspect_segment_angle=None,
    aspect_segment_duration=None,
    passage_time_padding=10,
):
    if filterbank is None:
        filterbank = decidecade_filter(lower_bound=10, upper_bound=50_000, window_duration=1, overlap=0.5)
    if propagation_model is None:
        propagation_model = propagation.MlogR(m=20)

    if isinstance(propagation_model, propagation.PropagationModel):
        propagation_model = propagation_model.compensate_propagation

    if background_noise is None:
        def background_noise(received_power, **kwargs):
            return received_power

    passage_powers = []
    for passage_idx, passage in enumerate(passages):
        cpa = positional.closest_point(passage.recording.sensor, passage.track)
        time_segments = positional.aspect_segments(
            reference=passage.recording.sensor,
            track=passage.track,
            angles=aspect_angles,
            segment_min_length=aspect_segment_length,
            segment_min_angle=aspect_segment_angle,
            segment_min_duration=aspect_segment_duration,
        )
        time_start = positional.time_to_datetime(time_segments.time.min()).subtract(seconds=passage_time_padding)
        time_stop = positional.time_to_datetime(time_segments.time.max()).add(seconds=passage_time_padding)
        time_data = passage.recording.sampling.subwindow(start=time_start, stop=time_stop).time_data()
        received_power = filterbank(time_data)

        segment_powers = []
        for segment_idx, segment in time_segments.groupby('segment'):
            received_segment = received_power.sampling.subwindow(segment).reduce(np.mean, dim='time')
            source = passage.track.sampling.subwindow(segment.time.sel(edge='center'))

            compensated_segment = background_noise(
                received_segment,
                receiver=passage.recording.sensor,
                time=source.time,
            )
            source_segment = propagation_model(
                compensated_segment,
                receiver=passage.recording.sensor,
                source=source,
            )
            source_segment = source_segment.assign_coords(
                segment=segment.segment,
                latitude=source.latitude,
                longitude=source.longitude,
                time=source.time,
            )
            segment_powers.append(source_segment)

        segment_powers = xr.concat(segment_powers, dim='segment')
        passage_powers.append(segment_powers.assign_coords(cpa=cpa.distance))
    source_powers = xr.concat(passage_powers, dim='passage')
    return source_powers


def time_frame_settings(
    duration=None,
    step=None,
    overlap=None,
    resolution=None,
    num_frames=None,
    signal_length=None,
):
    """Calculates time frame overlap settings from various input parameters.

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
        Desired frequency resolution in Hz. Equals `1/duration`
    num_frames : int
        The total number of frames in the signal
    signal_length : float
        The total length of the signal, in seconds

    Returns
    -------
    dict with keys
        - "duration"
        - "step"
        - "overlap"
        - "resolution"
        and if `signal_length` was given
        - "num_frames"
        - "signal_length"

    Raises
    ------
    ValueError
        If the inputs are not sufficient to determine the frame,
        or if priorities cannot be disambiguated.

    Notes
    -----
    The parameters will be used in the following priority:
    1. signal_length, num_frames
    2. step, duration
    3. resolution (only if duration not given)
    4. overlap (default 0)

    Each frame idx=[0, ..., num_frames - 1] has
    - start = idx * step
    - stop = idx * step + duration

    The last frame thus ends at (num_frames - 1) step + duration.
    The overlap relations are:
    - duration = step / (1 - overlap)
    - step = duration (1 - overlap)
    - overlap = 1 - step / duration

    This gives us the following total list of priorities:
    1) `signal_length` and `num_frames` are given
        a) `step` or `duration` (not both!) given
        b) `resolution` is given
        c) `overlap` is given
    2) `step` and `duration` given
    3) `step` given
        a) `resolution` is given
        b) `overlap` is given
    4) `duration` given (`resolution` is ignored, `overlap` is used)
    5) `resolution` given
    For cases 2-5, `num_frames` is calculated if `signal_length` is given,
    and a new truncated `signal_length` is returned.
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
        raise ValueError("Must give at least one of (`step`, `duration`, `resolution`) or the pair of `signal_length` and `num_frames`.")

    settings = {
        "duration": duration,
        "step": step,
        "overlap": overlap,
        "resolution": 1 / duration,
    }
    if signal_length is not None:
        num_frames = num_frames or np.math.floor((signal_length - duration) / step + 1)
        settings["num_frames"] = num_frames
        settings["signal_length"] = (num_frames - 1) * step + duration
    return settings


@_tools.prebind
def fft(time_signal, nfft=None):
    nfft = nfft or time_signal.sizes['time']
    return xr.apply_ufunc(
        np.fft.rfft,
        time_signal,
        input_core_dims=[['time']],
        output_core_dims=[['frequency']],
        kwargs={'n': nfft},
    ).assign_coords(frequency=np.fft.rfftfreq(nfft, 1 / time_signal.time.rate), time=time_signal.time[0]).rename(time='start_time')


@_tools.prebind
def ifft(spectrum, nfft=None):
    if nfft is None:
        is_odd = np.any(spectrum.isel(frequency=-1).data.imag)
        nfft = (spectrum.sizes['frequency'] - 1) * 2 + (1 if is_odd else 0)

    time_data = xr.apply_ufunc(
        np.fft.irfft,
        spectrum,
        input_core_dims=[['frequency']],
        output_core_dims=[['time']],
        kwargs={'n': nfft},
    ).drop_vars('start_time')
    samplerate = spectrum.frequency[1].item() * nfft
    if hasattr(spectrum, 'start_time'):
        return recordings.time_data(time_data, start_time=spectrum.start_time, samplerate=samplerate)
    time_data.coords['time'] = np.arange(nfft) / samplerate
    time_data.coords['time'].attrs['rate'] = samplerate
    return time_data


@_tools.prebind
def spectrogram(time_signal, window_duration=None, window='hann', overlap=0.5, *args, **kwargs):
    fs = time_signal.sampling.rate
    window_samples = round(window_duration * fs)
    overlap_samples = round(window_duration * overlap * fs)
    f, t, Sxx = scipy.signal.spectrogram(
        x=time_signal.data,
        fs=fs,
        window=window,
        nperseg=window_samples,
        noverlap=overlap_samples,
        axis=time_signal.dims.index('time'),
    )
    dims = list(time_signal.dims)
    dims[dims.index('time')] = 'frequency'
    dims.append('time')
    return recordings.time_frequency_data(
        data=Sxx.copy(),  # Using a copy here is a performance improvement in later processing stages.
        # The array returned from the spectrogram function is the real part of the original stft, reshaped.
        # This means that the array takes twice the memory (the imaginary part is still around),
        # and it's not contiguous which slows down filtering a lot.
        samplerate=fs / (window_samples - overlap_samples),
        start_time=time_signal.sampling.window.start.add(seconds=t[0]),
        frequency=f,
        bandwidth=fs / window_samples,
        dims=tuple(dims),
        coords=time_signal.coords,
    )


@_tools.prebind
def nth_decade_filter(
    time_signal,
    bands_per_decade,
    time_step=None,
    window_duration=None,
    overlap=None,
    lower_bound=None,
    upper_bound=None,
    hybrid_resolution=False,
    scaling='density',
):
    if None not in (time_step, window_duration):
        overlap = 1 - time_step / window_duration
    elif time_step is not None:
        if overlap is None:
            if hybrid_resolution:
                # Set overlap to achieve hybrid resolution
                overlap = 1 - time_step * hybrid_resolution
            else:
                overlap = 0.5
        window_duration = time_step / (1 - overlap)
    elif window_duration is not None:
        # Cannot set overlap from hybrid resolution, the window duration is already set.
        if overlap is None:
            overlap = 0.5
        time_step = window_duration * (1 - overlap)
    elif hybrid_resolution:
        window_duration = 1 / hybrid_resolution
        if overlap is None:
            overlap = 0.5
        time_step = window_duration * (1 - overlap)
    else:
        # TODO: We could possibly use the entire time signal?
        raise ValueError('Must give at least one of `time_step` and `window_duration`.')

    if not (lower_bound or hybrid_resolution):
        raise ValueError(
            'Cannot have a log-spaced filterbank without lower frequency bound. Specify either `lower_bound` or `hybrid_resolution`.'
        )

    # We could relax these if we want to interpolate. This needs to be implemented in the calculations below.
    if hybrid_resolution:
        if hybrid_resolution is True:
            hybrid_resolution = 1 / window_duration
        if hybrid_resolution * window_duration < 1:
            raise ValueError(
                f'Hybrid filterbank with resolution of {hybrid_resolution:.2f} Hz '
                f'cannot be calculated from temporal windows of {window_duration:.2f} s.'
            )
    else:
        lowest_bandwidth = lower_bound * (10 ** (0.5 / bands_per_decade) - 10 ** (-0.5 / bands_per_decade))
        if lowest_bandwidth * window_duration < 1:
            raise ValueError(
                f'{bands_per_decade}th-decade filter band at {lower_bound:.2f} Hz with bandwidth of {lowest_bandwidth:.2f} Hz '
                f'cannot be calculated from temporal windows of {window_duration:.2f} s.'
            )

    spec = spectrogram(
        time_signal=time_signal,
        window_duration=window_duration,
        overlap=overlap,
        window=('tukey', 2 * overlap),
    ).transpose('frequency', ...)  # Put the frequency axis first for ease of indexing later

    log_band_scaling = 10 ** (0.5 / bands_per_decade)
    upper_bound = upper_bound or spec.frequency.data[-1] / log_band_scaling
    # Get frequency vectors
    if hybrid_resolution:
        minimum_bandwidth_frequency = hybrid_resolution / (log_band_scaling - 1 / log_band_scaling)
        first_log_idx = np.math.ceil(bands_per_decade * np.log10(minimum_bandwidth_frequency / 1e3))
        last_linear_idx = np.math.floor(minimum_bandwidth_frequency / hybrid_resolution)

        while (last_linear_idx + 0.5) * hybrid_resolution > 1e3 * 10 ** ((first_log_idx - 0.5) / bands_per_decade):
            # Condition is "upper edge of last linear band is higher than lower edge of first logarithmic band"
            last_linear_idx += 1
            first_log_idx += 1

        if last_linear_idx * hybrid_resolution > upper_bound:
            last_linear_idx = np.math.floor(upper_bound / hybrid_resolution)
        if lower_bound is not None:
            first_linear_idx = np.math.ceil(lower_bound / hybrid_resolution)
        else:
            first_linear_idx = 0
    else:
        first_linear_idx = last_linear_idx = 0
        first_log_idx = np.round(bands_per_decade * np.log10(lower_bound / 1e3))

    last_log_idx = round(bands_per_decade * np.log10(upper_bound / 1e3))

    lin_centers = np.arange(first_linear_idx, last_linear_idx) * hybrid_resolution
    lin_lowers = lin_centers - 0.5 * hybrid_resolution
    lin_uppers = lin_centers + 0.5 * hybrid_resolution

    log_centers = 1e3 * 10 ** (np.arange(first_log_idx, last_log_idx + 1) / bands_per_decade)
    log_lowers = log_centers / log_band_scaling
    log_uppers = log_centers * log_band_scaling

    centers = np.concatenate([lin_centers, log_centers])
    lowers = np.concatenate([lin_lowers, log_lowers])
    uppers = np.concatenate([lin_uppers, log_uppers])

    spec_data = spec.data
    banded_data = np.full(centers.shape + spec_data.shape[1:], np.nan)
    spectral_resolution = 1 / window_duration

    for idx, (l, u) in enumerate(zip(lowers, uppers)):
        l_idx = np.math.floor(l / spectral_resolution + 0.5)  # + 0.5 to consider fft bin lower edge
        u_idx = np.math.ceil(u / spectral_resolution - 0.5)  # - 0.5 to consider fft bin upper edge
        l_idx = max(l_idx, 0)
        u_idx = min(u_idx, spec_data.shape[0] - 1)

        if l_idx == u_idx:
            # This can only happen if both frequencies l and u are within the same fft bin.
            # Since we don't allow the fft bins to be larger than the output bins, we thus have the exact same band.
            banded_data[idx] = spec_data[l_idx]
        else:
            first_weight = l_idx + 0.5 - l / spectral_resolution
            last_weight = u / spectral_resolution - u_idx + 0.5
            # Sum the components fully within the output bin `[l_idx + 1:u_idx]`, and weighted components partially in the band.
            this_band = (
                spec_data[l_idx + 1 : u_idx].sum(axis=0)
                + spec_data[l_idx] * first_weight
                + spec_data[u_idx] * last_weight
            )
            banded_data[idx] = this_band * (spectral_resolution / (u - l))  # Rescale the power density.
    banded = recordings.time_frequency_data(
        data=banded_data,
        start_time=spec.sampling.window.start,
        samplerate=spec.sampling.rate,
        frequency=centers,
        bandwidth=uppers - lowers,
        dims=('frequency',) + spec.dims[1:],
        coords=spec.coords,
    )
    if not scaling == 'density':
        banded *= banded.bandwidth
    return banded


decidecade_filter = nth_decade_filter(bands_per_decade=10, hybrid_resolution=False)
hybrid_millidecade_filter = nth_decade_filter(bands_per_decade=1000, hybrid_resolution=1)


def convert_to_radiated_noise(source_power, source_depth, mode=None, power=True):
    if mode is None or not mode:
        return source_power
    kd = 2 * np.pi * source_power.frequency / 1500 * source_depth
    mode = mode.lower()
    if mode == 'iso':
        compensation = (14 * kd**2 + 2 * kd**4) / (14 + 2 * kd**2 + kd**4)
    elif mode == 'average farfield':
        compensation = 1 / (1 / 2 + 1 / (2 * kd**2))
    elif mode == 'isomatch':
        truncation_angle = np.radians(54.3)
        lf_comp = 2 * kd**2 * (truncation_angle - np.sin(truncation_angle) * np.cos(truncation_angle)) / truncation_angle
        compensation = 1 / (1 / 2 + 1 / lf_comp)
    elif mode == 'none':
        compensation = 1
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    if power:
        return source_power * compensation
    else:
        return source_power + 10 * np.log10(compensation)
