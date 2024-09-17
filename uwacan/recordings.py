"""Reading recordings from files on disk.

This module contains classes used to read data created by
field data recorders, typically recording hydrophone data
as audio files.

.. currentmodule:: uwacan.recordings

Main recording classes
----------------------
.. autosummary::
    :toctree: generated

    SoundTrap
    SylenceLP
    LoggerheadDSG
    MultichannelAudioInterfaceRecording

Utilities
---------
.. autosummary::
    :toctree: generated

    RecordingArray
    TimeCompensation
    calibrate_raw_data
    dBx_to_peak_volts

Implementation interfaces
-------------------------
.. autosummary::
    :toctree: generated

    Recording
    FileRecording
    AudioFileRecording

"""

import numpy as np
from . import _core, positional
import abc
import soundfile
import pendulum
import xarray as xr
from pathlib import Path


def dBx_to_peak_volts(db):
    """Convert dBu or dBV to peak volts.

    Parameters
    ----------
    db : str
        Decibel value as a string with units, e.g., ``"10dBu"``, ``"-20dBV"``.

    Returns
    -------
    volts : float
        Peak voltage corresponding to the input decibel value.

    Raises
    ------
    ValueError
        If the input string does not contain a valid dB unit (``"dBu"`` or ``"dBV"``).
    """
    if not np.ndim(db) == 0:
        return np.vectorize(dBx_to_peak_volts)(db)
    db = db.lower()
    if "dbu" in db:
        dbu = float(db.replace("dbu", "").strip())
        # dBu is an RMS level -> multiply with 2**0.5
        # dBu reference is 1mW over 600Ω, i.e. sqrt(0.6) volts
        volts = 10 ** (dbu / 20) * 2**0.5 * 0.6**0.5
    elif "dbv" in db:
        dbv = float(db.replace("dbv", "").strip())
        # dBV is an RMS level -> multiply with 2**0.5
        # dBV reference is 1V
        volts = 10 ** (dbv / 20) * 2**0.5
    else:
        raise ValueError(f"Unknown dB volts reference unit in {db}")
    return volts


def calibrate_raw_data(
    raw_data,
    sensitivity=None,
    gain=None,
    adc_range=None,
    file_range=None,
):
    """Calibrates raw data read from files into physical units.

    There are three conversion steps handled in this calibration function:

    1) The transducer conversion from physical quantity ``q`` into voltage ``u``
    2) Amplification of the transducer voltage ``u`` to ADC voltage ``v``
    3) Conversion from ADC voltage ``v`` to digital values ``d`` in the file.

    The sensitivity and gain inputs to this function are in decibels, converted to linear
    values as ``s = 10 ** (sensitivity / 20)`` and ``g = 10 ** (gain / 20)``.
    The ``adc_range`` is specified as the peak voltage that the ADC can handle,
    which should be recorded as ``file_range`` in the raw data.

    The equations that govern this are

    1) ``u = q * s``, sensitivity ``s`` in V/Q, e.g. V/Pa.
    2) ``v = u * g``, gain ``g`` is unitless.
    3) ``d / d_ref = v / v_ref``, relating file values to ADC voltage input.

    for a final expression of ``q = d * (v_ref / d_ref / s / g)``.
    All conversion factors default to 1 if not given.

    Parameters
    ----------
    raw_data : array_like
        The raw input data read from a file.
    sensitivity : array_like
        Sensitivity of the sensor, in dB re. V/Q,
        where Q is the desired physical unit.
    gain : array_like
        The gain applied to the voltage from the sensor, in dB.
    adc_range : array_like
        The peak voltage that the ADC can handle.
    file_range : array_like
        The peak value that the raw data contains,
        corresponding to the ``adc_range``.

    Returns
    -------
    q : array_like
        The calibrated values, as per the equations above.

    """
    calibration = 1.0
    # Avoiding in-place operations since they cannot handle broadcasting
    if adc_range is not None:
        calibration = calibration * adc_range
    if file_range is not None:
        calibration = calibration / file_range
    if gain is not None:
        calibration = calibration / 10 ** (gain / 20)
    if sensitivity is not None:
        calibration = calibration / 10 ** (sensitivity / 20)

    return raw_data * calibration


class _LazyPropertyMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__property_cache = {}

    @staticmethod
    def _lazy_property(key):
        def getter(self):
            try:
                return self.__property_cache[key]
            except KeyError:
                self.__property_cache.update(self._lazy_load())
            return self.__property_cache[key]

        return property(getter)

    @abc.abstractmethod
    def _lazy_load(self):
        return {}


class TimeCompensation:
    """Compensates time drift and offset in a recording.

    This is based on the actual and recorded time of one or more events.
    These have to be detected elsewhere, and the times for them are
    given here to build the model.
    If a single pair of times is given, the offset between them is used to compensate.
    If multiple pairs are given, the offset will be linearly interpolated between them.

    Parameters
    ----------
    actual_time : time_like or [time_like]
        Actual time for synchronization event(s).
    recorded_time : time_like or [time_like]
        Recorded time for synchronization event(s).
    """

    def __init__(self, actual_time, recorded_time):
        if isinstance(actual_time, str):
            actual_time = [actual_time]
        if isinstance(recorded_time, str):
            recorded_time = [recorded_time]
        try:
            iter(actual_time)
        except TypeError:
            actual_time = [actual_time]
        try:
            iter(recorded_time)
        except TypeError:
            recorded_time = [recorded_time]

        actual_time = list(map(_core.time_to_datetime, actual_time))
        recorded_time = list(map(_core.time_to_datetime, recorded_time))

        self._time_offset = [
            (recorded - actual).total_seconds() for (recorded, actual) in zip(recorded_time, actual_time)
        ]
        if len(self._time_offset) > 1:
            self._actual_timestamps = [t.timestamp() for t in actual_time]
            self._recorded_timestamps = [t.timestamp() for t in recorded_time]

    def recorded_to_actual(self, recorded_time):
        """Convert a recorded time to the actual time."""
        recorded_time = _core.time_to_datetime(recorded_time)
        if len(self._time_offset) == 1:
            time_offset = self._time_offset[0]
        else:
            time_offset = np.interp(recorded_time.timestamp(), self._recorded_timestamps, self._time_offset)
        return recorded_time - pendulum.duration(seconds=time_offset)

    def actual_to_recorded(self, actual_time):
        """Convert an actual time to the time recorded."""
        actual_time = _core.time_to_datetime(actual_time)
        if len(self._time_offset) == 1:
            time_offset = self._time_offset[0]
        else:
            time_offset = np.interp(actual_time.timestamp(), self._actual_timestamps, self._time_offset)
        return actual_time + pendulum.duration(seconds=time_offset)


class Recording:
    """Base class for recordings.

    This class defines the interface for what a
    recording needs to implement for the rest
    of the package to use it.
    """

    def __init__(self, sensor=None):
        self.sensor = sensor

    @property
    @abc.abstractmethod
    def samplerate(self):
        """The samplerate of the recording, in Hz."""

    @property
    @abc.abstractmethod
    def num_channels(self):
        """The number of channel in the recording, and the read data."""

    @property
    @abc.abstractmethod
    def time_window(self):
        """A `~uwacan.TimeWindow` that covers the recording."""

    @abc.abstractmethod
    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        """Select a subset of the recording.

        See `~uwacan.TimeWindow.subwindow` for details on the parameters.
        """

    @abc.abstractmethod
    def time_data(self):
        """Read stored time data.

        This method reads the recorded data from
        disk, and returns it as a `~uwacan.TimeData` object.
        """


class RecordingArray(Recording):
    """Holds multiple separate recordings.

    This class handles multiple different recording
    instances at once. This is typically needed
    when more than one hardware recorder was used
    for a field trial, and the data from them should
    be analyzed together.

    Parameters
    ----------
    *recordings : `Recording`
        The recording objects.
    """

    def __init__(self, *recordings):
        self.recordings = {recording.sensor.label: recording for recording in recordings}

    @property
    def samplerate(self):
        """The samplerate(s) of the recordings."""
        rates = [recording.samplerate for recording in self.recordings.values()]
        if np.ptp(rates) == 0:
            return rates[0]
        return xr.DataArray(rates, dims="sensor", coords={"sensor": list(self.recordings.keys())})

    @property
    def num_channels(self):
        """The total number of channels."""
        return sum(recording.num_channels for recording in self.recordings.values())

    @property
    def sensor(self):
        """The sensors used, as a `~uwacan.sensor_array`."""
        return positional.SensorArray.concatenate([rec.sensor for rec in self.recordings.values()])

    @property
    def time_window(self):  # noqa: D102, takes the docstring from the superclass
        windows = [recording.time_window for recording in self.recordings.values()]
        return _core.TimeWindow(
            start=max(w.start for w in windows),
            stop=min(w.stop for w in windows),
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):  # noqa: D102, takes the docstring from the superclass
        subwindow = self.time_window.subwindow(
            time, start=start, stop=stop, center=center, duration=duration, extend=extend
        )
        return type(self)(*[recording.subwindow(subwindow) for recording in self.recordings.values()])

    def time_data(self):  # noqa: D102, takes the docstring from the superclass
        if np.ndim(self.samplerate) > 0:
            raise NotImplementedError("Stacking time data from recording with different samplerates not implemented!")
        return _core.TimeData(
            xr.concat([recording.time_data().data for recording in self.recordings.values()], dim="sensor")
        )


class FileRecording(Recording):
    """Base class for recordings using multiple files.

    This class has some interface definitions and some
    shared logic for implementing recordings that use
    multiple files to store the data.

    Subclasses need to implement a `RecordedFile` inner class,
    some way to read the files (typically a classmethod), and
    the `time_data` function (typically using `raw_data`).

    .. autoclass:: uwacan.recordings::FileRecording.RecordedFile

    """

    allowable_interrupt = 0
    """How long gap is allowed between files when reading."""

    class RecordedFile(abc.ABC):
        """Interface class for single recording files.

        This interface class defines how subclasses
        should implement wrappers around individual files.
        """

        def __init__(self, filepath):
            super().__init__()
            self.filepath = Path(filepath)

        @property
        def filepath(self):
            """The `Path` to the file."""
            return self._filepath

        @filepath.setter
        def filepath(self, filepath):
            if not isinstance(filepath, Path):
                filepath = Path(filepath)
            self._filepath = filepath

        @abc.abstractmethod
        def read_data(self, start_idx, stop_idx):
            """Read raw data from the file.

            Parameters
            ----------
            start_idx : int
                The starting index to read from, inclusive.
            stop_idx : int
                The last index to read to, exclusive.

            Returns
            -------
            data : array_like
                The data read from disk.
            """

        @property
        @abc.abstractmethod
        def start_time(self):
            """The start time of this file."""

        @property
        @abc.abstractmethod
        def stop_time(self):
            """The stop time of this file."""

        @property
        @abc.abstractmethod
        def duration(self):
            """The duration of this file."""

        @property
        @abc.abstractmethod
        def num_samples(self):
            """The number of samples in this file, per channel."""

        @property
        @abc.abstractmethod
        def num_channels(self):
            """The number of channels in this file."""

        @property
        @abc.abstractmethod
        def samplerate(self):
            """The samplerate in this file."""

        def __bool__(self):
            return self.filepath.exists()

        def __contains__(self, time):
            return (self.start_time <= time) and (time <= self.stop_time)

    def __init__(self, files, assume_sorted=False, **kwargs):
        super().__init__(**kwargs)
        if not assume_sorted:
            files = sorted(files, key=lambda f: f.start_time)
        self.files = files

    @property
    def samplerate(self):  # noqa: D102, takes the docstring from the superclass
        return self.files[0].samplerate

    @property
    def num_channels(self):  # noqa: D102, takes the docstring from the superclass
        return self.files[0].num_channels

    @property
    def time_window(self):  # noqa: D102, takes the docstring from the superclass
        try:
            return self._window
        except AttributeError:
            self._window = _core.TimeWindow(
                start=self.files[0].start_time,
                stop=self.files[-1].stop_time,
            )
        return self._window

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):  # noqa: D102, takes the docstring from the superclass
        new_window = self.time_window.subwindow(
            time, start=start, stop=stop, center=center, duration=duration, extend=extend
        )
        new = type(self)(
            files=self.files,
            sensor=self.sensor,
        )
        new._window = new_window
        return new

    def _find_file_time(self, time):
        """Find a file containing a certain time."""
        time = _core.time_to_datetime(time)
        for file in reversed(self.files):
            # Going backwards since the files check the start time first,
            # so all files starting later will not have to check their stop time.
            # The stop time usually requires opening the file, which is slower.
            # This will also prioritize a later file, which is good since we're
            # usually looking for a file to start reading from (and then it's
            # better to have more samples left in the file).
            if time in file:
                return file
        else:
            raise ValueError(f"Time {time} does not exist inside any recorded files")

    def check_file_continuity(self, start_time=None, stop_time=None, allowable_interrupt=None, mode="raise"):
        """Check the continuity of recorded data.

        Parameters
        ----------
        start_time : datetime, optional
            The start time of the period to check for continuity. If not provided,
            the start of `self.time_window` will be used.
        stop_time : datetime, optional
            The stop time of the period to check for continuity. If not provided,
            the end of `self.time_window` will be used.
        allowable_interrupt : float, optional
            How much of a gap to allow between files. Will by default use the
            class attribute.
        mode : {"raise", "return", "print"}, optional
            The action to take when an interruption is found.
            - "raise" (default): raises a `ValueError` with details about the interruption.
            - "return": returns `False` if an interruption is found, `True` otherwise.
            - "print": prints a warning message with details about the interruption and continues execution.

        Returns
        -------
        bool
            Returns `True` if the data is continuous within the specified time range.
            If mode is set to "return", it returns `False` if an interruption is found.
            No return value if mode is set to "raise" or "print".

        Raises
        ------
        ValueError
            If `mode` is set to "raise" and an interruption larger than `self.allowable_interrupt`
            is detected between the files, a `ValueError` is raised with details of the missing time.

        Notes
        -----
        The method checks the continuity of data by comparing the `stop_time` of each file
        with the `start_time` of the next file within the specified range. If the gap between
        two files exceeds `self.allowable_interrupt` (in seconds), it is considered an interruption.
        """
        if start_time is None:
            start_time = self.time_window.start
        if stop_time is None:
            stop_time = self.time_window.stop
        if allowable_interrupt is None:
            allowable_interrupt = self.allowable_interrupt
        first_file = self._find_file_time(start_time)
        first_idx = self.files.index(first_file)
        last_file = self._find_file_time(stop_time)
        last_idx = self.files.index(last_file)

        for early, late in zip(self.files[first_idx : last_idx - 1], self.files[first_idx + 1 : last_idx]):
            interrupt = (late.start_time - early.stop_time).total_seconds()
            if interrupt > allowable_interrupt:
                message = (
                    f"Data is not continuous, missing {interrupt} seconds between files\n "
                    f"{early.filepath} ending at {early.stop_time}\n"
                    f"{late.filepath} starting at {late.start_time}"
                )
                if mode == "raise":
                    raise ValueError(message)
                elif mode == "return":
                    return False
                else:
                    print(message)
        return True

    def raw_data(self, reference_time=None, start_offset=None, samples_to_read=None):
        """Read raw data from multiple files.

        Retrieve raw data samples starting from a specified reference time, with an optional
        offset and a specified number of samples to read. By default, all the data from
        ``self.time_window.start`` to ``self.time_window.stop`` will be read.
        This method handles file continuity and reads across multiple files if necessary.

        Parameters
        ----------
        reference_time : date-like, optional
            The reference time from which to count samples. If not provided,
            defaults to the start time of `self.time_window`.
        start_offset : int, optional
            The number of samples to offset from the reference time. Positive values move forward,
            negative values move backward. Defaults to 0 (start from the reference time).
        samples_to_read : int, optional
            The number of samples to read. If not provided, it will calculate the number of
            samples available from the reference time until the end of the time window.

        Returns
        -------
        numpy.ndarray
            The raw data read from the files, concatenated into a single NumPy array.

        Notes
        -----
        This function reads raw data from one or more files, starting from the `reference_time`.
        The data might span multiple files, and this function ensures that the data is read
        continuously, across file boundaries if necessary.
        The default use case will read all the data selected as the time window for the recorder.
        Using a custom reference time, start offset, and samples to read allows consistent reading
        of consecutive samples across file boundaries.
        The start times of files are often not exact, but the samples are measured consecutively.
        """
        samplerate = self.samplerate
        if reference_time is None:
            reference_time = self.time_window.start
        else:
            reference_time = _core.time_to_datetime(reference_time)
        if start_offset is None:
            start_offset = 0
        if samples_to_read is None:
            samples_to_read = int(
                (self.time_window.stop - reference_time.add(seconds=start_offset / samplerate)).total_seconds()
                * samplerate
            )

        earliest = min(reference_time, reference_time.add(seconds=start_offset / samplerate))
        latest = max(reference_time, reference_time.add(seconds=(start_offset + samples_to_read) / samplerate))
        self.check_file_continuity(earliest, latest)

        # We need to find where to start reading.
        # Our pointer is `start_idx` in the `file_idx` file.
        # This needs to be moved by `start_offset` samples.
        file_idx = self.files.index(self._find_file_time(reference_time))
        start_idx = int(np.floor((reference_time - self.files[file_idx].start_time).total_seconds() * samplerate))
        if start_offset > 0:
            remaining_offset = start_offset
            # We should look forwards to find the first file
            while remaining_offset != 0:
                # Will current start_index + remaining_offset take us outside this file?
                if self.files[file_idx].num_samples <= remaining_offset + start_idx:
                    # Remove the remainder of this file from the start offset
                    remaining_offset -= self.files[file_idx].num_samples - start_idx
                    # Point to the beginning of the next file.
                    file_idx += 1
                    start_idx = 0
                else:
                    # This is the first file to read, starting at the remaining offset.
                    start_idx += remaining_offset
                    remaining_offset = 0
        elif start_offset < 0:
            remaining_offset = start_offset
            # We should look backwards to find the first file
            while remaining_offset != 0:
                # Will current start_index + remaining_offset take us outside this file?
                if start_idx + remaining_offset < 0:
                    # There is start_idx samples left in this file, and we also move one step more to get into the next file.
                    remaining_offset += start_idx + 1
                    # Point to the last sample of the previous file.
                    file_idx -= 1
                    start_idx = self.files[file_idx].num_samples - 1
                else:
                    # This is the first file to read.
                    # Shift the pointer by remaining offset.
                    start_idx += remaining_offset
                    remaining_offset = 0

        # Now, file_idx and start_idx point to the first sample to read.

        if self.files[file_idx].num_samples - start_idx >= samples_to_read:
            # The data exists within this file, we can use an early return from here
            data = self.files[file_idx].read_data(start_idx, start_idx + samples_to_read)
            return data

        # Read data from consecutive files
        chunks = []
        remaining_samples = samples_to_read
        for file in self.files[file_idx:]:
            chunk = file.read_data(start_idx=start_idx, stop_idx=min(remaining_samples + start_idx, file.num_samples))
            remaining_samples -= chunk.shape[0]
            chunks.append(chunk)
            start_idx = 0
            if remaining_samples <= 0:
                break
        data = np.concatenate(chunks, axis=0)
        if data.shape[-1] != samples_to_read:
            raise RuntimeError(f"Read {data.shape[0]} samples from files, expected {samples_to_read} samples.")

        return data

    def select_file_time(self, time):
        """Get a recording for a specific file, by time.

        This finds the file corresponding to a specific time,
        then returns a recording subwindow corresponding
        to that file.
        """
        time = _core.time_to_datetime(time)
        for file in reversed(self.files):
            if file.start_time > time:
                continue
            if file.stop_time < time:
                raise ValueError(f"Time {time} does not exist inside any recorded files.")
            return self.subwindow(start=file.start_time, stop=file.stop_time)

    def select_file_name(self, name):
        """Get a recording for a specific file, by name.

        This finds the file with a specific name,
        then returns a recording subwindow corresponding
        to that file.
        """
        stem = Path(name).stem
        for file in self.files:
            if stem == file.filepath.stem:
                return self.subwindow(start=file.start_time, stop=file.stop_time)
        raise ValueError(f"Could not file file matching name '{name}'")

    def rolling(self, duration=None, step=None, overlap=0, copy_on_out=False):
        """Generate rolling frames of raw time data.

        Parameters
        ----------
        duration : float
            The size of each frame, in seconds.
        step : float
            The step between consecutive frames, in seconds.
        overlap : float, default=0
            The fraction of overlap between consecutive frames. Should be less than one.
            Negative values will make "gaps" in the output.
        copy_on_out : bool, default=False
            This sets if the frames will be copied before yielded.
            If False, all output frames point to the same memory buffer.

        Yields
        ------
        np.ndarray
            The raw data for each frame.
        """
        samplerate = self.samplerate
        start_time = self.time_window.start
        stop_time = self.time_window.stop

        settings = _core.time_frame_settings(duration=duration, step=step, overlap=overlap, signal_length=(stop_time - start_time).total_seconds(), samplerate=samplerate)
        samples_per_frame = settings["samples_per_frame"]
        sample_overlap = settings["sample_overlap"]
        sample_reuse = max(0, sample_overlap)

        buffer = np.zeros((samples_per_frame, self.num_channels)).squeeze()
        if sample_reuse:
            buffer[samples_per_frame - sample_reuse :] = self.raw_data(
                reference_time=start_time, start_offset=0, samples_to_read=sample_reuse
            )

        for frame_idx in range(settings["num_frames"]):
            buffer[:sample_reuse] = buffer[samples_per_frame - sample_reuse :]
            start_idx = frame_idx * (samples_per_frame - sample_overlap)
            buffer[sample_reuse:] = self.raw_data(
                reference_time=start_time, start_offset=start_idx + sample_reuse, samples_to_read=samples_per_frame - sample_reuse
            )
            out = buffer.copy() if copy_on_out else buffer
            yield out


class AudioFileRecording(FileRecording):
    """Class for audio file recordings.

    This class handles reading audio files using the
    `soundfile` python package.
    This is a fully functional class, but reading data
    requires a ``start_time_parser`` function passed to the
    `read_folder` classmethod. A more convenient approach
    is to subclass this class and customize the `read_folder`
    classmethod.
    """

    file_range = None
    """The input range of the read files."""
    gain = None
    """The gain of this recording."""
    adc_range = None
    """The voltage peak range of the adc in this recording."""

    @classmethod
    def read_folder(
        cls,
        folder,
        start_time_parser,
        sensor=None,
        file_filter=None,
        time_compensation=None,
        glob_pattern="**/*.wav",
        file_kwargs=None,
    ):
        """Read all matching files in a folder and parse their start times.

        Parameters
        ----------
        folder : str or Path
            The path to the folder containing the files.
        start_time_parser : str or callable
            If a string is provided, it is treated as a format string and will be used
            to parse the start time from the filename. If a callable is provided, it
            should accept a file path and return a `pendulum.DateTime` object representing the start time.
        sensor : str or None, optional
            The sensor associated with the files.
        file_filter : callable or None, optional
            A callable that accepts a file path and returns True if the file should be processed,
            and False otherwise. If None, all files matching the ``glob_pattern`` are processed.
        time_compensation : `TimeCompensation`, int, or callable, optional
            - If a `TimeCompensation` object is provided, it is used to adjust the recorded times.
            - If an number is provided, it is treated as a time offset in seconds and subtracted from recorded times.
            - If a callable is provided, it should accept a timestamp and return a compensated timestamp.
            - If None, no time compensation is applied.
        glob_pattern : str, optional
            A glob pattern used to match files in the folder, by default ``"**/*.wav"``.
        file_kwargs : dict or callable, optional
            Additional keyword arguments to be passed when creating the `RecordedFile` instances.
            If a callable is provided, it should accept a file path and return a dictionary of keyword arguments.
            If None, no additional keyword arguments are passed to the files.

        Returns
        -------
        cls
            An instance of the class containing the loaded files.

        Raises
        ------
        RuntimeError
            If the folder does not exist, is not a directory, or no matching files are found.
        """
        folder = Path(folder)
        if not folder.exists():
            raise RuntimeError(f"'{folder}' does not exist")
        if not folder.is_dir():
            raise RuntimeError(f"'{folder}' is not a folder")

        if isinstance(start_time_parser, str):
            start_time_format = start_time_parser

            def start_time_parser(file):
                return pendulum.from_format(file.stem, start_time_format)

        if time_compensation is None:

            def time_compensation(timestamp):
                return timestamp

        if isinstance(time_compensation, TimeCompensation):
            time_compensation = time_compensation.recorded_to_actual
        if not callable(time_compensation):
            offset = pendulum.duration(seconds=time_compensation)

            def time_compensation(timestamp):
                return timestamp - offset

        if file_filter is None:

            def file_filter(filepath):
                return True

        if file_kwargs is None:

            def file_kwargs(filepath):
                return {}

        if not callable(file_kwargs):
            _file_kwargs = file_kwargs

            def file_kwargs(filepath):
                return _file_kwargs

        files = []
        for file in Path(folder).glob(glob_pattern):
            if file_filter(file):
                start_time = start_time_parser(file)
                files.append(cls.RecordedFile(file, time_compensation(start_time), **file_kwargs(file)))

        if not files:
            raise RuntimeError(f"No matching files found in '{folder}'")

        return cls(
            files=files,
            sensor=sensor,
        )

    class RecordedFile(FileRecording.RecordedFile, _LazyPropertyMixin):
        """Wrapper for audio files."""

        def __init__(self, filepath, start_time):
            super().__init__(filepath=filepath)
            self._start_time = start_time

        def _lazy_load(self):
            sfi = soundfile.info(self.filepath.as_posix())
            return super()._lazy_load() | dict(
                num_samples=sfi.frames,
                num_channels=sfi.channels,
                samplerate=sfi.samplerate,
            )

        @property
        def start_time(self):  # noqa: D102, takes the docstring from the superclass
            return self._start_time

        num_samples = _LazyPropertyMixin._lazy_property("num_samples")
        num_channels = _LazyPropertyMixin._lazy_property("num_channels")
        samplerate = _LazyPropertyMixin._lazy_property("samplerate")

        @property
        def stop_time(self):  # noqa: D102, takes the docstring from the superclass
            return self.start_time.add(seconds=self.duration)

        @property
        def duration(self):  # noqa: D102, takes the docstring from the superclass
            return self.num_samples / self.samplerate

        def read_data(self, start_idx=None, stop_idx=None):  # noqa: D102, takes the docstring from the superclass
            return soundfile.read(self.filepath.as_posix(), start=start_idx, stop=stop_idx, dtype="float32")[0]

    def time_data(self):  # noqa: D102, takes the docstring from the superclass
        data = self.raw_data()
        if np.ndim(data) == 1:
            dims = "time"
            coords = None
        elif np.ndim(data) == 2:
            if self.sensor is not None and "sensor" in self.sensor and np.shape(data)[1] == self.sensor["sensor"].size:
                dims = ("time", "sensor")
                coords = {"sensor": self.sensor["sensor"]}
            else:
                dims = ("time", "channel")
                if self.sensor is not None and "channel" in self.sensor:
                    coords = {"channel": self.sensor["channel"]}
                else:
                    coords = None
        else:
            raise NotImplementedError("Audio files with more than 2 dimensions are not supported")
        data = _core.TimeData(
            data=data,
            samplerate=self.samplerate,
            start_time=self.time_window.start,
            dims=dims,
            coords=coords,
        )
        data = calibrate_raw_data(
            raw_data=data,
            sensitivity=self.sensor.get("sensitivity", None),
            gain=self.gain,
            adc_range=self.adc_range,
            file_range=self.file_range,
        )
        return data

    def rolling(self, duration=None, step=None, overlap=0, return_numpy=False):
        """Generate rolling frames of time data.

        Parameters
        ----------
        duration : float
            The size of each frame, in seconds.
        step : float
            The step between consecutive frames, in seconds.
        overlap : float, default=0
            The fraction of overlap between consecutive frames. Should be less than one.
            Negative values will make "gaps" in the output.
        return_numpy : bool, default=False
            Whether to return the output as a NumPy array (default False).
            If False, returns a `uwacan.TimeData` object.

        Yields
        ------
        np.ndarray or uwacan._core.TimeData
            The data for each frame, calibrated if applicable.
        """
        settings = _core.time_frame_settings(duration=duration, step=step, overlap=overlap, samplerate=self.samplerate)
        calibration = calibrate_raw_data(
            1,
            gain=self.gain,
            sensitivity=self.sensor.get("sensitivity"),
            adc_range=self.adc_range,
            file_range=self.file_range,
        )
        dummy_data = self.subwindow(start=True, duration=0).time_data().data
        calibration = xr.align(dummy_data, calibration)[1].data

        start_time = self.time_window.start
        offsets = np.arange(settings["samples_per_frame"]) * 1e9 / self.samplerate
        time = _core.time_to_np(start_time) + offsets.astype("timedelta64[ns]")

        for frame_idx, frame in enumerate(super().rolling(duration=duration, step=step, overlap=overlap, copy_on_out=False)):
            frame = frame * calibration
            if return_numpy:
                yield frame
            else:
                # Rounding the sample count to nanoseconds in each frame avoids accumulating rounding errors
                yield _core.TimeData(
                    frame,
                    time=time + np.timedelta64(int(frame_idx * settings["sample_step"] * 1e9 / self.samplerate), "ns"),
                    samplerate=self.samplerate,
                    coords=dummy_data.coords,
                    dims=dummy_data.dims,
                )


class SoundTrap(AudioFileRecording):
    """Class to read data from OceanInstruments SoundTrap recorders.

    The main way to read SoundTrap data is through the
    `read_folder` classmethod.
    """

    allowable_interrupt = 1
    gain = None
    adc_range = None
    file_range = 1

    @classmethod
    def read_folder(cls, folder, sensor=None, serial_number=None, time_compensation=None):
        """Read files in a folder, filtered on an optional serial number.

        Parameters
        ----------
        folder : str or Path
            The path to the folder containing the files.
        sensor : str or None, optional
            The sensor associated with the files.
        serial_number : int or None, optional
            If provided, only files with the matching serial number in their filename will be processed.
            If None, all files in the folder will be processed.
        time_compensation : `TimeCompensation`, int, or callable, optional
            - If a `TimeCompensation` object is provided, it is used to adjust the recorded times.
            - If an number is provided, it is treated as a time offset in seconds and subtracted from recorded times.
            - If a callable is provided, it should accept a timestamp and return a compensated timestamp.
            - If None, no time compensation is applied.

        Returns
        -------
        cls
            An instance of the class containing the loaded files.

        Raises
        ------
        RuntimeError
            If the folder does not exist, is not a directory, or no matching files are found.

        Notes
        -----
        This method filters the files in the folder based on the provided ``serial_number`` and
        parses the start time from the filenames using a specific format (``"YYMMDDHHmmss"``).
        It then delegates the actual file reading to the `read_folder` method of the parent class.
        """
        if serial_number is None:

            def file_filter(filepath):
                return True
        else:

            def file_filter(filepath):
                return int(filepath.stem.split(".")[0]) == serial_number

        def start_time_parser(filepath):
            return pendulum.from_format(filepath.stem.split(".")[1], "YYMMDDHHmmss")

        return super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
        )


class SylenceLP(AudioFileRecording):
    """Class to read data from RTsys SylenceLP recorders.

    The main way to read Sylence data is through the
    `read_folder` classmethod.
    """

    adc_range = 2.5
    file_range = 1
    allowable_interrupt = 1

    class RecordedFile(AudioFileRecording.RecordedFile):  # noqa: D106, takes the docstring from the superclass
        def _lazy_load(self):  # noqa: D102, takes the docstring from the superclass
            with self.filepath.open("rb") as file:
                base_header = file.read(36)
                # chunk_id = base_header[0:4].decode('ascii')  # always equals RIFF
                # file_size = int.from_bytes(base_header[4:8], byteorder='little', signed=False)  # total file size not important
                # chunk_format = base_header[8:12].decode('ascii')  # always equals WAVE
                # subchunk_id = base_header[12:16].decode('ascii')  # always equals fmt
                # subchunk_size = int.from_bytes(base_header[16:20], byteorder='little', signed=False))  # always equals 16
                # audio_format = int.from_bytes(base_header[20:22], byteorder='little', signed=False))  # not important in current implementation
                num_channels = int.from_bytes(base_header[22:24], byteorder="little", signed=False)
                if num_channels != 1:
                    raise ValueError(
                        f"Expected file for SylenceLP with a single channel, read file with {num_channels} channels"
                    )
                samplerate = int.from_bytes(base_header[24:28], byteorder="little", signed=False)
                # byte rate = int.from_bytes(base_header[28:32], byteorder='little', signed=False)  # not important in current implementation
                bytes_per_sample = int.from_bytes(base_header[32:34], byteorder="little", signed=False)
                bitdepth = int.from_bytes(base_header[34:36], byteorder="little", signed=False)

                conf_header = file.peek(8)  # uses peak to keep indices aligned with the manual
                conf_size = int.from_bytes(conf_header[4:8], byteorder="little", signed=False)
                if conf_size != 460:
                    raise ValueError(f"Incorrect size of SylenceLP config: '{conf_size}'B, expected 460B")
                conf_header = file.read(conf_size + 8)

                subchunk_id = conf_header[:4].decode("ascii")  # always conf
                if subchunk_id != "conf":
                    raise ValueError(f"Expected 'conf' section in SylenceLP config, found '{subchunk_id}'")
                # subchunk_size = int.from_bytes(conf_header[4:8], byteorder='little', signed=False)  # the same as conf_size
                config_version = int.from_bytes(conf_header[8:12], byteorder="little", signed=False)
                if config_version != 2:
                    raise NotImplementedError(f"Cannot handle SylenceLP config version {config_version}")
                # recording_start = datetime.datetime.fromtimestamp(int.from_bytes(conf_header[16:24], byteorder='little', signed=True))  # This value is not actually when the recording starts. No idea what it actually is
                channel = conf_header[24:28].decode("ascii")
                if channel.strip("\x00") != "":
                    raise NotImplementedError(
                        f"No implementation for multichannel SylenceLP recorders, found channel specification '{channel}'"
                    )
                samplerate_alt = np.frombuffer(conf_header[28:32], dtype="f4").squeeze()
                if samplerate != samplerate_alt:
                    raise ValueError(
                        f"Mismatched samplerate for hardware and file, read file samplerate {samplerate} and config samplerate {samplerate_alt}"
                    )

                hydrophone_sensitivity = np.frombuffer(conf_header[32:48], dtype="f4")
                gain = np.frombuffer(conf_header[48:64], dtype="f4")
                # gain_correction = np.frombuffer(conf_header[64:80], dtype='f4')  # is just 1/gain
                serialnumber = conf_header[80:100].decode("ascii")
                active_channels = conf_header[100:104].decode("ascii")
                if active_channels != "A\x00\x00\x00":
                    raise NotImplementedError(
                        f"No implementation for multichannel SylenceLP recorders, found channel specification '{active_channels}'"
                    )

                data_header = file.read(4).decode("ascii")
                if data_header != "data":
                    raise ValueError(f"Expected file header 'data', read {data_header}")
                data_size = int.from_bytes(file.read(4), byteorder="little", signed=False)

            num_samples = data_size / bytes_per_sample
            if int(num_samples) != num_samples:
                raise ValueError(f"Size of data is not divisible by bytes per sample, file '{self.name}' is corrupt!")

            return super()._lazy_load() | dict(
                samplerate=samplerate,
                bitdepth=bitdepth,
                num_samples=int(num_samples),
                hydrophone_sensitivity=hydrophone_sensitivity[0],
                serial_number=serialnumber.strip("\x00"),
                gain=-20 * np.log10(gain[0]),
            )

        bitdepth = _LazyPropertyMixin._lazy_property("bitdepth")
        hydrophone_sensitivity = _LazyPropertyMixin._lazy_property("hydrophone_sensitivity")
        serial_number = _LazyPropertyMixin._lazy_property("serial_number")
        gain = _LazyPropertyMixin._lazy_property("gain")

    @property
    def gain(self):  # noqa: D102, takes the docstring from the superclass
        return self.files[0].gain

    @classmethod
    def read_folder(cls, folder, sensor=None, time_compensation=None, file_filter=None):
        """Read all files in a folder.

        Parameters
        ----------
        folder : str or Path
            The path to the folder containing the files.
        sensor : str or None, optional
            The sensor associated with the files.
        time_compensation : `TimeCompensation`, int, or callable, optional
            - If a `TimeCompensation` object is provided, it is used to adjust the recorded times.
            - If an number is provided, it is treated as a time offset in seconds and subtracted from recorded times.
            - If a callable is provided, it should accept a timestamp and return a compensated timestamp.
            - If None, no time compensation is applied.
        file_filter : callable or None, optional
            A callable that accepts a file path and returns True if the file should be processed,
            and False otherwise. If None, all files are processed.

        Returns
        -------
        cls
            An instance of the class containing the loaded files.

        Raises
        ------
        RuntimeError
            If the folder does not exist, is not a directory, or no matching files are found.

        """

        def start_time_parser(filepath):
            return pendulum.from_format(filepath.stem[9:], "YYYY-MM-DD_HH-mm-ss")

        return super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
        )


class MultichannelAudioInterfaceRecording(AudioFileRecording):
    """Class for handling multichannel audio interface recordings."""

    file_range = 1

    @property
    def gain(self):  # noqa: D102, takes the docstring from the superclass
        return self.sensor.get("gain", None)

    @property
    def adc_range(self):  # noqa: D102, takes the docstring from the superclass
        return self.sensor.get("adc_range", None)

    class RecordedFile(AudioFileRecording.RecordedFile):  # noqa: D106, takes the docstring from the superclass
        def __init__(self, filepath, start_time, channels):
            super().__init__(filepath=filepath, start_time=start_time)
            self.channels = list(channels)

        def read_data(self, start_idx=None, stop_idx=None):  # noqa: D102, takes the docstring from the superclass
            all_channels = soundfile.read(
                self.filepath.as_posix(),
                start=start_idx,
                stop=stop_idx,
                dtype="float32",
                always_2d=True,
            )[0]
            return all_channels[:, self.channels]

        @property
        def num_channels(self):  # noqa: D102, inherits from superclass.
            return len(self.channels)

    @classmethod
    def _merge_channel_info(cls, sensor, channel, gain, adc_range):
        """Merge channel information with the sensor data.

        This function has two main operating modes, depending on if
        there is existing sensor information or not.

        1. There is sensor information: The channel, gain, and adc_range
        will be passed to `uwacan.positional.Sensor.with_data`, and the
        resulting `~uwacan.positional.Sensor` object is returned.
        This allows using dictionaries to supply the channel, gain, and adc_range.
        2. If there is no sensor information: The channels will be used as
        the dimension and coordinate, and must as such be an array_like.
        The gain and adc_range has to be compatible with this channel information.

        Parameters
        ----------
        sensor : `uwacan.positional.Sensor` or None
            The sensor to which the channel information will be merged.
            If None, a new dataset is created, and the output is a dataset.
        channel : array_like, or dict
            Channel information to be added to the sensor dataset.
        gain : array_like, scalar, or dict
            Gain information to be added to the sensor dataset.
        adc_range : array_like, scalar, or dict
            ADC range information to be added to the sensor dataset.
        """
        if sensor is None:
            sensor = xr.Dataset()
            if channel is not None:
                if not isinstance(channel, xr.DataArray):
                    channel = xr.DataArray(channel, dims="channel", coords={"channel": channel})
                sensor["channel"] = channel
            if gain is not None:
                if not isinstance(gain, xr.DataArray) and np.ndim(gain) != 0:
                    gain = xr.DataArray(gain, dims="channel", coords={"channel": channel})
                sensor["gain"] = gain
            if adc_range is not None:
                if not isinstance(adc_range, xr.DataArray) and np.ndim(adc_range) != 0:
                    adc_range = xr.DataArray(adc_range, dims="channel", coords={"channel": channel})
                sensor["adc_range"] = adc_range
            return sensor

        assigns = {}
        if "channel" not in sensor:
            if channel is None:
                channel = list(range(len(sensor.sensors)))
            assigns["channel"] = channel
        elif channel is not None:
            raise ValueError(
                "Should not give explicit channel if the channel information is already in the sensor information"
            )

        if "gain" not in sensor:
            if gain is None:
                gain = 0
            assigns["gain"] = gain
        elif gain is not None:
            raise ValueError(
                "Should not give explicit gain if the gain information is already in the sensor information"
            )

        if "adc_range" not in sensor:
            if adc_range is None:
                adc_range = 1
            assigns["adc_range"] = adc_range
        elif adc_range is not None:
            raise ValueError(
                "Should not give explicit adc_range if the adc_range information is already in the sensor information"
            )
        sensor = sensor.with_data(**assigns)
        return sensor

    @classmethod
    def read_folder(
        cls,
        folder,
        start_time_parser,
        channel=None,
        gain=None,
        adc_range=None,
        one_recording_per_file=False,
        sensor=None,
        file_filter=None,
        time_compensation=None,
        glob_pattern="**/*.wav",
    ):
        """Read files in a folder.

        This method collects audio files from the specified folder into a recording object.
        The sensor and audio interface settings can be supplied in two ways, depending on if
        there is sensor information or not:

        1. There is sensor information: Use `uwacan.sensor_array` to specify
           the sensor particulars. Give the ``channel``, ``gain``, and ``adc_range``
           as dicts with the sensor names as keys, or scalars for all the sensors.
        2. If there is no sensor information: Give channel labels as a list to the ``channel``,
           and array_like or scalar ``gain`` and ``adc_range``.

        Parameters
        ----------
        folder : str or Path
            The folder containing the audio files.
        start_time_parser : callable or str
            - A function to parse the start time from file names, or
            - a sting specifying the datetime format, e.g., ``"YYYY-MM-DD_HH-mm-ss"``.

        sensor : `~uwacan.positional.Sensor`
            Sensor information with sensitivity, positions, etc.
        channel : dict or array_like
            The channel index in the read data, from 0.

            1. A mapping from sensor names to channel index, if sensor information is given.
            2. A list of channel labels, if no sensor information is given.

        gain : dict, array_like, or scalar
            The gain used for the interface, in dB.

            1. A mapping from sensor names to interface gain, if sensor information is given.
            2. A list of gains, if no sensor information is given.
            3. A single gain for all interface channels/sensors.

        adc_range : dict, array_like, or scalar
            The peak voltage input of the ADC.

            1. A mapping from sensor names to interface ADC range, if sensor information is given.
            2. A list of ADC ranges, if no sensor information is given.
            3. A single ADC range for all interface channels/sensors

        one_recording_per_file : bool, optional
            If True, the output will be a list of recordings, one for each file.
        file_filter : callable, optional
            A function to filter files based on specific criteria. Will be called with the file path.
            The file is skipped if the filter returns ``False``.
        time_compensation : `TimeCompensation`, int, or callable, optional
            - If a `TimeCompensation` object is provided, it is used to adjust the recorded times.
            - If an number is provided, it is treated as a time offset in seconds and subtracted from recorded times.
            - If a callable is provided, it should accept a timestamp and return a compensated timestamp.
            - If None, no time compensation is applied.
        glob_pattern : str, optional
            The glob pattern to match files in the folder. Defaults to ``"**/*.wav"``.

        """
        sensor = cls._merge_channel_info(sensor=sensor, channel=channel, gain=gain, adc_range=adc_range)
        recordings = super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
            glob_pattern=glob_pattern,
            file_kwargs={"channels": sensor["channel"].values},
        )
        if not one_recording_per_file:
            return recordings
        return [recordings.subwindow(start=file.start_time, stop=file.stop_time) for file in recordings.files]


class LoggerheadDSG(AudioFileRecording):
    """Class to read data from Loggerhead DSG recorders.

    The main way to read Loggerhead data is through the
    `read_folder` classmethod.
    """

    allowable_interrupt = 1
    adc_range = None
    file_range = 1

    @classmethod
    def read_folder(cls, folder, sensor=None, time_compensation=None, file_filter=None):
        """Read all files in a folder.

        Parameters
        ----------
        folder : str or Path
            The path to the folder containing the files.
        sensor : str or None, optional
            The sensor associated with the files.
        time_compensation : `TimeCompensation`, int, or callable, optional
            - If a `TimeCompensation` object is provided, it is used to adjust the recorded times.
            - If an number is provided, it is treated as a time offset in seconds and subtracted from recorded times.
            - If a callable is provided, it should accept a timestamp and return a compensated timestamp.
            - If None, no time compensation is applied.
        file_filter : callable or None, optional
            A callable that accepts a file path and returns True if the file should be processed,
            and False otherwise. If None, all files are processed.

        Returns
        -------
        cls
            An instance of the class containing the loaded files.

        Raises
        ------
        RuntimeError
            If the folder does not exist, is not a directory, or no matching files are found.

        """

        def start_time_parser(filepath):
            return pendulum.from_format(filepath.stem[:15], "YYYYMMDDTHHmmss")

        return super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
        )

    @property
    def gain(self):  # noqa: D102, takes the docstring from the superclass
        return self.files[0].gain

    class RecordedFile(AudioFileRecording.RecordedFile):  # noqa: D106, takes the docstring from the superclass
        def _lazy_load(self):  # noqa: D102, takes the docstring from the superclass
            gain = self.filepath.stem.split("_")[2]
            if not gain.endswith("dB"):
                raise ValueError(
                    f"File `{self.filepath}` does not seem to be a file from a Loggerhead DSG, could not extract gain"
                )
            return super()._lazy_load() | dict(gain=float(gain.rstrip("dB")))

        gain = _LazyPropertyMixin._lazy_property("gain")
