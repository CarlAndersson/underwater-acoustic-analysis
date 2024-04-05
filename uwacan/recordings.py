import numpy as np
from . import positional, sensors
import abc
import soundfile
import pendulum
import xarray as xr
from pathlib import Path


class _SampleTimer:
    def __init__(self, xr_obj):
        if 'time' not in xr_obj.dims:
            raise TypeError(".sampling accessor only available for xarrays with 'time' coordinate")
        self._xr_obj = xr_obj

    @property
    def rate(self):
        return self._xr_obj.coords['time'].attrs['rate']

    @property
    def num(self):
        return self._xr_obj.sizes['time']

    @property
    def window(self):
        start = positional.time_to_datetime(self._xr_obj.time.data[0])
        # Calculating duration from number and rate means the stop points to the sample after the last,
        # which is more intuitive when considering signal durations etc.
        return positional.TimeWindow(
            start=start,
            duration=self.num / self.rate,
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        original_window = self.window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        if isinstance(new_window, positional.TimeWindow):
            start = (new_window.start - original_window.start).total_seconds()
            stop = (new_window.stop - original_window.start).total_seconds()
            # Indices assumed to be seconds from start
            start = np.math.floor(start * self.rate)
            stop = np.math.ceil(stop * self.rate)
            idx = slice(start, stop)
        else:
            idx = (new_window - original_window.start).total_seconds()
            idx = round(idx * self.rate)

        new_obj = self._xr_obj.isel(time=idx)
        return new_obj


class _StampedTimer:
    def __init__(self, xr_obj):
        self._xr_obj = xr_obj

    @property
    def window(self):
        start = positional.time_to_datetime(self._xr_obj.time.data[0])
        stop = positional.time_to_datetime(self._xr_obj.time.data[-1])
        return positional.TimeWindow(start=start, stop=stop)

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
        original_window = self.window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
        if isinstance(new_window, positional.TimeWindow):
            start = new_window.start.in_tz('UTC').naive()
            stop = new_window.stop.in_tz('UTC').naive()
            return self._xr_obj.sel(time=slice(start, stop))
        else:
            return self._xr_obj.sel(time=new_window.in_tz('UTC').naive(), method='nearest')


def _make_sampler(xr_obj):
    if 'time' not in xr_obj.dims:
        raise TypeError(".sampling accessor only available for xarrays with 'time' coordinate")
    if 'rate' in xr_obj.time.attrs:
        return _SampleTimer(xr_obj)
    else:
        return _StampedTimer(xr_obj)


xr.register_dataarray_accessor('sampling')(_make_sampler)
xr.register_dataset_accessor('sampling')(_make_sampler)


def calibrate_raw_data(
    raw_data,
    sensitivity=None,
    gain=None,
    adc_range=None,
    file_range=None,
):
    """Calibrates raw data read from files into physical units.

    There are three conversion steps handled in this calibration function:
    1) The transducer conversion from physical quantity (q) into voltage (u)
    2) Amplification of the transducer voltage (u) to ADC voltage (v)
    3) Conversion from ADC voltage (v) to digital values (d) in the file.

    The sensitivity and gain inputs to this function are in decibels, converted to linear
    values as `s = 10 ** (sensitivity / 20)` and `g = 10 ** (gain / 20)`.
    The `adc_range` is specified as the peak voltage that the ADC can handle,
    which should be recorded as `file_range` in the raw data.

    The equations that govern this are
    1) `u = q * s`, sensitivity s in V/Q, e.g. V/Pa.
    2) `v = u * g`, gain g is unitless.
    3) `d / d_ref = v / v_ref`, relating file values to ADC voltage input.
    for a final expression of `q = d * (v_ref / d_ref / s / g)`.
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
        corresponding to the `adc_range`.

    Returns
    -------
    q : array_like
        The calibrated values, as per the equations above.

    Note
    ----
        No assumptions about input dimensions are done - the inputs
        should either be scalar or broadcast properly with the raw data.
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


def time_data(data, start_time, samplerate, dims=None, coords=None):
    if not isinstance(data, xr.DataArray):
        if dims is None:
            if data.ndim == 1:
                dims = 'time'
            else:
                raise ValueError(f'Cannot guess dimensions for time data with {data.ndim} dimensions')
        data = xr.DataArray(data, dims=dims)

    n_samples = data.sizes['time']
    start_time = positional.time_to_np(start_time)
    offsets = np.arange(n_samples) * 1e9 / samplerate
    time = start_time + offsets.astype('timedelta64[ns]')
    data = data.assign_coords(
        time=('time', time, {'rate': samplerate}),
        **{name: coord for (name, coord) in (coords or {}).items() if name != 'time'}
    )

    return data


def frequency_data(data, frequency, bandwidth, dims=None, coords=None):
    if not isinstance(data, xr.DataArray):
        if dims is None:
            if data.ndim == 1:
                dims = 'frequency'
            else:
                raise ValueError(f'Cannot guess dimensions for frequency data with {data.ndim} dimensions')
        data = xr.DataArray(data, dims=dims)
    data = data.assign_coords(
        frequency=frequency,
        bandwidth=('frequency', np.broadcast_to(bandwidth, np.shape(frequency))),
         **{name: coord for (name, coord) in (coords or {}).items() if name != 'frequency'}
    )
    return data


def time_frequency_data(data, start_time, samplerate, frequency, bandwidth, dims=None, coords=None):
    if not isinstance(data, xr.DataArray):
        if dims is None:
            raise ValueError('Cannot guess dimensions for time-frequency data')
        data = xr.DataArray(data, dims=dims)
    data = time_data(data, start_time=start_time, samplerate=samplerate)
    data = frequency_data(data, frequency, bandwidth)
    return data.assign_coords(**{name: coord for (name, coord) in (coords or {}).items() if name not in ('time', 'frequency', 'bandwidth')})


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

        actual_time = list(map(positional.time_to_datetime, actual_time))
        recorded_time = list(map(positional.time_to_datetime, recorded_time))

        self._time_offset = [(recorded - actual).total_seconds() for (recorded, actual) in zip(recorded_time, actual_time)]
        if len(self._time_offset) > 1:
            self._actual_timestamps = [t.timestamp() for t in actual_time]
            self._recorded_timestamps = [t.timestamp() for t in recorded_time]

    def recorded_to_actual(self, recorded_time):
        recorded_time = positional.time_to_datetime(recorded_time)
        if len(self._time_offset) == 1:
            time_offset = self._time_offset[0]
        else:
            time_offset = np.interp(recorded_time.timestamp(), self._recorded_timestamps, self._time_offset)
        return recorded_time - pendulum.duration(seconds=time_offset)

    def actual_to_recorded(self, actual_time):
        actual_time = positional.time_to_datetime(actual_time)
        if len(self._time_offset) == 1:
            time_offset = self._time_offset[0]
        else:
            time_offset = np.interp(actual_time.timestamp(), self._actual_timestamps, self._time_offset)
        return actual_time + pendulum.duration(seconds=time_offset)


class Recording(abc.ABC):
    class _Sampling(abc.ABC):
        def __init__(self, recording):
            self.recording = recording

        @property
        @abc.abstractmethod
        def rate(self):
            ...

        @property
        @abc.abstractmethod
        def window(self):
            ...

        @abc.abstractmethod
        def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
            ...

    def __init__(self, sensor=None):
        self.sampling = self._Sampling(self)  # pylint: disable=abstract-class-instantiated
        self.sensor = sensor

    @property
    @abc.abstractmethod
    def num_channels(self):
        ...

    @abc.abstractmethod
    def time_data(self):
        ...


class RecordingArray(Recording):
    class _Sampling(Recording._Sampling):
        @property
        def rate(self):
            rates = [recording.sampling.rate for recording in self.recording.recordings.values()]
            if np.ptp(rates) == 0:
                return rates[0]
            return xr.DataArray(rates, dims='sensor', coords={'sensor': list(self.recording.recordings.keys())})

        @property
        def window(self):
            windows = [recording.sampling.window for recording in self.recording.recordings.values()]
            return positional.TimeWindow(
                start=max(w.start for w in windows),
                stop=min(w.stop for w in windows),
            )

        def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
            subwindow = self.window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
            return type(self.recording)(*[
                recording.sampling.subwindow(subwindow)
                for recording in self.recording.recordings.values()
            ])

    def __init__(self, *recordings):
        self.sampling = self._Sampling(self)
        self.recordings = {
            recording.sensor.sensor.values.item(): recording
            for recording in recordings
        }

    def time_data(self):
        if np.ndim(self.sampling.rate) > 0:
            raise NotImplementedError('Stacking time data from recording with different samplerates not implemented!')
        return xr.concat([recording.time_data() for recording in self.recordings.values()], dim='sensor')

    @property
    def num_channels(self):
        return sum(recording.num_channels for recording in self.recordings.values())

    @property
    def sensor(self):
        return sensors.sensor_array(*[rec.sensor for rec in self.recordings.values()])


class FileRecording(Recording):
    allowable_interrupt = 0

    class RecordedFile(abc.ABC):
        def __init__(self, filepath):
            self.filepath = Path(filepath)

        @property
        def filepath(self):
            return self._filepath

        @filepath.setter
        def filepath(self, filepath):
            if not isinstance(filepath, Path):
                filepath = Path(filepath)
            self._filepath = filepath

        @abc.abstractmethod
        def read_info(self):
            """Reads information about the recorded file.

            Subclasses should implement this reader and store
            the following attributes on the instance.
            - _num_channels: the number of channels in the recording.
            - _samplerate: the number of samples per second per channel.
            - _start_time: The start time of the recording, as a DateTime.
            - _stop_time: The stop time of the recording, as a DateTime.

            If any of the above are not stored in the file, they can either
            be set in the __init__ of the subclass, or the corresponding
            property can be overridden in the subclass.
            """

        @abc.abstractmethod
        def read_data(self, start_idx, stop_idx):
            ...

        @staticmethod
        def _lazy_property(key):
            def getter(self):
                try:
                    return getattr(self, '_' + key)
                except AttributeError:
                    self.read_info()
                return getattr(self, '_' + key)
            return property(getter)

        num_channels = _lazy_property('num_channels')
        samplerate = _lazy_property('samplerate')
        start_time = _lazy_property('start_time')
        stop_time = _lazy_property('stop_time')

        @property
        def duration(self):
            return (self.stop_time - self.start_time).total_seconds()

        def __bool__(self):
            return self.filepath.exists()

    class _Sampling(Recording._Sampling):
        @property
        def rate(self):
            return self.recording.files[0].samplerate

        @property
        def window(self):
            try:
                return self._window
            except AttributeError:
                self._window = positional.TimeWindow(
                    start=self.recording.files[0].start_time,
                    stop=self.recording.files[-1].stop_time,
                )
            return self._window

        def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None, extend=None):
            original_window = self.window
            new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration, extend=extend)
            new = type(self.recording)(
                files=self.recording.files,
                sensor=self.recording.sensor,
            )
            new.sampling._window = new_window
            return new

    def __init__(self, files, assume_sorted=False, **kwargs):
        super().__init__(**kwargs)
        if not assume_sorted:
            files = sorted(files, key=lambda f: f.start_time)
        self.files = files

    @property
    def num_channels(self):
        return self.files[0].num_channels

    def raw_data(self):
        """Read data spread over multiple files."""

        # NOTE: We calculate the sample indices in this "collection" function and not in the file.read_data
        # functions for a reason. In many cases the start and stop times in the file labels are not perfect,
        # but the data is actually written without dropouts or repeats.
        # This means that if we allow each file to calculate it's own indices, we can end up with incorrect number
        # of read samples based only on the sporadic time labels in the files.
        # E.g. say that file 0 has a timestamp 10:00:00 and is 60 minutes and 1.5 seconds long.
        # File 1 would then have the timestamp 11:00:01, but it actually starts at 11:00:01.500.
        # Now, asking for data from 11:00:00 to 11:00:02 we expect 2*samplerate number of samples.
        # File 0 will read 1.5 seconds of data, regardless of where we calculate the sample indices.
        # Calculating the indices in the file-local functions, file 1 wold read 1 second of data.
        # Calculating the indices in the collection function, we would know that we have read 1.5 seconds
        # of data, and ask file 1 for 0.5 seconds of data.
        # This could be remedied if we update the file start times from the file stop time of the previous file,
        # but until such a procedure is implemented upon file gathering, we stick with calculating sample indices here.
        samplerate = self.sampling.rate
        start_time = self.sampling.window.start
        stop_time = self.sampling.window.stop

        samples_to_read = round((stop_time - start_time).total_seconds() * samplerate)
        for file in reversed(self.files):
            if file.start_time <= start_time:
                break
        else:
            raise ValueError(f'Cannot read data starting from {start_time}, earliest file start is {file.start_time}')

        if stop_time <= file.stop_time:
            # The requested data exists within one file.
            # Read the data from file and add it to the signal array.
            start_idx = np.math.floor((start_time - file.start_time).total_seconds() * samplerate)
            stop_idx = start_idx + samples_to_read
            read_signals = file.read_data(start_idx=start_idx, stop_idx=stop_idx)
        else:
            # The requested data spans multiple files
            files_to_read = []
            for file in self.files[self.files.index(file):]:
                files_to_read.append(file)
                if file.stop_time >= stop_time:
                    break
            else:
                raise ValueError(f'Cannot read data extending to {stop_time}, last file ends at {file.stop_time}')

            # Check that the file boundaries are good
            for early, late in zip(files_to_read[:-1], files_to_read[1:]):
                interrupt = (late.start_time - early.stop_time).total_seconds()
                if interrupt > self.allowable_interrupt:
                    raise ValueError(
                        f'Data is not continuous, missing {interrupt} seconds between files '
                        f'ending at {early.stop_time} and starting at {late.start_time}\n'
                        f'{early.filepath}\n{late.filepath}'
                    )

            read_chunks = []

            start_idx = np.math.floor((start_time - files_to_read[0].start_time).total_seconds() * samplerate)
            chunk = files_to_read[0].read_data(start_idx=start_idx)
            read_chunks.append(chunk)
            remaining_samples = samples_to_read - chunk.shape[-1]

            for file in files_to_read[1:-1]:
                chunk = file.read_data()
                read_chunks.append(chunk)
                remaining_samples -= chunk.shape[-1]
            chunk = files_to_read[-1].read_data(stop_idx=remaining_samples)
            read_chunks.append(chunk)
            remaining_samples -= chunk.shape[-1]
            assert remaining_samples == 0

            read_signals = np.concatenate(read_chunks, axis=-1)
        return read_signals

    def select_file_time(self, time):
        time = positional.time_to_datetime(time)
        for file in reversed(self.files):
            if file.start_time > time:
                continue
            if file.stop_time < time:
                raise ValueError(f'Time {time} does not exist inside any recorded files.')
            return self.sampling.subwindow(start=file.start_time, stop=file.stop_time)

    def select_file_name(self, name):
        stem = Path(name).stem
        for file in self.files:
            if stem == file.filepath.stem:
                return self.sampling.subwindow(start=file.start_time, stop=file.stop_time)
        raise ValueError(f"Could not file file matching name '{name}'")


class AudioFileRecording(FileRecording):
    file_range = None
    gain = None
    adc_range = None

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

    class RecordedFile(FileRecording.RecordedFile):
        def __init__(self, filepath, start_time):
            super().__init__(filepath=filepath)
            self._start_time = start_time

        def read_info(self):
            sfi = soundfile.info(self.filepath.as_posix())
            self._stop_time = self.start_time.add(seconds=sfi.duration)
            self._samplerate = sfi.samplerate
            self._num_channels = sfi.channels

        def read_data(self, start_idx=None, stop_idx=None):
            return soundfile.read(self.filepath.as_posix(), start=start_idx, stop=stop_idx, dtype='float32')[0]

    def time_data(self):
        data = self.raw_data()
        if np.ndim(data) == 1:
            dims = 'time'
        elif np.ndim(data) == 2:
            if self.sensor is not None and 'sensor' in self.sensor and np.shape(data)[1] == self.sensor.sensor.size:
                dims = ("time", "sensor")
            else:
                dims = ("time", "channel")
        else:
            raise NotImplementedError('Audio files with more than 2 dimensions are not supported')
        data = time_data(
            data=data,
            samplerate=self.sampling.rate,
            start_time=self.sampling.window.start,
            dims=dims,
        )
        if self.sensor is not None:
            if 'sensor' in self.sensor:
                data.coords['sensor'] = self.sensor.sensor
            elif 'channel' in self.sensor:
                data.coords['channel'] = self.sensor.channel
        data = calibrate_raw_data(
            raw_data=data,
            sensitivity=getattr(self.sensor, 'sensitivity', None),
            gain=self.gain,
            adc_range=self.adc_range,
            file_range=self.file_range
        )
        return data


class SoundTrap(AudioFileRecording):
    allowable_interrupt = 1
    gain = None
    adc_range = None
    file_range = 1

    @classmethod
    def read_folder(cls, folder, sensor=None, serial_number=None, time_compensation=None):
        if serial_number is None:
            def file_filter(filepath):
                return True
        else:
            def file_filter(filepath):
                return int(filepath.stem.split('.')[0]) == serial_number

        def start_time_parser(filepath):
            return pendulum.from_format(filepath.stem.split('.')[1], 'YYMMDDHHmmss')

        return super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
        )


class SylenceLP(AudioFileRecording):
    adc_range = 2.5
    file_range = 1
    allowable_interrupt = 1

    class RecordedFile(AudioFileRecording.RecordedFile):
        def read_info(self):
            with self.filepath.open('rb') as file:
                base_header = file.read(36)
                # chunk_id = base_header[0:4].decode('ascii')  # always equals RIFF
                # file_size = int.from_bytes(base_header[4:8], byteorder='little', signed=False)  # total file size not important
                # chunk_format = base_header[8:12].decode('ascii')  # always equals WAVE
                # subchunk_id = base_header[12:16].decode('ascii')  # always equals fmt
                # subchunk_size = int.from_bytes(base_header[16:20], byteorder='little', signed=False))  # always equals 16
                # audio_format = int.from_bytes(base_header[20:22], byteorder='little', signed=False))  # not important in current implementation
                num_channels = int.from_bytes(base_header[22:24], byteorder='little', signed=False)
                if num_channels != 1:
                    raise ValueError(f"Expected file for SylenceLP with a single channel, read file with {num_channels} channels")
                samplerate = int.from_bytes(base_header[24:28], byteorder='little', signed=False)
                # byte rate = int.from_bytes(base_header[28:32], byteorder='little', signed=False)  # not important in current implementation
                bytes_per_sample = int.from_bytes(base_header[32:34], byteorder='little', signed=False)
                bitdepth = int.from_bytes(base_header[34:36], byteorder='little', signed=False)

                conf_header = file.peek(8)  # uses peak to keep indices aligned with the manual
                conf_size = int.from_bytes(conf_header[4:8], byteorder='little', signed=False)
                if conf_size != 460:
                    raise ValueError(f"Incorrect size of SylenceLP config: '{conf_size}'B, expected 460B")
                conf_header = file.read(conf_size + 8)

                subchunk_id = conf_header[:4].decode('ascii')  # always conf
                if subchunk_id != 'conf':
                    raise ValueError(f"Expected 'conf' section in SylenceLP config, found '{subchunk_id}'")
                # subchunk_size = int.from_bytes(conf_header[4:8], byteorder='little', signed=False)  # the same as conf_size
                config_version = int.from_bytes(conf_header[8:12], byteorder='little', signed=False)
                if config_version != 2:
                    raise NotImplementedError(f'Cannot handle SylenceLP config version {config_version}')
                # recording_start = datetime.datetime.fromtimestamp(int.from_bytes(conf_header[16:24], byteorder='little', signed=True))  # This value is not actually when the recording starts. No idea what it actually is
                channel = conf_header[24:28].decode('ascii')
                if channel.strip('\x00') != '':
                    raise NotImplementedError(f"No implementation for multichannel SylenceLP recorders, found channel specification '{channel}'")
                samplerate_alt = np.frombuffer(conf_header[28:32], dtype='f4').squeeze()
                if samplerate != samplerate_alt:
                    raise ValueError(f"Mismatched samplerate for hardware and file, read file samplerate {samplerate} and config samplerate {samplerate_alt}")

                hydrophone_sensitivity = np.frombuffer(conf_header[32:48], dtype='f4')
                gain = np.frombuffer(conf_header[48:64], dtype='f4')
                # gain_correction = np.frombuffer(conf_header[64:80], dtype='f4')  # is just 1/gain
                serialnumber = conf_header[80:100].decode('ascii')
                active_channels = conf_header[100:104].decode('ascii')
                if active_channels != 'A\x00\x00\x00':
                    raise NotImplementedError(f"No implementation for multichannel SylenceLP recorders, found channel specification '{active_channels}'")

                data_header = file.read(4).decode('ascii')
                if data_header != 'data':
                    raise ValueError(f"Expected file header 'data', read {data_header}")
                data_size = int.from_bytes(file.read(4), byteorder='little', signed=False)

            num_samples = data_size / bytes_per_sample
            if int(num_samples) != num_samples:
                raise ValueError(f"Size of data is not divisible by bytes per sample, file '{self.name}' is corrupt!")

            self._samplerate = samplerate
            self._bitdepth = bitdepth
            # self._start_time = recording_start  # The start property in the file headers is incorrect... It might be the timestamp when the file was created, but in local time instead of UTC? This is useless since the files are pre-created.
            self._stop_time = self.start_time + pendulum.duration(seconds=num_samples / samplerate)
            self._hydrophone_sensitivity = hydrophone_sensitivity[0]
            self._serial_number = serialnumber.strip('\x00')
            self._gain = -20 * np.log10(gain[0])

        bitdepth = FileRecording.RecordedFile._lazy_property('bitdepth')
        hydrophone_sensitivity = FileRecording.RecordedFile._lazy_property('hydrophone_sensitivity')
        serial_number = FileRecording.RecordedFile._lazy_property('serial_number')
        gain = FileRecording.RecordedFile._lazy_property('gain')

    @property
    def gain(self):
        return self.files[0].gain

    @classmethod
    def read_folder(cls, folder, sensor=None, time_compensation=None, file_filter=None):
        def start_time_parser(filepath):
            return pendulum.from_format(filepath.stem[9:], 'YYYY-MM-DD_HH-mm-ss')

        return super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
        )


class MultichannelAudioInterfaceRecording(AudioFileRecording):
    file_range = 1

    @property
    def gain(self):
        return getattr(self.sensor, 'gain', None)

    @property
    def adc_range(self):
        return getattr(self.sensor, 'adc_range', None)

    class RecordedFile(AudioFileRecording.RecordedFile):
        def __init__(self, filepath, start_time, channels):
            super().__init__(filepath=filepath, start_time=start_time)
            self.channels = list(channels)

        def read_data(self, start_idx=None, stop_idx=None):
            all_channels = soundfile.read(
                self.filepath.as_posix(),
                start=start_idx,
                stop=stop_idx,
                dtype="float32",
                always_2d=True,
            )[0]
            return all_channels[:, self.channels]

    @classmethod
    def _merge_channel_info(cls, sensor, channel, gain, adc_range):
        if sensor is None:
            sensor = xr.Dataset()
            if channel is not None:
                if not isinstance(channel, xr.DataArray):
                    channel = xr.DataArray(channel, dims='channel', coords={'channel': channel})
                sensor['channel'] = channel
            if gain is not None:
                if not isinstance(gain, xr.DataArray) and np.ndim(gain) != 0:
                    gain = xr.DataArray(gain, dims='channel', coords={'channel': channel})
                sensor['gain'] = gain
            if adc_range is not None:
                if not isinstance(adc_range, xr.DataArray) and np.ndim(adc_range) != 0:
                    adc_range = xr.DataArray(adc_range, dims='channel', coords={'channel': channel})
                sensor['adc_range'] = adc_range
            return sensor

        assigns = {}
        if "channel" not in sensor:
            if channel is None:
                channel = list(range(sensor.sensor.size))
            assigns["channel"] = sensors.align_property_to_sensors(sensor, channel, allow_scalar=False)
        elif channel is not None:
            raise ValueError(
                "Should not give explicit channel if the channel information is already in the sensor information"
            )

        if "gain" not in sensor:
            if gain is None:
                gain = 0
            assigns["gain"] = sensors.align_property_to_sensors(sensor, gain, allow_scalar=True)
        elif gain is not None:
            raise ValueError(
                "Should not give explicit gain if the gain information is already in the sensor information"
            )

        if "adc_range" not in sensor:
            if adc_range is None:
                adc_range = 1
            assigns["adc_range"] = sensors.align_property_to_sensors(sensor, adc_range, allow_scalar=True)
        elif adc_range is not None:
            raise ValueError(
                "Should not give explicit adc_range if the adc_range information is already in the sensor information"
            )
        return sensor.assign(assigns)

    @classmethod
    def read_folder(
        cls,
        folder,
        start_time_parser,
        channel=None,
        gain=None,
        adc_range=None,
        one_recorder_per_file=False,
        sensor=None,
        file_filter=None,
        time_compensation=None,
        glob_pattern="**/*.wav",
    ):
        sensor = cls._merge_channel_info(sensor=sensor, channel=channel, gain=gain, adc_range=adc_range)
        recordings = super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
            glob_pattern=glob_pattern,
            file_kwargs={"channels": sensor.channel.values},
        )
        if not one_recorder_per_file:
            return recordings
        return [recordings.sampling.subwindow(start=file.start_time, stop=file.stop_time) for file in recordings.files]


class LoggerheadDSG(AudioFileRecording):
    allowable_interrupt = 1
    adc_range = None
    file_range = 1

    @classmethod
    def read_folder(cls, folder, sensor=None, time_compensation=None, file_filter=None):
        def start_time_parser(filepath):
            return pendulum.from_format(filepath.stem[:15], 'YYYYMMDDTHHmmss')

        return super().read_folder(
            folder=folder,
            start_time_parser=start_time_parser,
            sensor=sensor,
            file_filter=file_filter,
            time_compensation=time_compensation,
        )

    @property
    def gain(self):
        return self.files[0].gain

    class RecordedFile(AudioFileRecording.RecordedFile):
        gain = FileRecording.RecordedFile._lazy_property("gain")

        def read_info(self):
            super().read_info()
            gain = self.filepath.stem.split("_")[2]
            if not gain.endswith("dB"):
                raise ValueError(f"File `{self.filepath}` does not seem to be a file from a Loggerhead DSG, could not extract gain")
            self._gain = float(gain.rstrip('dB'))
