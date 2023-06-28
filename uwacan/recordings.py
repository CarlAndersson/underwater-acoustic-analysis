import numpy as np
from . import positional
import os
import re
import abc
import soundfile
import pendulum
import xarray as xr


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
        start = positional._sanitize_datetime_input(self._xr_obj.time.data[0])
        # Calculating duration from number and rate means the stop points to the sample after the last,
        # which is more intuitive when considering signal durations etc.
        return positional.TimeWindow(
            start=start,
            duration=self.num / self.rate,
        )

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        original_window = self.window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration)
        start = (new_window.start - original_window.start).total_seconds()
        stop = (new_window.stop - original_window.start).total_seconds()
        # Indices assumed to be seconds from start
        start = np.math.floor(start * self.rate)
        stop = np.math.ceil(stop * self.rate)

        new_obj = self._xr_obj.isel(time=slice(start, stop))
        return new_obj


class _StampedTimer:
    def __init__(self, xr_obj):
        self._xr_obj = xr_obj

    @property
    def window(self):
        start = positional._sanitize_datetime_input(self._xr_obj.time.data[0])
        stop = positional._sanitize_datetime_input(self._xr_obj.time.data[-1])
        return positional.TimeWindow(start=start, stop=stop)

    def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        original_window = self.window
        new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration)
        start = new_window.start.in_tz('UTC').naive()
        stop = new_window.stop.in_tz('UTC').naive()
        new_obj = self._xr_obj.sel(time=slice(start, stop))
        return new_obj


def _make_sampler(xr_obj):
    if 'time' not in xr_obj.dims:
        raise TypeError(".sampling accessor only available for xarrays with 'time' coordinate")
    if 'rate' in xr_obj.time.attrs:
        return _SampleTimer(xr_obj)
    else:
        return _StampedTimer(xr_obj)


xr.register_dataarray_accessor('sampling')(_make_sampler)
xr.register_dataset_accessor('sampling')(_make_sampler)


def time_data(data, start_time=None, samplerate=None, calibration=None, dims=None):
    if not isinstance(data, xr.DataArray):
        if dims is None:
            if data.ndim == 1:
                dims = 'time'
            else:
                raise ValueError(f'Cannot guess dimensions for time data with {data.ndim} dimensions')
        data = xr.DataArray(data, dims=dims)

    if 'time' not in data.coords:
        if None in (start_time, samplerate):
            raise TypeError('Cannot create structured time data without starting time and samplerate')
        n_samples = data.sizes['time']

        if not isinstance(start_time, np.datetime64):
            start_time = np.datetime64(start_time)
        offsets = np.arange(n_samples) * 1e9 / samplerate
        time = start_time + offsets.astype('timedelta64[ns]')
        data = data.assign_coords(time=('time', time, {'rate': samplerate}))
    elif (start_time is not None) or (samplerate is not None):
        raise TypeError('Should not re-specify start time or samplerate for structured time data.')

    if calibration is not None:
        calibration = np.asarray(calibration).astype('float32')
        c = 10**(calibration / 20) / 1e-6  # Calibration values are given as dB re. 1μPa
        if 'channel' in data.dims:
            shape = [1] * data.ndim
            shape[data.get_axis_num('channel')] = -1
            c.shape = tuple(shape)
        if data.dtype in (np.int8, np.int16, np.int32, np.float32):
            c = c.astype(np.float32)
        data = data / c

    return data


def frequency_data(data, frequency, bandwidth, dims=None):
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
    )
    return data


def time_frequency_data(data, start_time, samplerate, frequency, bandwidth, dims=None):
    if not isinstance(data, xr.DataArray):
        if dims is None:
            raise ValueError('Cannot guess dimensions for time-frequency data')
        data = xr.DataArray(data, dims=dims)
    data = time_data(data, start_time=start_time, samplerate=samplerate)
    data = frequency_data(data, frequency, bandwidth)
    return data


def _read_chunked_files(files, start_time, stop_time, allowable_interrupt=1):
    """Read data spread over multiple files

    Parameters
    ----------
    files : list of RecordedFile or other object with compatible API.
        List of information about the files. Each info item must have the attributes
        `samplerate`, `start_time` and `stop_time`. The list must be ordered in chronological
        order. The files must also have a `read_data(start_idx, stop_idx)` method, which
        reads the file at the designated sample indices, returning a numpy array of shape
        `(ch, stop_idx - start_idx)` or `(stop_idx - start_idx,)`. Omitting either of the
        indices should default to reading from the start of the file and to the end of the
        file respectively.
    start_time : datetime
        The start of the segment to read.
    stop_time : datetime
        The end of the segment to read.
    """

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
    samplerate = files[0].samplerate
    samples_to_read = round((stop_time - start_time).total_seconds() * samplerate)
    for file in reversed(files):
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
        for file in files[files.index(file):]:
            files_to_read.append(file)
            if file.stop_time >= stop_time:
                break
        else:
            raise ValueError(f'Cannot read data extending to {stop_time}, last file ends at {file.stop_time}')

        # Check that the file boundaries are good
        for early, late in zip(files_to_read[:-1], files_to_read[1:]):
            interrupt = (late.start_time - early.stop_time).total_seconds()
            if interrupt > allowable_interrupt:
                raise ValueError(
                    f'Data is not continuous, missing {interrupt} seconds between files '
                    f'ending at {early.stop_time} and starting at {late.start_time}\n'
                    f'{early.name}\n{late.name}'
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


def _read_chunked_files(files, start_time, stop_time, allowable_interrupt=1):
    """Read data spread over multiple files

    Parameters
    ----------
    files : list of RecordedFile or other object with compatible API.
        List of information about the files. Each info item must have the attributes
        `samplerate`, `start_time` and `stop_time`. The list must be ordered in chronological
        order. The files must also have a `read_data(start_idx, stop_idx)` method, which
        reads the file at the designated sample indices, returning a numpy array of shape
        `(ch, stop_idx - start_idx)` or `(stop_idx - start_idx,)`. Omitting either of the
        indices should default to reading from the start of the file and to the end of the
        file respectively.
    start_time : datetime
        The start of the segment to read.
    stop_time : datetime
        The end of the segment to read.
    """

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
    samplerate = files[0].samplerate
    samples_to_read = round((stop_time - start_time).total_seconds() * samplerate)
    for file in reversed(files):
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
        for file in files[files.index(file):]:
            files_to_read.append(file)
            if file.stop_time >= stop_time:
                break
        else:
            raise ValueError(f'Cannot read data extending to {stop_time}, last file ends at {file.stop_time}')

        # Check that the file boundaries are good
        for early, late in zip(files_to_read[:-1], files_to_read[1:]):
            interrupt = (late.start_time - early.stop_time).total_seconds()
            if interrupt > allowable_interrupt:
                raise ValueError(
                    f'Data is not continuous, missing {interrupt} seconds between files '
                    f'ending at {early.stop_time} and starting at {late.start_time}\n'
                    f'{early.name}\n{late.name}'
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


class RecordTimeCompensation:
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

        actual_time = list(map(positional._sanitize_datetime_input, actual_time))
        recorded_time = list(map(positional._sanitize_datetime_input, recorded_time))

        self._time_offset = [(recorded - actual).total_seconds() for (recorded, actual) in zip(recorded_time, actual_time)]
        if len(self._time_offset) > 1:
            self._actual_timestamps = [t.timestamp() for t in actual_time]
            self._recorded_timestamps = [t.timestamp() for t in recorded_time]

    def recorded_to_actual(self, recorded_time):
        recorded_time = positional._sanitize_datetime_input(recorded_time)
        if len(self._time_offset) == 1:
            time_offset = self._time_offset[0]
        else:
            time_offset = np.interp(recorded_time.timestamp(), self._recorded_timestamps, self._time_offset)
        return recorded_time - pendulum.duration(seconds=time_offset)

    def actual_to_recorded(self, actual_time):
        actual_time = positional._sanitize_datetime_input(actual_time)
        if len(self._time_offset) == 1:
            time_offset = self._time_offset[0]
        else:
            time_offset = np.interp(actual_time.timestamp(), self._actual_timestamps, self._time_offset)
        return actual_time + pendulum.duration(seconds=time_offset)


class RecordedFile(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def read_info(self):
        ...

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

    @property
    @abc.abstractmethod
    def samplerate(self):
        ...

    @property
    @abc.abstractmethod
    def start_time(self):
        ...

    @property
    @abc.abstractmethod
    def stop_time(self):
        ...

    @property
    def duration(self):
        return (self.stop_time - self.start_time).total_seconds()

    def __bool__(self):
        return os.path.exists(self.name)


class HydrophoneArray:
    class _Sampling:
        def __init__(self, hydrophone):
            # TODO: remove this after reorganization so that this internal class can subclass Hydrophone._Sampling
            self.hydrophone = hydrophone

        @property
        def rate(self):
            return self.hydrophone.hydrophones[0].sampling.rate

        @property
        def window(self):
            return self.hydrophone.hydrophones[0].sampling.window

        def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
            return type(self.hydrophone)(*[
                hphn.sampling.subwindow(time, start=start, stop=stop, center=center, duration=duration)
                for hphn in self.hydrophone.hydrophones
            ])

    def __init__(self, *hydrophones):
        self.hydrophones = hydrophones
        self.sampling = self._Sampling(self)
        # TODO: run some checks:
        # - restrict the time periods of the hydrophones
        # - ensure that all samplerates are the same

    @property
    def num_channels(self):
        return sum(hphn.num_channels for hphn in self.hydrophones)

    @property
    def depths(self):
        return [hphn.depth for hphn in self.hydrophones]

    @property
    def positions(self):
        return [hphn.position for hphn in self.hydrophones]

    @property
    def time_data(self):
        return xr.concat([hphn.time_data for hphn in self.hydrophones], dim='receiver')


class ColocatedHydrophoneArray(HydrophoneArray):
    @property
    def position(self):
        return self.hydrophones[0].position


class Hydrophone:
    class _Sampling(abc.ABC):
        def __init__(self, hydrophone):
            self.hydrophone = hydrophone

        @property
        @abc.abstractmethod
        def rate(self):
            ...

        @property
        @abc.abstractmethod
        def window(self):
            ...

        @abc.abstractmethod
        def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
            ...

    def __init__(
        self,
        depth,
        position,
    ):
        self.sampling = self._Sampling(self)
        self.depth = depth
        self.position = positional.Position(position)

    @property
    def num_channels(self):
        return 1


class SoundTrap(Hydrophone):
    #  The file starts are taken from the timestamps in the filename, which is quantized to 1s.
    allowable_interrupt = 1

    class _Sampling(Hydrophone._Sampling):
        def __init__(self, hydrophone):
            super().__init__(hydrophone)

        @property
        def rate(self):
            return self.hydrophone.files[0].samplerate

        @property
        def window(self):
            try:
                return self._window
            except AttributeError:
                self._window = positional.TimeWindow(
                    start=self.hydrophone.files[0].start_time,
                    stop=self.hydrophone.files[-1].stop_time,
                )
            return self._window

        def subwindow(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
            original_window = self.window
            new_window = original_window.subwindow(time, start=start, stop=stop, center=center, duration=duration)
            new = type(self.hydrophone)(
                folder=self.hydrophone.folder,
                serial_number=self.hydrophone.serial_number,
                calibration=self.hydrophone.calibration,
                depth=self.hydrophone.depth,
                position=self.hydrophone.position,
                _files=self.hydrophone.files
            )
            new.sampling._window = new_window
            return new

    class RecordedFile(RecordedFile):
        pattern = r'(\d{4})\.(\d{12}).wav'
        def __init__(self, name, time_compensation):
            super().__init__(name=name)
            name = os.path.basename(name)
            if not (match := re.match(self.pattern, name)):
                return
            serial_number, time = match.groups()
            self.serial_number = int(serial_number)
            self._start_time = time_compensation(pendulum.from_format(time, 'YYMMDDHHmmss'))

        def __bool__(self):
            return super().__bool__() and hasattr(self, 'serial_number') and bool(self.serial_number)

        def read_info(self):
            sfi = soundfile.info(self.name)
            self._stop_time = self.start_time + pendulum.duration(seconds=sfi.duration)
            self._samplerate = sfi.samplerate

        start_time = RecordedFile._lazy_property('start_time')
        stop_time = RecordedFile._lazy_property('stop_time')
        samplerate = RecordedFile._lazy_property('samplerate')

        def read_data(self, start_idx=None, stop_idx=None):
            return soundfile.read(self.name, start=start_idx, stop=stop_idx, dtype='float32')[0]

    def __init__(self, folder, serial_number, calibration, time_compensation=None, _files=None, **kwargs):
        """Read a folder with SoundTrap data.

        Parameters
        ----------
        folder : str
            Path to the folder with the data.
        key : int, str
            The serial number of the Hydrophone.
            Can be given as an integer or a string.
        calibrations : dict or numeric
            A dict with the calibration values of the SoundTraps.
            If a single value is given, it will be used for all read data.
            Give as a value in dB re. 1/μPa, e.g. -188.5
        depth : dict or numeric
            A dict with the depths of the SoundTraps, in meters.
            If a single value is given, it will be used for all read data.
        time_offset : {numeric, callable}, optional
            Time offset that will be added to the timestamps stored in files.
            For custom offsets, pass a function with the signature
                `offset = time_offset(timestamp, serial_number)`
            that returns the offset for the file timestamp and particular serial number.
        """
        super().__init__(**kwargs)
        self.calibration = calibration
        self.folder = folder
        self.serial_number = serial_number

        if _files is None:
            self.files = []
            if time_compensation is None:
                def time_compensation(timestamp):
                    return timestamp
            if isinstance(time_compensation, RecordTimeCompensation):
                time_compensation = time_compensation.recorded_to_actual
            elif not callable(time_compensation):
                offset = pendulum.duration(seconds=time_compensation)
                def time_compensation(timestamp):
                    return timestamp - offset

            for file in sorted(filter(lambda x: x.is_file(), os.scandir(self.folder)), key=lambda x: x.name):
                file = self.RecordedFile(name=os.path.join(self.folder, file), time_compensation=time_compensation)
                if file and (file.serial_number == self.serial_number):
                    self.files.append(file)
        else:
            self.files = _files

    @property
    def time_data(self):
        read_signals = _read_chunked_files(
            files=self.files,
            start_time=self.sampling.window.start,
            stop_time=self.sampling.window.stop,
            allowable_interrupt=self.allowable_interrupt,
        )
        return time_data(
            data=read_signals,
            calibration=self.calibration,
            samplerate=self.sampling.rate,
            start_time=self.sampling.window.start,
        )


class SylenceLP(Hydrophone):
    allowable_interrupt = 1
    voltage_range = 2.5

    class RecordedFile(RecordedFile):
        patten = r"channel([A-D])_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2}).wav"
        def __init__(self, name, time_compensation):
            super().__init__(name)
            if not super().__bool__():
                return

            basename = os.path.basename(self.name)
            if not (match := re.match(self.patten, basename)):
                return
            channel, year, month, day, hour, minute, second = match.groups()
            self._start_time = time_compensation(pendulum.datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second),
            ))

        def __bool__(self):
            return super().__bool__() and hasattr(self, '_start_time')

        def read_data(self, start_idx=None, stop_idx=None):
            return soundfile.read(self.name, start=start_idx, stop=stop_idx, dtype='float32')[0]

        def read_info(self):
            with open(self.name, 'rb') as file:
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
            self._stop_time = self.start_time + pendulum.duraion(seconds=num_samples / samplerate)
            self._hydrophone_sensitivity = hydrophone_sensitivity[0]
            self._serial_number = serialnumber.strip('\x00')
            self._gain = 20 * np.log10(gain[0])

        samplerate = RecordedFile._lazy_property('samplerate')
        bitdepth = RecordedFile._lazy_property('bitdepth')
        start_time = RecordedFile._lazy_property('start_time')
        stop_time = RecordedFile._lazy_property('stop_time')
        hydrophone_sensitivity = RecordedFile._lazy_property('hydrophone_sensitivity')
        serial_number = RecordedFile._lazy_property('serial_number')
        gain = RecordedFile._lazy_property('gain')

    def __init__(self, folder, time_compensation=None, _files=None, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder
        if _files is None:
            self.files = []

            if time_compensation is None:
                def time_compensation(timestamp):
                    return timestamp
            elif isinstance(time_compensation, RecordTimeCompensation):
                time_compensation = time_compensation.recorded_to_actual
            elif not callable(time_compensation):
                offset = pendulum.duration(seconds=time_compensation)
                def time_compensation(timestamp):
                    return timestamp - offset

            for directory in sorted(filter(lambda x: x.is_dir(), os.scandir(self.folder)), key=lambda x: x.name):
                for file in sorted(filter(lambda x: x.is_file(), os.scandir(directory.path)), key=lambda x: x.name):
                    if file := self.RecordedFile(file.path, time_compensation=time_compensation):
                        self.files.append(file)
        else:
            self.files = _files

        start_time = self.files[0].start_time
        stop_time = self.files[-1].stop_time
        self.time_period = positional.TimeWindow(start=start_time, stop=stop_time)

    @property
    def samplerate(self):
        return self.files[0].samplerate

    @property
    def calibration(self):
        hydrophone_sensitivity = self.files[0].hydrophone_sensitivity
        gain = self.files[0].gain
        return hydrophone_sensitivity + gain - 20 * np.log10(self.voltage_range)

    @property
    def time_data(self):
        read_signals = _read_chunked_files(
            files=self.files,
            start_time=self.time_period.start,
            stop_time=self.time_period.stop,
            allowable_interrupt=self.allowable_interrupt,
        )
        return time_data(
            data=read_signals,
            calibration=self.calibration,
            samplerate=self.samplerate,
            start_time=self.time_period.start,
        )
