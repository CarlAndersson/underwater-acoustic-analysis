"""Storage and reading of recorded acoustic signals."""
import pendulum
import numpy as np
from . import positional, signals, _core
import os
import re
import soundfile
import abc


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


class Recording(_core.Leaf):
    # def __init__(self, key=None):
        # self.key = key

    # @property
    # def identifiers(self):
    #     ids = self._identifiers
    #     if len(ids) == 0:
    #         return None
    #     if len(ids) == 1:
    #         return ids[0]
    #     return ids

    # @identifiers.setter
    # def identifiers(self, identifiers):
    #     if identifiers is None or isinstance(identifiers, str):
    #         identifiers = [identifiers]
    #     else:
    #         try:
    #             iter(identifiers)
    #         except TypeError as err:
    #             if str(err).endswith('object is not iterable'):
    #                 identifiers = [identifiers]
    #             else:
    #                 raise
    #             identifiers = list(identifiers)
    #     self._identifiers = identifiers

    # class _mapped_property:
    #     class SingleDataMapper(dict):
    #         def __init__(self, data, **kwargs):
    #             super().__init__(**kwargs)
    #             self.data = data

    #         def __getitem__(self, key):
    #             try:
    #                 return super().__getitem__(key)
    #             except KeyError:
    #                 return self.data

        # def __init__(self, preprocessor):
        #     self.name = preprocessor.__name__
        #     self.preprocessor = preprocessor

        # def __get__(self, owner, owner_class=None):
        #     data = getattr(owner, '_' + self.name)
        #     if isinstance(data, self.SingleDataMapper):
        #         return data.data
        #     return data

        # def __set__(self, owner, data):
        #     try:
        #         # Multiple data
        #         data = {key: self.preprocessor(owner, val) for key, val in data.items()}
        #     except AttributeError as err:
        #         if str(err).endswith("object has no attribute 'items'"):
        #             data = self.SingleDataMapper(self.preprocessor(owner, data))
        #         else:
        #             raise

        #     setattr(owner, '_' + self.name, data)

    # def time_range(self, start=None, stop=None, center=None, duration=None):
    #     """Restrict the time range.

    #     This gets a window to the same time signal over a specified time period.
    #     The time period can be specified with any combination of two of the input
    #     parameters. The times can be specified either as `datetime` objects,
    #     or as strings on following format: `YYYYMMDDhhmmssffffff`.
    #     The microsecond part can be optionally omitted, and any non-digit characters
    #     are removed. Some examples include
    #     - 20220525165717
    #     - 2022-05-25_16-57-17
    #     - 2022/05/25 16:57:17.123456

    #     Parameters
    #     ----------
    #     start : datetime or string
    #         The start of the time window
    #     stop : datetime or string
    #         The end of the time window
    #     center : datetime or string
    #         The center of the time window
    #     duration : numeric
    #         The total duration of the time window, in seconds.
    #     """
    #     time_window = positional.TimeWindow(
    #         start=start,
    #         stop=stop,
    #         center=center,
    #         duration=duration,
    #     )
    #     return self[time_window]

    # def copy(self):
    #     obj = type(self).__new__(type(self))
    #     # obj._identifiers = self._identifiers
    #     obj.key = self.key
    #     return obj

    @abc.abstractmethod
    def __getitem__(self, window):
        """Restrict the time range.

        This gets the same signal but restricted to a time range specified
        with a TimeWindow object.

        Parameters
        ----------
        window : `timestamps.TimeWindow`
            The time window to restrict to.
        """
        ...

    # @property
    # @abc.abstractmethod
    # def signal(self):
    #     """Get the actual time signal in the appropriate time window."""
    #     ...

    # @property
    # def _leaves(self):
    #     yield self


class Hydrophone(Recording):
    def __init__(
        self,
        channel=None,
        position=None,
        depth=None,
        **kwargs
    ):
        metadata = {'channel': channel}
        if position is not None:
            metadata['hydrophone position'] = positional.Position(position)
        if depth is not None:
            metadata['hydrophone depth'] = depth
        # metadata = {'position': position, 'depth': depth, 'calibration': calibration}
        super().__init__(**kwargs, metadata=metadata)
        # self.position = position
        # self.depth = depth
        # self.calibration = calibration

    # def copy(self, **kwargs):
    #     obj = super().copy(**kwargs)
    #     obj._position = self._position
    #     obj._depth = self._depth
    #     obj._calibration = self._calibration
    #     return obj
    def copy(self, **kwargs):
        obj = super().copy(**kwargs)
        return obj

    # @property
    # def depth(self):
    #     return self._depth

    # @depth.setter
    # def depth(self, value):
    #     self._depth = value

    # @property
    # def calibration(self):
    #     return self._calibration

    # @calibration.setter
    # def calibration(self, value):
    #     self._calibration = value

    # @property
    # def position(self):
    #     return self._position

    # @position.setter
    # def position(self, value):
    #     if not isinstance(value, positional.Position):
    #         try:
    #             value = positional.Position(**value)
    #         except TypeError as err:
    #             if 'argument after ** must be a mapping' not in str(err):
    #                 raise
    #             else:
    #                 value = positional.Position(*value)
    #     self._position = value

    # @Recording._mapped_property
    # def depth(self, value):
    #     return float(value) if value is not None else None

    # @Recording._mapped_property
    # def calibration(self, value):
    #     return float(value) if value is not None else None

    # @Recording._mapped_property
    # def position(self, value):
    #     if isinstance(value, positional.Position):
    #         return value
    #     elif value is not None:
    #         try:
    #             longitude, latitude = value
    #         except ValueError:
    #             raise ValueError(f'Cannot unpack position data {value} into longitude and latitude!')
    #         return positional.Position(longitude=longitude, latitude=latitude)


class HydrophoneArray(_core.Branch):
    def __init__(self, *hydrophones, position=None, depth=None):
        metadata = {}
        if position is not None:
            metadata['hydrophone position'] = positional.Position(position)
        if depth is not None:
            metadata['hydrophone depth'] = depth
        super().__init__(*hydrophones, _layer='channel', metadata=metadata)
        # self.hydrophones = hydrophones

    @property
    def hydrophones(self):
        return self._children

    # def __getitem__(self, window):
        # return signals.ArraySignal(*(hydrophone[window] for hydrophone in self.hydrophones), _layer='channels')

    # def __getitem__(self, window):
        # return type(self)(*[hydrophone[window] for hydrophone in self.hydrophones])

    @property
    def data(self):
        return signals.DataStack(
            *(hydrophone.data for hydrophone in self.hydrophones),
            _layer=self._layer,
            metadata=self.metadata.data,
        )


class SoundTrap(Hydrophone):
    # @Recording._mapped_property
    # def files(self, list_of_files):
    #     if list_of_files is None or len(list_of_files) == 0:
    #         list_of_files = []
    #     return list_of_files

    #  The file starts are taken from the timestamps in the filename, which is quantized to 1s.
    allowable_interrupt = 1

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

    def __init__(self, folder, serial_number, time_compensation=None, calibration=None, **kwargs):
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
        super().__init__(channel=serial_number, **kwargs)
        # self.timezone = timezone
        self.calibration = calibration
        # tz = positional.dateutil.tz.gettz(self.timezone)
        # serial_numbers = self.identifiers
        # if serial_numbers is None:
        #     pattern = r'\d{4}'
        # elif isinstance(serial_numbers, str):
        #     pattern = serial_numbers
        # else:
        #     try:
        #         iter(serial_numbers)
        #     except TypeError as err:
        #         if str(err).endswith('object is not iterable'):
        #             serial_numbers = [serial_numbers]
        #         else:
        #             raise
        #     pattern = '|'.join(map(str, serial_numbers))

        self.folder = folder

        if time_compensation is None:
            def time_compensation(timestamp):
                return timestamp
        elif not callable(time_compensation):
            offset = pendulum.duration(seconds=time_compensation)
            def time_compensation(timestamp):
                return timestamp - offset
        # elif isinstance(time_offset, dict):
        #     # TODO: move this logic to the soundtraparray class
        #     time_offset_dict = time_offset
        #     def time_offset(timestamp, serial_number):
        #         try:
        #             return time_offset_dict[serial_number]
        #         except KeyError:
        #             pass
        #         try:
        #             return time_offset_dict[serial_number + '_' + timestamp]
        #         except KeyError:
        #             pass
        #         try:
        #             return time_offset_dict[(serial_number, timestamp)]
        #         except KeyError:
        #             pass
        #         if int(serial_number) != serial_number:
        #             return time_offset(int(serial_number), timestamp)
        #         raise KeyError(f'Could not find time offset for serial number {serial_number} and timestamp {timestamp}')

        # self._identifiers = []
        # files = {}
        self.files = []

                # if info.samplerate != self.samplerate:
                #     raise ValueError('Cannot handle multiple samplerates in one soundtrap object')
        for file in sorted(filter(lambda x: x.is_file(), os.scandir(self.folder)), key=lambda x: x.name):
            file = self.RecordedFile(name=os.path.join(self.folder, file), time_compensation=time_compensation)
            if file and (file.serial_number == self.serial_number):
                self.files.append(file)
        start_time = self.files[0].start_time
        stop_time = self.files[-1].stop_time
        self.time_window = self._raw_time_window = positional.TimeWindow(start=start_time, stop=stop_time)

    @property
    def samplerate(self):
        return self.files[0].samplerate

    @property
    def serial_number(self):
        return int(self.metadata['channel'])

    def copy(self, **kwargs):
        obj = super().copy(**kwargs)
        obj.folder = self.folder
        obj.files = self.files
        obj.calibration = self.calibration
        obj.time_window = self.time_window
        obj._raw_time_window = self._raw_time_window
        return obj

    # def __getitem__(self, selected_time_window):
    #     if selected_time_window.start < self._raw_time_window.start:
    #         raise ValueError(f'Cannot select data starting at {selected_time_window.start} from recording starting at {self._raw_time_window.start}')
    #     if selected_time_window.stop > self._raw_time_window.stop:
    #         raise ValueError(f'Cannot select data until {selected_time_window.stop} from recording ending at {self._raw_time_window.stop}')

    #     samples_to_read = round((selected_time_window.stop - selected_time_window.start).total_seconds() * self.samplerate)
    #     for info in reversed(self.files):
    #         if info.start_time <= selected_time_window.start:
    #             break
    #     else:
    #         raise ValueError(f'Cannot read data starting from {selected_time_window.start}, earliest file start is {info.start_time}')

    #     if selected_time_window.stop <= info.stop_time:
    #         # The requested data exists within one file.
    #         # Read the data from file and add it to the signal array.
    #         start_idx = np.math.floor((selected_time_window.start - info.start_time).total_seconds() * self.samplerate)
    #         stop_idx = start_idx + samples_to_read
    #         read_signals = soundfile.read(info.name, start=start_idx, stop=stop_idx, dtype='float32')[0]

    #     else:

    #         # The requested data spans multiple files
    #         files_to_read = []
    #         for info in self.files[self.files.index(info):]:
    #             files_to_read.append(info)
    #             if info.stop_time >= selected_time_window.stop:
    #                 break
    #         else:
    #             raise ValueError(f'Cannot read data extending to {selected_time_window.stop}, last file ends at {info.stop_time}')

    #         # Check that the file boundaries are good
    #         for early, late in zip(files_to_read[:-1], files_to_read[1:]):
    #             interrupt = (late.start_time - early.stop_time).total_seconds()
    #             if interrupt > self.allowable_interrupt:
    #                 raise ValueError(
    #                     f'Data is not continuous, missing {interrupt} seconds between files '
    #                     f'ending at {early.stop_time} and starting at {late.start_time}\n'
    #                     f'{early.name}\n{late.name}'
    #                 )

    #         read_chunks = []

    #         start_idx = np.math.floor((selected_time_window.start - files_to_read[0].start_time).total_seconds() * self.samplerate)
    #         chunk = soundfile.read(files_to_read[0].name, start=start_idx, dtype='float32')[0]
    #         read_chunks.append(chunk)
    #         remaining_samples = samples_to_read - chunk.size
    #         for file in files_to_read[1:-1]:
    #             chunk = soundfile.read(file.name, dtype='float32')[0]
    #             read_chunks.append(chunk)
    #             remaining_samples -= chunk.size
    #         chunk = soundfile.read(files_to_read[-1].name, stop=remaining_samples, dtype='float32')[0]
    #         read_chunks.append(chunk)
    #         remaining_samples -= chunk.size
    #         assert remaining_samples == 0

    #         read_signals = np.concatenate(read_chunks, axis=0)

    #     if self.calibration is None:
    #         signal = signals.Time(
    #             data=read_signals,
    #             _name=self._name,
    #             samplerate=self.samplerate,
    #             start_time=selected_time_window.start
    #         )
    #     else:
    #         signal = signals.Pressure.from_raw_and_calibration(
    #             data=read_signals,
    #             calibration=self.calibration,
    #             _name=self._name,
    #             samplerate=self.samplerate,
    #             start_time=selected_time_window.start,
    #         )
    #     signal.depth = self.depth
    #     signal.position = self.position
    #     return signal
    # def copy(self, deep=False):
    #     obj = super().copy(deep=deep)
    #     obj.folder = self.folder
    #     obj.files = self.files
    #     obj.time_window = self.time_window
    #     obj._raw_time_window = self._raw_time_window
    #     return obj

    def __getitem__(self, window):
        if window.start < self._raw_time_window.start:
            raise ValueError(f'Cannot select data starting at {window.start} from recording starting at {self._raw_time_window.start}')
        if window.stop > self._raw_time_window.stop:
            raise ValueError(f'Cannot select data until {window.stop} from recording ending at {self._raw_time_window.stop}')
        obj = self.copy()
        obj.time_window = window
        return obj

    @property
    def data(self):
        read_signals = _read_chunked_files(
            files=self.files,
            start_time=self.time_window.start,
            stop_time=self.time_window.stop,
            allowable_interrupt=self.allowable_interrupt,
        )




        if self.calibration is None:
            signal = signals.Time(
                data=read_signals,
                samplerate=self.samplerate,
                start_time=self.time_window.start,
                metadata=self.metadata.data
            )
        else:
            signal = signals.Pressure.from_raw_and_calibration(
                data=read_signals,
                calibration=self.calibration,
                samplerate=self.samplerate,
                start_time=self.time_window.start,
                metadata=self.metadata.data
            )
        # signal.depth = self.depth
        # signal.position = self.position
        return signal

        # read_signals = np.stack([read_signals[sn] for sn in self._identifiers], axis=0).squeeze()
        # read_signals = np.atleast_2d(read_signals)
        # calibrations = [self.calibration[sn] for sn in self._identifiers]
        # if None not in calibrations:
        #     return signals.Pressure.from_raw_and_calibration(
        #         read_signals,
        #         calibrations,
        #         samplerate=self.samplerate,
        #         start_time=self.time_window.start,
        #     )
        # else:
        #     return signals.Signal(
        #         read_signals,
        #         samplerate=self.samplerate,
        #         start_time=self.time_window.start
        #     )


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

    def __init__(self, folder, time_compensation=None, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder
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

        start_time = self.files[0].start_time
        stop_time = self.files[-1].stop_time
        self.time_window = self._raw_time_window = positional.TimeWindow(start=start_time, stop=stop_time)

    @property
    def samplerate(self):
        return self.files[0].samplerate

    @property
    def serial_number(self):
        return self.files[0].serial_number

    @property
    def calibration(self):
        hydrophone_sensitivity = self.files[0].hydrophone_sensitivity
        gain = self.files[0].gain
        return hydrophone_sensitivity + gain - 20 * np.log10(self.voltage_range)

    def __getitem__(self, window):
        if window.start < self._raw_time_window.start:
            raise ValueError(f'Cannot select data starting at {window.start} from recording starting at {self._raw_time_window.start}')
        if window.stop > self._raw_time_window.stop:
            raise ValueError(f'Cannot select data until {window.stop} from recording ending at {self._raw_time_window.stop}')
        obj = self.copy()
        obj.time_window = window
        return obj

    def copy(self, **kwargs):
        obj = super().copy(**kwargs)
        obj.folder = self.folder
        obj.files = self.files
        obj.time_window = self.time_window
        obj._raw_time_window = self._raw_time_window
        return obj

    @property
    def data(self):
        read_signals = _read_chunked_files(
            files=self.files,
            start_time=self.time_window.start,
            stop_time=self.time_window.stop,
            allowable_interrupt=self.allowable_interrupt,
        )

        signal = signals.Pressure.from_raw_and_calibration(
            data=read_signals,
            calibration=self.calibration,
            samplerate=self.samplerate,
            start_time=self.time_window.start,
            metadata=self.metadata.data
        )
        return signal
