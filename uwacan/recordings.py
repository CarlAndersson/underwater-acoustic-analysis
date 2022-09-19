"""Storage and reading of recorded acoustic signals."""
import datetime
import numpy as np
from . import positional, signals
import os
import re
import soundfile
import abc


class Recording(abc.ABC):
    def __init__(self, identifiers=None):
        self.identifiers = identifiers

    @property
    def identifiers(self):
        ids = self._identifiers
        if len(ids) == 0:
            return None
        if len(ids) == 1:
            return ids[0]
        return ids

    @identifiers.setter
    def identifiers(self, identifiers):
        if identifiers is None or isinstance(identifiers, str):
            identifiers = [identifiers]
        else:
            try:
                iter(identifiers)
            except TypeError as err:
                if str(err).endswith('object is not iterable'):
                    identifiers = [identifiers]
                else:
                    raise
                identifiers = list(identifiers)
        self._identifiers = identifiers

    class _mapped_property:
        class SingleDataMapper(dict):
            def __init__(self, data, **kwargs):
                super().__init__(**kwargs)
                self.data = data

            def __getitem__(self, key):
                try:
                    return super().__getitem__(key)
                except KeyError:
                    return self.data

        def __init__(self, preprocessor):
            self.name = preprocessor.__name__
            self.preprocessor = preprocessor

        def __get__(self, owner, owner_class=None):
            data = getattr(owner, '_' + self.name)
            if isinstance(data, self.SingleDataMapper):
                return data.data
            return data

        def __set__(self, owner, data):
            try:
                # Multiple data
                data = {key: self.preprocessor(owner, val) for key, val in data.items()}
            except AttributeError as err:
                if str(err).endswith("object has no attribute 'items'"):
                    data = self.SingleDataMapper(self.preprocessor(owner, data))
                else:
                    raise

            setattr(owner, '_' + self.name, data)

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
        time_window = positional.TimeWindow(
            start=start,
            stop=stop,
            center=center,
            duration=duration,
        )
        return self[time_window]

    def copy(self, deep=False):
        obj = type(self).__new__(type(self))
        obj._identifiers = self._identifiers
        return obj

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

    @property
    @abc.abstractmethod
    def signal(self):
        """Get the actual time signal in the appropriate time window."""
        ...


class Hydrophone(Recording):
    def __init__(
        self,
        position=None,
        depth=None,
        calibration=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.position = position
        self.depth = depth
        self.calibration = calibration

    def copy(self, deep=False):
        obj = super().copy(deep=deep)
        obj._position = self._position
        obj._depth = self._depth
        obj._calibration = self._calibration
        return obj

    @Recording._mapped_property
    def depth(self, value):
        return float(value) if value is not None else None

    @Recording._mapped_property
    def calibration(self, value):
        return float(value) if value is not None else None

    @Recording._mapped_property
    def position(self, value):
        if isinstance(value, positional.Position):
            return value
        elif value is not None:
            try:
                longitude, latitude = value
            except ValueError:
                raise ValueError(f'Cannot unpack position data {value} into longitude and latitude!')
            return positional.Position(longitude=longitude, latitude=latitude)


class SoundTrap(Hydrophone):
    @Recording._mapped_property
    def files(self, list_of_files):
        if list_of_files is None or len(list_of_files) == 0:
            list_of_files = []
        return list_of_files

    #  The file starts are taken from the timestamps in the filename, which is quantized to 1s.
    allowable_interrupt = 1

    def __init__(self, folder, timezone='UTC', **kwargs):
        """Read a folder with SoundTrap data.

        Parameters
        ----------
        folder : str
            Path to the folder with the data.
        calibrations : dict or numeric
            A dict with the calibration values of the SoundTraps.
            If a single value is given, it will be used for all read data.
            Give as a value in dB re. 1/μPa, e.g. -188.5
        depth : dict or numeric
            A dict with the depths of the SoundTraps, in meters.
            If a single value is given, it will be used for all read data.
        identifiers : int, str, list
            The serial numbers of the Hydrophones, optional filter for only reading a subset of the data.
            Can be given as an integer, a string, or as a list of ints or strings.
        """
        super().__init__(**kwargs)
        self.timezone = timezone
        tz = positional.dateutil.tz.gettz(self.timezone)
        serial_numbers = self.identifiers
        if serial_numbers is None:
            pattern = r'\d{4}'
        elif isinstance(serial_numbers, str):
            pattern = serial_numbers
        else:
            try:
                iter(serial_numbers)
            except TypeError as err:
                if str(err).endswith('object is not iterable'):
                    serial_numbers = [serial_numbers]
                else:
                    raise
            pattern = '|'.join(map(str, serial_numbers))

        self.folder = folder
        pattern = '(' + pattern + r')\.(\d{12}).wav'

        self._identifiers = []
        files = {}

        for file in sorted(os.listdir(self.folder)):
            if match := re.match(pattern, file):
                sn, time = match.groups()
                sn = int(sn)
                if sn not in self._identifiers:
                    self._identifiers.append(sn)
                    files[sn] = []
                info = soundfile.info(os.path.join(self.folder, file))
                info.start_time = datetime.datetime.strptime(time, r'%y%m%d%H%M%S').replace(tzinfo=tz)
                info.stop_time = info.start_time + datetime.timedelta(seconds=info.duration)
                files[sn].append(info)
                # if info.samplerate != self.samplerate:
                #     raise ValueError('Cannot handle multiple samplerates in one soundtrap object')
        start_time = max([fls[0].start_time for fls in files.values()])
        stop_time = min([fls[-1].stop_time for fls in files.values()])
        if len(files) == 1:
            files, = files.values()
        self.files = files
        self.time_window = self._raw_time_window = positional.TimeWindow(start=start_time, stop=stop_time)

    @property
    def samplerate(self):
        return self._files[self._identifiers[0]][0].samplerate

    def copy(self, deep=False):
        obj = super().copy(deep=deep)
        obj.folder = self.folder
        obj._files = self._files
        obj.time_window = self.time_window
        obj._raw_time_window = self._raw_time_window
        return obj

    def __getitem__(self, window):
        if window.start < self._raw_time_window.start:
            raise ValueError(f'Cannot select data starting at {window.start} from recording starting at {self._raw_time_window.start}')
        if window.stop > self._raw_time_window.stop:
            raise ValueError(f'Cannot select data until {window.stop} from recording ending at {self._raw_time_window.stop}')
        obj = self.copy()
        obj.time_window = window
        return obj

    @property
    def signal(self):
        read_signals = {}
        samples_to_read = round((self.time_window.stop - self.time_window.start).total_seconds() * self.samplerate)
        for sn in self._identifiers:
            files = self._files[sn]
            for info in reversed(files):
                if info.start_time <= self.time_window.start:
                    break
            else:
                raise ValueError(f'Cannot read data starting from {self.time_window.start}, earliest file start is {info.start_time}')

            if self.time_window.stop <= info.stop_time:
                # The requested data exists within one file.
                # Read the data from file and add it to the signal array.
                ...
                start_idx = np.math.floor((self.time_window.start - info.start_time).total_seconds() * self.samplerate)
                stop_idx = start_idx + samples_to_read
                read_signals[sn] = soundfile.read(info.name, start=start_idx, stop=stop_idx)[0]
                continue  # Go to the next serial number

            # The requested data spans multiple files
            files_to_read = []
            for info in files[files.index(info):]:
                files_to_read.append(info)
                if info.stop_time >= self.time_window.stop:
                    break
            else:
                raise ValueError(f'Cannot read data extending to {self.time_window.stop}, last file ends at {info.stop_time}')

            # Check that the file boundaries are good
            for early, late in zip(files_to_read[:-1], files_to_read[1:]):
                interrupt = (late.start_time - early.stop_time).total_seconds()
                if interrupt > self.allowed_interrupt:
                    raise ValueError(
                        f'Data is not continuous, missing {interrupt} seconds between files '
                        f'ending at {early.stop_time} and starting at {late.start_time}\n'
                        f'{early.name}\n{late.name}'
                    )

            read_chunks = []

            start_idx = (self.time_window.start - files_to_read[0].start_time).total_seconds() * self.samplerate
            chunk = soundfile.read(files_to_read[0].name, start=start_idx)[0]
            read_chunks.append(chunk)
            samples_to_read -= chunk.size
            for file in files_to_read[1:-1]:
                chunk = soundfile.read(file.name)
                read_chunks.append(chunk)
                samples_to_read -= chunk.size
            chunk = soundfile.read(files_to_read[-1].name, stop=samples_to_read)
            read_chunks.append(chunk)
            samples_to_read -= chunk.size
            assert samples_to_read == 0

            read_signals[sn] = np.concatenate(read_chunks, axis=0)
            # Ready to read from the collected files and store in the signal array

        read_signals = np.stack([read_signals[sn] for sn in self._identifiers], axis=0).squeeze()
        calibrations = [self.calibration[sn] for sn in self._identifiers]
        if None not in calibrations:
            return signals.Pressure.from_raw_and_calibration(
                read_signals,
                calibrations,
                samplerate=self.samplerate,
                start_time=self.time_window.start,
            )
        else:
            return signals.Signal(
                read_signals,
                samplerate=self.samplerate,
                start_time=self.time_window.start
            )
