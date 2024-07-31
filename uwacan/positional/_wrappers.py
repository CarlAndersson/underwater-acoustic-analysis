import collections.abc
import re
import numpy as np
import xarray as xr
import pendulum
from . import implementations as impl


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
            # It's not a period, so it shold be a single datetime. Parse or convert, check valitidy.
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


class _Coordinates(collections.abc.MutableMapping):
    def __init__(self, coordinates=None, /, latitude=None, longitude=None):
        if coordinates is None:
            coordinates = xr.Dataset(data_vars={"latitude": latitude, "longitude": longitude})
        if isinstance(coordinates, _Coordinates):
            coordinates = coordinates._data.copy()
        self._data = coordinates

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

    @property
    def coordinates(self):
        return self._data[["latitude", "longitude"]]

    @property
    def latitude(self):
        return self._data['latitude']

    @property
    def longitude(self):
        return self._data['longitude']

    def distance_to(self, other):
        """Calculates the distance to another coordinate."""
        other = self._ensure_latlon(other)
        return impl.distance_to(self.latitude, self.longitude, other.latitude, other.longitude)

    def bearing_to(self, other):
        """Calculates the bearing to another coordinate."""
        other = self._ensure_latlon(other)
        return impl.bearing_to(self.latitude, self.longitude, other.latitude, other.longitude)

    def shift_position(self, distance, bearing):
        """Shifts this position by a distance in a certain bearing"""
        other = self._ensure_latlon(other)
        lat, lon = impl.shift_position(self.latitude, self.longitude, distance, bearing)
        return type(self)(latitude=lat, longitude=lon)

    @classmethod
    def _ensure_latlon(cls, data):
        if not (hasattr(data, "latitude") and hasattr(data, "longitude")):
            # If it doesn't have lat and long we need to construct an object which
            # has them. If we get lists of values we will get a `Position` with lat,lon
            # arrays here. This usually doesn't work, but we only need to access them,
            # which will work fine. The `Position` can handle many of the other
            # useful stuff, like strings and tuples
            data = Position(data)
        return data

    @classmethod
    def from_local_mercator(cls, easting, northing, reference_coordinate, **kwargs):
        r"""Convert local mercator coordinates into wgs84 coordinates.

        Conventions here are :math:`\lambda` as the longitude and :math:`\varphi` as the latitude,
        :math:`x` is easting and :math:`y` is northing.

        .. math::

            \lambda &= \lambda_0 + \frac{x}{R} \\
            \varphi &= 2\arctan\left[\exp \left(\frac{y + y_0}{R}\right)\right] - \frac{\pi}{2}

        The northing offset :math:`y_0` is computed by converting the reference point into
        a mercator projection with the `wgs84_to_local_mercator` function, using (0, 0) as
        the reference coordinates.
        """
        reference_coordinate = Position(reference_coordinate)
        lat, lon = impl.local_mercator_to_wgs84(
            easting, northing,
            reference_coordinate.latitude, reference_coordinate.longitude
        )
        return cls(latitude=lat, longitude=lon, **kwargs)

    def to_local_mercator(self, reference_coordinate):
        r"""Convert wgs84 coordinates into a local mercator projection.

        Conventions here are :math:`\lambda` as the longitude and :math:`\varphi` as the latitude,
        :math:`x` is easting and :math:`y` is northing.

        .. math::
            x &= R(\lambda -\lambda _{0})\\
            y &= R\ln \left[\tan \left(\frac{\pi}{4} + \frac{\varphi - \varphi_0}{2}\right)\right]
        """
        reference_coordinate = Position(reference_coordinate)
        easting, northing = impl.wgs84_to_local_mercator(
            self.latitude, self.longitude,
            reference_coordinate.latitude, reference_coordinate.longitude
        )
        return easting, northing

    def local_length_scale(self):
        """How many nautical miles one longitude minute is

        This gives the apparent length scale for the x-axis in
        mercator projections, i.e., cos(latitude).
        The scaleratio for an x-axis should be set to this value,
        if equal length x- and y-axes are desired, e.g.,
        ```
        xaxis=dict(
            title_text='Longitude',
            constrain='domain',
            scaleanchor='y',
            scaleratio=pos.local_length_scale(),
        ),
        yaxis=dict(
            title_text='Latitude',
            constrain='domain',
        ),
        ```
        """
        # We take the mean so that it works with subclasses with arrays, e.g., Line.
        return np.cos(np.radians(self.latitude.mean().item()))


class Position(_Coordinates):
    @staticmethod
    def parse_coordinates(*args, **kwargs):
        try:
            return kwargs['latitude'], kwargs['longitude']
        except KeyError:
            pass

        if len(args) == 1:
            arg = args[0]
            try:
                return arg.latitude, arg.longitude
            except AttributeError:
                pass
            try:
                return arg['latitude'], arg['longitude']
            except (KeyError, TypeError):
                pass
            if isinstance(arg, str):
                matches = re.match(
                    r"""((?P<latdeg>[+\-\d.]+)°?)?((?P<latmin>[\d.]+)')?((?P<latsec>[\d.]+)")?(?P<lathemi>[NS])?"""
                    r"""[,]?"""
                    r"""((?P<londeg>[+\-\d.]+)°?)?((?P<lonmin>[\d.]+)')?((?P<lonsec>[\d.]+)")?(?P<lonhemi>[EW])?""",
                    re.sub(r"\s", "", arg)
                ).groupdict()
                if not matches["latdeg"] or not matches["londeg"]:
                    raise ValueError(f"Cannot parse coordinate string '{arg}'")

                digits_to_parse = len(re.sub(r"\D", "", arg))
                digits_parsed = 0
                latitude = float(matches["latdeg"])
                lat_sign = 1 if latitude >= 0 else -1
                digits_parsed += len(re.sub(r"\D", "", matches["latdeg"]))
                longitude = float(matches["londeg"])
                lon_sign = 1 if longitude >= 0 else -1
                digits_parsed += len(re.sub(r"\D", "", matches["londeg"]))

                if matches["latmin"]:
                    latitude += lat_sign * float(matches["latmin"]) / 60
                    digits_parsed += len(re.sub(r"\D", "", matches["latmin"]))
                if matches["lonmin"]:
                    longitude += lon_sign * float(matches["lonmin"]) / 60
                    digits_parsed += len(re.sub(r"\D", "", matches["lonmin"]))

                if matches["latsec"]:
                    latitude += lat_sign * float(matches["latsec"]) / 3600
                    digits_parsed += len(re.sub(r"\D", "", matches["latsec"]))
                if matches["lonsec"]:
                    longitude += lon_sign * float(matches["lonsec"]) / 3600
                    digits_parsed += len(re.sub(r"\D", "", matches["lonsec"]))

                if not digits_parsed == digits_to_parse:
                    raise ValueError(f"Could not parse coordinate string '{arg}', used only {digits_parsed} of {digits_to_parse} digits")

                if matches["lathemi"] == "S":
                    latitude = -abs(latitude)
                if matches["lonhemi"] == "W":
                    longitude = -abs(longitude)

                return latitude, longitude

            else:
                # We should never have just a single argument, try unpacking.
                *args, = arg

        if len(args) == 2:
            latitude, longitude = args
            return latitude, longitude
        elif len(args) == 4:
            (
                latitude_degrees, latitude_minutes,
                longitude_degrees, longitude_minutes
             ) = args
            latitude = latitude_degrees + latitude_minutes / 60
            longitude = longitude_degrees + longitude_minutes / 60
            return latitude, longitude
        elif len(args) == 6:
            (
                latitude_degrees, latitude_minutes, latitude_seconds,
                longitude_degrees, longitude_minutes, longitude_seconds
             ) = args
            latitude = latitude_degrees + latitude_minutes / 60 + latitude_seconds / 3600
            longitude = longitude_degrees + longitude_minutes / 60 + longitude_seconds / 3600
            return latitude, longitude
        else:
            raise TypeError(f"Undefined number of arguments for Position. {len(args)} was given, expects 2, 4, or 6.")

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (type(self), xr.Dataset)):
            super().__init__(args[0])
        else:
            latitude, longitude = self.parse_coordinates(*args, **kwargs)
            super().__init__(latitude=latitude, longitude=longitude)

    def __repr__(self):
        return f"{type(self).__name__}({self.latitude.item():.4f}, {self.longitude.item():.4f})"


class BoundingBox:
    def __init__(self, west, south, east, north):
        self.west = west
        self.south = south
        self.east = east
        self.north = north

    def __repr__(self):
        return f"{type(self).__name__}({self.west}, {self.south}, {self.east}, {self.north})"

    @property
    def north_west(self):
        return Position(self.north, self.west)

    @property
    def north_east(self):
        return Position(self.north, self.east)

    @property
    def south_west(self):
        return Position(self.south, self.west)

    @property
    def south_east(self):
        return Position(self.south, self.east)

    @property
    def center(self):
        return Position(latitude=(self.north + self.south) / 2, longitude=(self.west + self.east) / 2)

    def __contains__(self, position):
        position = Position(position)
        if (self.west <= position.longitude <= self.east) and (self.south <= position.latitude <= self.north):
            return True

    def overlaps(self, other):
        return (
                other.north_west in self
                or other.north_east in self
                or other.south_west in self
                or other.south_east in self
                or self.north_west in other
                or self.north_east in other
                or self.south_west in other
                or self.south_east in other
            )

    def zoom_level(self, pixels=800):
        center = self.center
        westing, northing = self.north_west.to_local_mercator(center)
        easting, southing = self.south_east.to_local_mercator(center)
        extent = max((northing - southing) / center.local_length_scale(), (easting - westing))
        # This has something to do with the size of a tile in pixels (256),
        # the length of the equator (40_000_000), and then some manual scaling
        # to fix the remainder of issues. Worked nice in plotly 5.18, calling mapbox.
        zoom = np.log2(40_000_000 * pixels / 256 / extent).item() - 1.2
        return zoom


class Line(_Coordinates):
    @classmethod
    def stack_positions(cls, positions, dim='point', **kwargs):
        """Stacks multiple positions into a line"""
        coordinates = [Position(pos).coordinates for pos in positions]
        coordinates = xr.concat(coordinates, dim=dim)
        return cls(coordinates, **kwargs)

    @classmethod
    def concatenate(cls, lines, dim=None, nan_between_lines=False, **kwargs):
        """Concatenates multiple lines

        If the lines are not connected, it is useful to set `nan_between_lines=True`, which puts
        a nan element between each line. This makes most plotting libraries split the lines in
        visualizations.
        """
        first_line_coords = lines[0].coordinates
        if dim is None:
            if len(first_line_coords.dims) != 1:
                raise ValueError("Cannot guess concatenation dimensions for multi-dimensional line.")
            dim = next(iter(first_line_coords.dims))

        if nan_between_lines:
            nan_data = xr.full_like(first_line_coords.isel({dim: 0}), np.nan).expand_dims(dim)
            lines = sum([[line.coordinates, nan_data] for line in lines], [])
        coordinates = xr.concat(lines, dim=dim)
        return cls(coordinates, **kwargs)


    def __repr__(self):
        return f"{type(self).__name__} with {self.latitude.size} points"

    @property
    def bounding_box(self):
        try:
            return self._bounding_box
        except AttributeError:
            pass
        west = self.longitude.min().item()
        east = self.longitude.max().item()
        north = self.latitude.max().item()
        south = self.latitude.min().item()
        self._bounding_box = BoundingBox(west=west, south=south, east=east, north=north)
        return self._bounding_box


class Track(Line):
    def __init__(self, data):
        super().__init__(data)

    @property
    def time(self):
        return self._data["time"]

    @property
