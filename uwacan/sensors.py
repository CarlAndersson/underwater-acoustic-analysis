from . import positional

import xarray as xr


def sensor(label, sensitivity=None, position=None, depth=None):
    data_vars = {}
    if sensitivity is not None:
        data_vars['sensitivity'] = sensitivity
    if position is not None:
        position = positional.position(position)
        data_vars['latitude'] = position.latitude
        data_vars['longitude'] = position.longitude
    if depth is not None:
        data_vars['depth'] = depth

    return xr.Dataset(data_vars=data_vars, coords={'sensor': label})


def sensor_array(*sensors):
    return xr.concat(sensors, dim='sensor')


def align_property_to_sensors(sensors, values, allow_scalar=False):
    sensor_names = sensors.sensor

    if isinstance(values, xr.DataArray):
        return values
    if allow_scalar:
        try:
            len(values)
        except TypeError:
            return values

    if len(values) != sensor_names.size:
        raise ValueError(f"Cannot assign {len(values)} values to {sensor_names.size} sensors")

    try:
        return xr.DataArray(values, coords={'sensor': sensor_names})
    except ValueError:
        pass
    return xr.DataArray([values[key] for key in sensor_names.values], coords={'sensor': sensor_names})
