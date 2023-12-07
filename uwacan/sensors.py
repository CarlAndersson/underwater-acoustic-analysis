from . import positional

import xarray as xr


def sensor(label, sensitivity=None, position=None, depth=None):
    data_vars = {}
    if sensitivity is not None:
        data_vars['sensitivity'] = sensitivity
    if position is not None:
        data_vars['latitude'] = position.latitude
        data_vars['longitude'] = position.longitude
    if depth is not None:
        data_vars['depth'] = depth

    return xr.Dataset(data_vars=data_vars, coords={'sensor': label})


def sensor_array(*sensors):
    return xr.concat(sensors, dim='sensor')
