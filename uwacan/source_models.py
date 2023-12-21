import numpy as np
import xarray as xr
from . import _tools


@_tools.prebind
def class_limit_curve(frequency, breakpoints=None, limits=None):
    # TODO: Enable outputting monopole or radiated levels
    # TODO: Enable outputting psd or total power. This will require some pondering on how to input the specifications...
    conditions = [frequency <= bp for bp in breakpoints] + [xr.DataArray(True)]
    values = [limit(frequency) for limit in limits]
    conditions = xr.concat(conditions, dim='conditions', coords='minimal')
    values = xr.concat(values, dim='conditions')
    return values.where(conditions).bfill('conditions').isel(conditions=0)


bureau_veritas_advanced = class_limit_curve(
    breakpoints=[50, 1e3],
    limits=[
        lambda f: 174 - 11 * np.log10(f),
        lambda f: 155.3 - 18 * np.log10(f / 50),
        lambda f: 131.9 - 22 * np.log10(f / 1000),
    ]
)


bureau_veritas_controlled = class_limit_curve(
    breakpoints=[50, 1e3],
    limits=[
        lambda f: 169 - 2 * np.log10(f),
        lambda f: 165.6 - 20 * np.log10(f / 50),
        lambda f: 139.6 - 20 * np.log10(f / 1000),
    ]
)


@_tools.prebind
def jomopans_echo_model(frequency, ship_class=None, speed=None, length=None):
    K = 191
    K_lf = 208

    D = 3
    match ship_class:
        case 'fishing':
            v_class = 6.4
        case 'tug':
            v_class = 3.7
        case 'naval':
            v_class = 11.1
        case 'recreational':
            v_class = 10.6
        case 'research':
            v_class = 8.0
        case 'cruise':
            v_class = 17.1
            D = 4
        case 'passenger':
            v_class = 9.7
        case 'bulker':
            v_class = 13.9
            D_lf = 0.8
        case 'container':
            v_class = 18
            D_lf = 0.8
        case 'vehicle':
            v_class = 15.8
            D_lf = 1
        case 'tanker':
            v_class = 12.4
            D_lf = 1
        case 'other':
            v_class = 7.4
        case 'dredger':
            v_class = 9.5
        case _:
            raise ValueError(f"Unknown ship class '{ship_class}'")

    f1 = 480 / v_class

    baseline =  K - 20 * np.log10(f1) - 10 * np.log10((1 - frequency / f1)**2 + D**2)
    if ship_class in {'container', 'vehicle', 'bulker', 'tanker'}:
        f_lf = 600 / v_class
        lf_baseline = K_lf - 40 * np.log10(f_lf) + 10 * np.log10(frequency) - 10 * np.log10((1 - (frequency / f_lf)**2)**2 + D_lf**2)
        baseline = xr.where(frequency < 100, lf_baseline, baseline)
    l = length * 3.28084 / 300
    return baseline + 60 * np.log10(speed / v_class) + 20*np.log10(l)
