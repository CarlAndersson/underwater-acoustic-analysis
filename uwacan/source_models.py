import numpy as np
import xarray as xr


def class_limit_curve(frequency, breakpoints, limits):
    """Evaluate a class limit curve defined by frequency breakpoints and limits.

    With N breakpoints, there are N+1 regions and corresponding limits.

    Parameters
    ----------
    frequency : array_like
        Input frequency data.
    breakpoints : list of float
        The frequency breakpoints, in ascending order.
    limits : list with the limits
        Each limit is either a callable which takes the frequency as the input,
        a single value valid within the entire range, or an array_like with the
        same shape as frequency.

    Returns
    -------
    array_like
        Array of the same shape as `frequency`.

    Examples
    --------
    class_limit_curve(
        np.array([50, 100, 200, 400, 800, 1600]),
        [150, 800],
        [10, 20, 30],
    )
    >>> [10, 10, 20, 20, 30, 30]
    """
    conditions = [frequency < b for b in breakpoints] + [np.full(np.shape(frequency), True)]
    limits = [limit(frequency) if callable(limit) else limit for limit in limits]
    levels = np.select(conditions, limits)
    try:
        wrapper = frequency.__array_wrap__
    except AttributeError:
        return levels
    else:
        return wrapper(levels)


def bureau_veritas_advanced(frequency=None):
    """The advanced vessel limit from Bureau Veritas.

    This ship level is a radiated noise level, as a spectral density level.
    """
    if frequency is None:
        frequency = 10 ** (np.arange(10, 48) / 10)  # Decidecade bands from 10 Hz to 50 kHz
        frequency = xr.DataArray(frequency, coords={"frequency": frequency})
    return class_limit_curve(
        frequency=frequency,
        breakpoints=[50, 1e3],
        limits=[
            lambda f: 174 - 11 * np.log10(f),
            lambda f: 155.3 - 18 * np.log10(f / 50),
            lambda f: 131.9 - 22 * np.log10(f / 1000),
        ]
    )


def bureau_veritas_controlled(frequency=None):
    """The controlled vessel limit from Bureau Veritas.

    This ship level is a radiated noise level, as a spectral density level.
    """
    if frequency is None:
        frequency = 10 ** (np.arange(10, 48) / 10)  # Decidecade bands from 10 Hz to 50 kHz
        frequency = xr.DataArray(frequency, coords={"frequency": frequency})
    return class_limit_curve(
        frequency=frequency,
        breakpoints=[50, 1e3],
        limits=[
            lambda f: 169 - 2 * np.log10(f),
            lambda f: 165.6 - 20 * np.log10(f / 50),
            lambda f: 139.6 - 20 * np.log10(f / 1000),
        ]
    )


def wales_heitmeyer(frequency):
    """Evaluate the Wales-Heitmeyer source model

    Note that this is a monopole source level model,
    with unknown validity for the source depths.

    Returns
    -------
    L : array_like
        The calculated source level, as a spectral density level.
    """
    return (
        230
        - 10 * np.log10(frequency ** 3.594)
        + 10 * np.log10((1 + (frequency / 340)**2)**0.917)
    )


def jomopans_echo_model(frequency, ship_class, speed, length):
    """Evaluate Jomopans-ECHO source model

    Make sure to use the correct units for speed and length.
    Note that this is a monopole source level model, assuming
    a source depth of 6 meters.

    Parameters
    ----------
    frequency : array_like
        The frequencies at which to evaluate
    ship_class : str
        Which ship class to use.
    speed : float
        The ship speed, in knots
    length : float
        The ship length, in meters


    Returns
    -------
    L : array_like
        The calculated source level, as a spectral density level.
    """
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
