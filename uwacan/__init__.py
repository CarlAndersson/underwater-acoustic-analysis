"""A collection of analysis methods for underwater acoustics, specialized on radiated noise from ships.

The main package namespace holds some commonly used functions and classes.

Classes for data handling
-------------------------
.. autosummary::
    :toctree: generated

    TimeData
    FrequencyData
    TimeFrequencyData

Classes for positions and sensors
---------------------------------
.. autosummary::
    :toctree: generated

    Position
    Track
    sensor

Other common operations
-----------------------
.. autosummary::
    :toctree: generated

    dB
    Transit
    TimeWindow
    load_data
"""

from ._version import version as __version__

del _version

from . import (  # noqa: E402
    positional,
    recordings,
    analysis,
    propagation,
    background,
    source_models,
    spectral,
)  # noqa: E402, F401

from ._core import (
    TimeWindow,
    dB,
    TimeData,
    FrequencyData,
    TimeFrequencyData,
    Transit,
)
from .positional import (
    Position,
    Track,
    sensor,
)

load_data = _core.xrwrap.load
