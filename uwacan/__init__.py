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
    sensor_array

Other common operations
-----------------------
.. autosummary::
    :toctree: generated

    dB
    Transit
    TimeWindow
"""

from . import _version  # noqa: E402

__version_info__ = _version.version_info
__version__ = _version.version
del _version

from . import (  # noqa: E402
    positional,
    recordings,
    analysis,
    propagation,
    background,
    source_models,
    visualization,
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
    sensor_array,
)
