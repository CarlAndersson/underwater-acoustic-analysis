"""A collection of analysis methods for underwater acoustics, specialized on radiated noise from ships."""
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
    Sensor,
    SensorArray,
)
