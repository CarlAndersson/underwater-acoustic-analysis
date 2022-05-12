"""A collection of analysis methods for underwater acoustics, specialized on radiated noise from ships."""
from . import _version  # noqa: E402
__version_info__ = _version.version_info
__version__ = _version.version
del _version

from . import positional  # noqa: E402, F401
