# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (tries to) adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Speeds up `correct_gps_offset` by about 2x by avoiding some xarray-work.

## [2.3.3]

### Changed
- `sel` and `isel` on the data wrappers no longer drop all-nan positions by default, i.e. `drop_allnan` defaults to `False`. Pass `drop_allnan=True` to get the old behaviour. Dropping was never restricted to the dimensions being selected, so e.g. `spectrogram.isel(sensor=0)` could silently remove an all-nan frequency band, it cast integer variables to float and strings to object, and it undid the nan separators that `concatenate(nan_between_items=True)` inserts on purpose. It also dominated the runtime of these methods, at roughly 2800 us versus 60 us per call on a rolling spectrogram frame.

## [2.3.2]
### Changed
- Default resolution for coordinate prints increased to 6 digits.

## Fixed
- Some warnings for future xarray changes.

## [2.3.1]

### Changed
- Improves the calculation speed of `positional` calculations when called with xarray data, e.g., from the class methods. The performance bump is around 10x for typical tracks. The change is that we're using `xarray.apply_ufunc` in a decorator to align the dimensions first, and then the computation takes place without the xarray overhead.
- Fixes handling of times wrapped in xarray DataArrays when using the time conversion methods.
- Changes time subwindow method for time data wrapper. This no longer computes the sample indices, but instead uses the actual times.

## [2.3.0]

### Added
- New functions for processing of swept sine propagation loss measurements.
- Adds the DNV silent-e transit criterion as a "source model".

### Changed
- The previous "private" filtering module is now exposed as a public `spectral` module. This exposes the filterbank funcionality used to compute spectrograms, and some simple fft wrappers.
- Uses the actual frequency band edges in nth decade band processing. This will enable more accurate processing in the future, in regards to conversions between different band types. The bandwidth is still available as a property, for processing that only need the bandwidth.
- Allows the use of pre-computed spectrograms in transit analysis.
