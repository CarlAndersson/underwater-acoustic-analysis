# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (tries to) adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.0]

### Added
- New `propagation.SmoothSemiCoherentImage` propagation model, the smoothed semi-coherent image method from Yubero et al. (2025) and ISO 17208-3. It sums a direct path, a bottom-reflected image path, and a multipath part, and needs the seabed density ratio and attenuation.
- New `propagation.Seabed` class for seabed properties: speed ratio, density ratio, attenuation, grain size, and porosity, all relative to the water where applicable. Models take a `seabed` argument that can be a preset name, a dict of properties, or a `Seabed`.
- `propagation.seabed_presets` with the full sediment table from Ainslie (2010), from very coarse sand to medium clay, including the class boundaries in between, e.g. `"coarse to medium sand"`.
- `propagation_loss` method on all non-local propagation models, giving `-10 log10` of the propagation factor.
- `propagation.slant_range` and `propagation.lf_hf_mix` as public utility functions. `lf_hf_mix` has a `power` argument to control the sharpness of the low- to high-frequency transition.
- `propagation.speed_of_sound_mackenzie`, the nine-term equation for the speed of sound in seawater.

### Changed
- Speeds up `correct_gps_offset` by about 2x by avoiding some xarray-work.
- **Breaking:** `power_propagation` on the propagation models is renamed to `propagation_factor`.
- **Breaking:** `SeabedCriticalAngle` takes a required `seabed` argument instead of `substrate_compressional_speed`. To keep using an absolute seabed speed, pass `seabed=Seabed(speed_ratio=seabed_speed / water_speed)`.
- **Breaking:** `seabed_properties` is replaced by `seabed_presets`, which holds `Seabed` objects. The preset speeds are now ratios that scale with the speed of sound used in a model, instead of fixed speeds computed for 1500 m/s.
- **Breaking:** `NonlocalPropagationModel.slant_range` is replaced by the module-level `propagation.slant_range`.
- The propagation models no longer inherit from each other. Each model now subclasses `NonlocalPropagationModel` directly and contains its full calculation, so a model can be read in one place. `SeabedCriticalAngle` is no longer a subclass of `SmoothLloydMirror`. The computed values are unchanged.
- `SeabedCriticalAngle` handles seabeds with a speed ratio of at most one by leaving out the cylindrical part, instead of returning NaN.

### Fixed
- `slant_range` with an array of source depths no longer raises an error.

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
