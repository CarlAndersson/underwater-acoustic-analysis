# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (tries to) adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New functions for processing of swept sine propagation loss measurements.
- Adds the DNV silent-e transit criteium as a "source model".

### Changed
- The previous "private" filtering module is now exposed as a public `spectral` module. This exposes the filterbank funcionality used to compute spectrograms, and some simple fft wrappers.
- Uses the actual frequency band edges in nth decade band processing. This will enable more accurate processing in the future, in regards to conversions between different band types. The bandwidth is still available as a property, for processing that only need the bandwidth.
- Allows the use of pre-computed spectrograms in transit analysis.
