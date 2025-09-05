Simple Analysis of Recording Levels
===================================

.. role:: python(code)
   :language: python

.. default-literal-role:: python

This tutorial covers the basic steps required to load recording data,
make a simple spectrogram, and how to compute and plot hourly averages over a single day.

.. dropdown:: Initialization code

    .. code-block:: python

        >>> import uwacan
        >>> from pathlib import Path

        >>> data_path = Path(r"D:\ExampleData\LongRecording")

Sensor information
------------------
To load data from recordings and convert it to physical units, we need to specify some information about the sensor used to do the recording.
For a simple level analysis, only the sensor sensitivity is required, but for other types of analysis the position and depth of the sensor might also be needed.
We store the information in a `~uwacan.sensor` object::

    >>> sensor = uwacan.sensor(
    ...     "RTsys 1",
    ...     sensitivity=-179.2,
    ...     position="58° 51.065'N 11° 4.569'E",
    ...     depth=18,
    ... )

The first argument gives a label. In ``uwacan``, sensors always have to be labeled.

Recordings
----------
Recordings are object is responsible for loading time-data from disk,
and performs conversions from the data storage format to physical units::

    >>> recording = uwacan.recordings.SylenceLP.read_folder(
    ...     data_path / "RTsys 1" / "timedata",
    ...     sensor=sensor,
    ... )
    >>> print(recording.time_window)
    TimeWindow(start=2024-08-01 00:19:58Z, stop=2024-08-04 00:26:07.952Z)

Some common recording types are implemented in `uwacan.recordings`.

Spectrogram
-----------
To compute a detailed `~uwacan.analysis.Spectrogram`, we start by selecting a shorter time to compute it over.
This is done using the `~uwacan.TimeWindow.subwindow` method, which is implemented on most of the objects which hold time dependent data::

    >>> selection = recording.subwindow(start="2024-08-02 06:30:00z", duration=10 * 60)
    >>> time_data = selection.time_data()
    >>> spectrogram = uwacan.analysis.Spectrogram.analyze_timedata(
    ...     time_data,
    ...     hybrid_resolution=2,
    ...     bands_per_decade=100,
    ...     min_frequency=10,
    ...     max_frequency=40e3,
    ... )

    >>> fig = spectrogram.make_figure()
    >>> fig.add_trace(spectrogram.plot(zmin=30, zmax=90))
    >>> fig.show()

.. only:: development

    .. code-block:: python
        fig.write_html(
            "docs/user/plots/simple_levels_short_spectrogram.html",
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            post_script=""
            "window.addEventListener('load', function() {"
            "    window.dispatchEvent(new Event('resize'));"
            "});"
        )

.. raw:: html
    :file: plots/simple_levels_short_spectrogram.html

Spectrum
--------
From this spectrogram we can easily compute the spectrum (power spectral density) for the same time window, by averaging over time::

    >>> spectrum = spectrogram.mean("time")
    >>> fig = spectrum.make_figure()
    >>> fig.add_trace(spectrum.plot(name="With pulse"))
    >>> fig.add_trace(
    ...     uwacan.analysis.Spectrogram.analyze_timedata(
    ...         recording.subwindow(start="2024-08-02 06:30:00z", stop="2024-08-02 06:38:00z"),
    ...         hybrid_resolution=2,
    ...         bands_per_decade=100,
    ...         min_frequency=10,
    ...         max_frequency=40e3,
    ...     ).mean("time").plot(name="Without pulse")
    ... )
    >>> fig.show()

.. only:: development

    .. code-block:: python
        fig.write_html(
            "docs/user/plots/simple_levels_short_level.html",
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            post_script=""
            "window.addEventListener('load', function() {"
            "    window.dispatchEvent(new Event('resize'));"
            "});"
        )

.. raw:: html
    :file: plots/simple_levels_short_level.html

Of interest here is the strong influence that the pulse at 06:38 has on the average level at high frequencies - rising it by up to 40 dB.
This pulse in the recording comes from the nearby acoustic releaser used during deployment.

Hourly spectra
--------------
Learning from how we compute spectrograms and spectra, we can compute a spectrum for each hour of the day.
To help us with accessing the recording in one-hour sequential chunks, we can use the `~uwacan.recordings.AudioFileRecording.rolling` method::

    >>> selection = recording.subwindow(start="2024-08-02 00:00:00z", stop="2024-08-03 00:00:00z")
    >>> spectra = []
    >>> for hour in selection.rolling(duration=3600, overlap=0):
    ...     spectrogram = uwacan.analysis.Spectrogram.analyze_timedata(
    ...         hour,
    ...         hybrid_resolution=2,
    ...         bands_per_decade=100,
    ...         min_frequency=10,
    ...         max_frequency=40e3,
    ...     )
    ...     spectrum = spectrogram.mean("time")
    ...     spectrum.coords["time"] = uwacan.time_to_np(spectrogram.time_window.center)
    ...     spectra.append(spectrum)
    >>> spectra = uwacan.concatenate(spectra, dim="time")

To plot this, we want to loop over the time dimension in the spectra. This is easiest done using the `~uwacan._core.xrwrap.groupby` method.
Combining this with some nice colorscale sampling from plotly, we arrive at::

    >>> fig = spectra.make_figure()
    >>> colors = plotly.colors.sample_colorscale("twilight", spectra.sizes["time"])
    >>> for color, (time, spectrum) in zip(colors, spectra.groupby("time")):
    ...     label = uwacan.time_to_datetime(time).py_datetime().strftime("%H:%M")
    ...     fig.add_trace(spectrum.plot(name=label, line_color=color))
    >>> fig.show()

.. only:: development

    .. code-block:: python

        fig.write_html(
            "docs/user/plots/simple_levels_per_hour.html",
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            post_script=""
            "window.addEventListener('load', function() {"
            "    window.dispatchEvent(new Event('resize'));"
            "});"
        )

.. raw:: html
    :file: plots/simple_levels_per_hour.html
