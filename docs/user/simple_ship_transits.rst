Simple Analysis of Ship Transits
================================

.. role:: python(code)
   :language: python

.. default-literal-role:: python

In this tutorial we will do the basic steps required to analyze the source levels of some ship transits.

.. dropdown:: Initialization code

    .. code-block:: python

        import uwacan
        from pathlib import Path
        import plotly.graph_objects as go

        data_path = Path(r"D:\ShipMeasurement")

Ship track
----------
Since we will analyze transits of a ship and compute the source level,
we need to know how the ship has traveled.
We load a position track from a file::

    track = uwacan.Track.read_blueflow(data_path / "blueflow.csv")

This returns a `uwacan.Track` object, which has many useful methods to select specific portions, calculate distances, etc.
The track can quickly be visualized on a map as::

    fig = track.make_figure()
    fig.add_trace(track.plot())

.. only:: development

    .. code-block:: python

        fig = track.make_figure()
        fig.add_trace(track.plot())
        fig.write_html(
            "docs/user/plots/simple_ship_transits_track.html",
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            post_script=""
            "window.addEventListener('load', function() {"
            "    window.dispatchEvent(new Event('resize'));"
            "});"
        )

.. raw:: html
    :file: plots/simple_ship_transits_track.html

Sensors
-------
To calculate ship source levels, we need information about the `~uwacan.sensor`
that were used to measure the sound.
The main information needed is the sensor sensitivity and its location::

    sensor = uwacan.sensor(
        "SoundTrap 1",
        sensitivity=-180,
        latitude=57.62325,
        longitude=11.57790,
        depth=40,
    )

In ``uwacan``, sensors always have to be labeled.

Recordings
----------
Recordings are object is responsible for loading time-data from disk,
and performs conversions from the data storage format to physical units::

    recording = uwacan.recordings.SoundTrap.read_folder(
        data_path / "SoundTrap 1",
        sensor=sensor,
        time_compensation=7200,
    )

Some common recoding types are implemented in `uwacan.recordings`.
In this example, the recording unit was not set up to use UTC, so we include a 2 hour time compensation.

Background noise
----------------
Background noise should also be recorded and compensated for.
A background noise compensation can be created using the `uwacan.background.Background`
class, but it needs a spectrum to work with.
We can create a suitable spectrum by first computing a spectrogram from time data,
then averaging it over time::

    time_data = recording.subwindow(center="2023-08-30 12:00:00z", duration=60).time_data()
    spectrogram = uwacan.spectral.Spectrogram(
        time_data,
        bands_per_decade=10,
        min_frequency=20,
        max_frequency=40_000,
    )
    spectrum = spectrogram.mean("time")
    background_noise = uwacan.background.Background(spectrum)

Propagation model
-----------------
To compute the source level from received levels, we also need a propagation model.
There are a couple simple models implemented in `uwacan.propagation`::

    propagation_model = uwacan.propagation.SeabedCriticalAngle(
        water_depth=50,
        substrate_compressional_speed=1600,
        speed_of_sound=1503,
    )
    track["depth"] = 5

Since this propagation model needs the source depth, we specify the depth in the
position track for the ship.

Filterbank
----------
Finally, we need to specify what type of filterbank should be used to
compute the frequency spectrum from the time data::

    filterbank = uwacan.spectral.Spectrogram(
        frame_overlap=0.5,
        bands_per_decade=100,
        hybrid_resolution=0.2,
        min_frequency=5,
        max_frequency=40e3,
    )

Transits
--------
Our main interest here is to calculate the ship source level from a number
of controlled transits.
A `uwacan.Transit` is a collection of both a `~uwacan.recordings.Recording` and a `~uwacan.Track`::

    transit = uwacan.Transit(recording, track)

If we have more than one transit, we need to manually separate them.
This is easiest done by selecting the time of each transit, using some
combination of start, stop, center, and duration::

    transits = [
        transit.subwindow(center="2023-08-30 12:20:00z", duration=300),
        transit.subwindow(start="2023-08-30 12:36:00z", stop="2023-08-30 12:41:00z"),
        transit.subwindow(start="2023-08-30 12:58:00z", duration=300),
        transit.subwindow(duration=300, stop="2023-08-30 13:34:00z"),
    ]

The `~uwacan.TimeWindow.subwindow` method is used a lot, and is implemented
on all ``uwacan`` objects where it makes sense to select data over time.
Transits can typically be selected to be much larger than the section
that should be analyzed: the final selection using the closest point of approach is done later.

Source levels
-------------
There are many ways to compute source levels, but the easiest way
is to average the source level spectrogram over some section
chosen around the closest point of approach (CPA) with some pre-set rules::

    ship_level = uwacan.analysis.ShipLevel.analyze_transits(
        *transits,
        filterbank=filterbank,
        propagation_model=propagation_model,
        background_noise=background_noise,
        transit_min_angle=30,
        transit_min_duration=30,
    )

This creates a `~uwacan.analysis.ShipLevel` object, which has the source level
and the received level for all transits and segments in the transits.

Analysis
--------
Most of the time, we want the average of all transits and segments, but averaged in different ways::

    source_level = ship_level.power_average("segment").level_average("transit").source_level
    fig = go.Figure()
    fig.add_scatter(x=source_level.frequency, y=source_level)
    fig.update_xaxes(type="log", title="Frequency in Hz")
    fig.update_yaxes(title="Source level in dB re. (1 μPa m)<sup>2</sup>/Hz")
    fig.show()

.. only:: development

    .. code-block:: python

        fig.write_html(
            "docs/user/plots/simple_ship_transits_source_level.html",
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            post_script=""
            "window.addEventListener('load', function() {"
            "    window.dispatchEvent(new Event('resize'));"
            "});"
        )

.. raw:: html
    :file: plots/simple_ship_transits_source_level.html

Another useful plot is to check the received level compared to the background noise::

    fig = go.Figure()
    fig.add_scatter(x=background_noise.frequency, y=uwacan.dB(background_noise), name="Background")
    for idx, transit in ship_level.power_average("segment").received_level.groupby("transit"):
        fig.add_scatter(x=transit.frequency, y=transit, name=f"Transit {idx}")
    fig.update_xaxes(type="log", title="Frequency in Hz")
    fig.update_yaxes(title="Received level in dB re. 1 μPa<sup>2</sup>/Hz")
    fig.show()

.. only:: development

    .. code-block:: python

        fig.write_html(
            "docs/user/plots/simple_ship_transits_received_levels.html",
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            post_script=""
            "window.addEventListener('load', function() {"
            "    window.dispatchEvent(new Event('resize'));"
            "});"
        )

.. raw:: html
    :file: plots/simple_ship_transits_received_levels.html
