import plotly.graph_objects as go
import numpy as np


def make_chart(center, zoom=14, mapbox_accesstoken=None):
    if mapbox_accesstoken is None:
        import os
        mapbox_accesstoken = os.getenv("MAPBOX_ACCESSTOKEN")
        if mapbox_accesstoken is None:
            import dotenv
            mapbox_accesstoken = dotenv.get_key(dotenv.find_dotenv(), "MAPBOX_ACCESSTOKEN")
    return go.Figure(layout=dict(
        mapbox=dict(
            accesstoken=mapbox_accesstoken,
            style='light',
            zoom=zoom,
            center=dict(lat=float(center.latitude), lon=float(center.longitude)),
        ),
        height=1200,
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        )
    ))


def plot_track(track, **kwargs):
    kwargs['lat'] = np.atleast_1d(track.latitude)
    kwargs['lon'] = np.atleast_1d(track.longitude)
    if hasattr(track, 'time'):
        if (
            track.time.size == track.latitude.size
            and 'hovertemplate' not in kwargs
            and 'customdata' not in kwargs
        ):
            kwargs['customdata'] = track.time
            kwargs['hovertemplate'] = '(%{lat}º, %{lon}º), %{customdata|%X}'
        else:
            kwargs.setdefault('hovertemplate', '(%{lat}º, %{lon}º)')
    return go.Scattermapbox(**kwargs)


def listen(recording, downsampling=1, upsampling=None, **kwargs):
    import sounddevice as sd
    from .recordings import Recording
    sd.stop()
    if isinstance(recording, Recording):
        recording = recording.time_data()
    if upsampling:
        recording = recording[::upsampling]
    recording -= recording.mean()
    recording /= np.max(np.abs(recording))
    sd.play(recording, samplerate=round(recording.sampling.rate / downsampling), **kwargs)
