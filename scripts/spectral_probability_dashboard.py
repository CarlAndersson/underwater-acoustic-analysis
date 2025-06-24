import tomllib
import uwacan
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State, Patch
import plotly.graph_objects as go
import base64
import io
import numpy as np
import xarray as xr
import wave
import plotly.io as pio
import argparse


pio.templates["transparent"] = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
)

pio.templates.default = "plotly_white+transparent"

# Parse command line arguments for data folder
parser = argparse.ArgumentParser(description="Spectral Probability Dashboard")
parser.add_argument('data_folder', type=str, help='Path to the folder containing data files')
args, unknown = parser.parse_known_args()

# Initialize the Dash app
app = dash.Dash(__name__)

scroll_zoom = False
save_image_options = {
    'format': 'svg', # one of png, svg, jpeg, webp
    'height': 500,
    'width': 700,
    'scale': 1,
  }

# Define the available data files by listing subdirectories in the provided data folder
base_folder = Path(args.data_folder)
DATA_FILES = [
    {'label': subdir.name, 'value': str(subdir)}
    for subdir in base_folder.iterdir() if subdir.is_dir()
]

# Define colors for drawn lines
LINE_COLORS = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

# Define the layout
app.layout = html.Div([
    html.H1("Spectral Probability Dashboard"),

    # Dropdown component
    html.Div([
        html.Label("Select Data File:"),
        dcc.Dropdown(
            id='file-selector',
            options=DATA_FILES,
            value=DATA_FILES[0]['value'],
            style={'width': '100%', 'margin': '10px'}
        )
    ]),

    # Store component for last selection
    dcc.Store(id='spectrogram-time-window-store'),
    dcc.Store(id='audio-time-window-store'),
    # Confirmation modal
    dcc.ConfirmDialog(
        id='confirm-large-window',
        message='This time window is larger than 6 hours. Loading the spectrogram might take a while. Do you want to continue?',
    ),
    # Confirmation modal for audio
    dcc.ConfirmDialog(
        id='confirm-large-audio',
        message='This time window is larger than 1 minute. Loading the audio might take a while. Do you want to continue?',
    ),

    # Plot components
    dcc.Loading(
        id="loading-spectral-probability",
        type="circle",
        children=[
            dcc.Graph(
                id='spectral-probability-plot',
                config={
                    "toImageButtonOptions": save_image_options | {"filename": "spectral-probability"},
                }
            )
        ],
        delay_show=250,
    ),
    dcc.Loading(
        id="loading-deviation",
        type="circle",
        children=[dcc.Graph(
            id='deviation-plot',
            config={
                "modeBarButtons": [["toImage", "pan2d", "select2d", "resetViews"]],
                "scrollZoom": scroll_zoom,
                "toImageButtonOptions": save_image_options | {"filename": "deviation"},
            },
        )],
        delay_show=250,
    ),
    dcc.Loading(
        id="loading-spectrogram",
        type="circle",
        children=[dcc.Graph(
            id='spectrogram-plot',
            config={
                "modeBarButtons": [
                    ["select2d", "pan2d", "zoom2d"],
                    ["zoomIn2d", "zoomOut2d", "autoScale2d"],
                    ["toImage"],
                ],
                "scrollZoom": scroll_zoom,
                "toImageButtonOptions": save_image_options | {"filename": "spectrogram"},
            }
        )]
    ),
    # Audio player component
    html.Audio(id='audio-player', controls=True, style={'width': '100%', 'margin': '10px'})
])

@app.callback(
    Output('spectral-probability-plot', 'figure'),
    Input('file-selector', 'value')
)
def plot_spectral_probability(selected_file):
    if selected_file is None:
        return go.Figure()

    selected_file = Path(selected_file)
    try:
        # Load and process the data
        spectral_probability_series = uwacan.load_data(selected_file / "spectral_probability_series.zarr.zip")
        spectral_probability = spectral_probability_series.mean(dim="time")

        # Create the figure
        fig = spectral_probability.make_figure()
        fig.add_trace(spectral_probability.plot(name="Spectral Probability"))

        # Add percentile lines
        fig.add_trace(
            spectral_probability.plot_percentile(
                0.5,
                line=dict(color="black", dash="solid"),
                name="Median",
            )
        )
        fig.add_trace(
            spectral_probability.plot_percentile(
                0.9,
                line=dict(color="black", dash="dash"),
                name="90th Percentile",
                legendgroup="percentiles",
            )
        )
        fig.add_trace(
            spectral_probability.plot_percentile(
                0.1,
                line=dict(color="black", dash="dash"),
                name="10th Percentile",
                legendgroup="percentiles",
                showlegend=False,
            )
        )

        # Add drawopenpath tool to the figure
        fig.update_layout(
            dragmode='drawopenpath',
            modebar_add=['drawopenpath'],
            newshape=dict(line=dict(color=LINE_COLORS[0], width=2)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                # groupclick="toggleitem"
            )
        )

        return fig

    except Exception as e:
        print(f"Error processing file: {e}")
        return go.Figure()

@app.callback(
    Output('deviation-plot', 'figure'),
    [Input('spectral-probability-plot', 'relayoutData'),
     Input('file-selector', 'value')],
    State("spectrogram-time-window-store", "data"),
    prevent_initial_call=True,
)
def plot_deviation_to_drawn_target(drawn_target_lines, selected_file, spectrogram_time_window):
    # If the file selector triggered the callback, return empty figures
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'file-selector':
        return go.Figure()

    if not drawn_target_lines or 'shapes' not in drawn_target_lines:
        return dash.no_update

    shapes = drawn_target_lines['shapes']
    if not shapes:
        return dash.no_update

    # Load the data for comparison
    selected_file = Path(selected_file)
    spectral_probability_series = uwacan.load_data(selected_file / "spectral_probability_series.zarr.zip")

    # Get the time range from the data
    start_time = spectral_probability_series.time_window.start.py_datetime()
    end_time = spectral_probability_series.time_window.stop.py_datetime()

    # Create deviation plot with all lines
    deviation_fig = go.Figure()

    # Process each shape and add its deviation to the plot
    for i, shape in enumerate(shapes):
        # Extract x and y coordinates from the path string
        path_str = shape['path']
        points = path_str.replace('M', '').split('L')

        # Parse the points into x and y coordinates
        x_coords = []
        y_coords = []
        for point in points:
            x, y = map(float, point.split(','))
            x_coords.append(x)
            y_coords.append(y)

        # Create target DataArray from drawn line
        target = xr.DataArray(
            data=np.array(y_coords),
            dims=["frequency"],
            coords={"frequency": np.array(x_coords)},
        ).drop_duplicates("frequency").sortby("frequency")

        # Perform comparison calculation
        selection = spectral_probability_series.sel(frequency=slice(min(target.frequency), max(target.frequency))).data
        target = target.interp(frequency=selection.frequency)
        weight = abs(target - selection.levels)
        deviation = (selection * weight).sum("levels") * spectral_probability_series.attrs["binwidth"]
        deviation = deviation.mean("frequency")

        # Add trace to deviation plot with matching color
        line_color = LINE_COLORS[i % len(LINE_COLORS)]
        deviation_fig.add_trace(
            go.Scatter(
                x=deviation.time,
                y=deviation,
                showlegend=False,
                line=dict(color=line_color)
            )
        )

    deviation_fig.update_layout(
        # title="Time Series Deviation",
        xaxis_title="Time",
        yaxis_title="Deviation",
        showlegend=False,
        # Add range selector with limits
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=3, label="3d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            # Set range limits
            range=[start_time, end_time],
            autorange=False,
            minallowed=start_time,
            maxallowed=end_time,
        ),
        # Configure drag and scroll behavior
        dragmode='select',
    )

    # If we have a last selection, apply it to the new figure
    if spectrogram_time_window:
        deviation_fig.add_selection(
            x0=spectrogram_time_window[0],
            x1=spectrogram_time_window[1],
            y0=float(deviation.min()),
            y1=float(deviation.max()),
        )

    return deviation_fig

@app.callback(
    Output('spectral-probability-plot', 'figure', allow_duplicate=True),
    Input('spectral-probability-plot', 'relayoutData'),
    prevent_initial_call=True
)
def update_target_line_colors(drawn_target_lines):
    if not drawn_target_lines or 'shapes' not in drawn_target_lines:
        return dash.no_update

    shapes = drawn_target_lines['shapes']
    if not shapes:
        return dash.no_update

    # Determine color index based on number of shapes
    color_index = (len(shapes) - 1) % len(LINE_COLORS)
    next_color_index = len(shapes) % len(LINE_COLORS)

    # Update the color of the latest shape using Patch
    patched_figure = Patch()
    patched_figure['layout']['shapes'][-1]['line']['color'] = LINE_COLORS[color_index]
    patched_figure['layout']['newshape']['line']['color'] = LINE_COLORS[next_color_index]

    return patched_figure

@app.callback(
    [Output('spectrogram-time-window-store', 'data', allow_duplicate=True),
     Output('confirm-large-window', 'displayed')],
    [Input('deviation-plot', 'selectedData'),
     Input('confirm-large-window', 'submit_n_clicks')],
    State('deviation-plot', 'selectedData'),
    prevent_initial_call=True
)
def store_spectrogram_time_window(trigger, confirm_clicks, selected_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'confirm-large-window' and confirm_clicks:
        if "range" in selected_data:
            return selected_data["range"]["x"], False
        return dash.no_update, False

    if "range" in selected_data:
        time_range = selected_data["range"]["x"]
        start_time = time_range[0]
        end_time = time_range[1]
        time_window = uwacan.TimeWindow(
            start=min(start_time, end_time) + "Z",
            stop=max(start_time, end_time) + "Z",
        )

        # Check if time window is larger than 6 hours
        window_duration = time_window.duration / 3600
        if window_duration > 6:
            return dash.no_update, True
        else:
            return time_range, False
    else:
        return dash.no_update, False

@app.callback(
    Output('spectrogram-plot', 'figure'),
    [Input('spectrogram-time-window-store', 'data'),
     Input('file-selector', 'value'),
     Input('spectral-probability-plot', 'relayoutData')],
    prevent_initial_call=True
)
def plot_spectrogram(spectrogram_time_window, selected_file, drawn_target_lines):
    # If the file selector triggered the callback, return an empty figure
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'file-selector':
        return go.Figure()

    if not spectrogram_time_window:
        return go.Figure()

    # Check if there are any drawn lines
    if not drawn_target_lines or 'shapes' not in drawn_target_lines or not drawn_target_lines['shapes']:
        return go.Figure()

    # Get the range from the selection
    start_time = spectrogram_time_window[0]
    end_time = spectrogram_time_window[1]
    time_window = uwacan.TimeWindow(
        start=min(start_time, end_time) + "Z",
        stop=max(start_time, end_time) + "Z",
    )

    # Load and process spectrogram data
    selected_file = Path(selected_file)
    spectrogram = uwacan.load_data(selected_file / "spectrogram.zarr.zip")
    spectrogram = spectrogram.subwindow(time_window)

    # Create spectrogram figure
    fig = spectrogram.make_figure()
    fig.add_trace(spectrogram.plot())
    fig.update_layout(
        dragmode='select',
        xaxis=dict(
            type="date",
            range=[time_window.start.py_datetime(), time_window.stop.py_datetime()],
            autorange=False,
            minallowed=time_window.start.py_datetime(),
            maxallowed=time_window.stop.py_datetime(),
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="minute", stepmode="backward"),
                    dict(count=5, label="5m", step="minute", stepmode="backward"),
                    dict(count=15, label="15m", step="minute", stepmode="backward"),
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=4, label="4h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="hour", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
        ),
        yaxis=dict(
            domain=(0.0, 0.7),
            autorange=True,
            fixedrange=True,
        ),
        yaxis2=dict(
            domain=(0.7, 1.0),
            type="linear",
            title="Deviation",
        ),
    )

    # Process drawn lines if available
    shapes = drawn_target_lines['shapes']
    for i, shape in enumerate(shapes):
        # Extract x and y coordinates from the path string
        path_str = shape['path']
        points = path_str.replace('M', '').split('L')

        # Parse the points into x and y coordinates
        x_coords = []
        y_coords = []
        for point in points:
            x, y = map(float, point.split(','))
            x_coords.append(x)
            y_coords.append(y)

        # Create target DataArray from drawn line
        target = xr.DataArray(
            data=np.array(y_coords),
            dims=["frequency"],
            coords={"frequency": np.array(x_coords)},
        ).drop_duplicates("frequency").sortby("frequency")

        # Compute deviation using spectrogram data
        selected_spectrogram_in_target = spectrogram.data.sel(frequency=slice(target.frequency.min(), target.frequency.max()))
        target = target.interp(frequency=selected_spectrogram_in_target.frequency)
        selected_deviation = np.abs(uwacan.dB(selected_spectrogram_in_target) - target).mean("frequency")

        # Add deviation trace with matching color
        line_color = LINE_COLORS[i % len(LINE_COLORS)]
        fig.add_scatter(
            x=selected_deviation.time,
            y=selected_deviation,
            showlegend=False,
            line=dict(color=line_color),
            yaxis="y2",
        )

    return fig

@app.callback(
    [Output('audio-time-window-store', 'data'),
     Output('confirm-large-audio', 'displayed')],
    Input('spectrogram-plot', 'selectedData'),
    Input('confirm-large-audio', 'submit_n_clicks'),
    State('spectrogram-plot', 'selectedData'),
    prevent_initial_call=True
)
def store_audio_time_window(trigger, confirm_clicks, selected_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, False

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'confirm-large-audio' and confirm_clicks:
        if not selected_data or "range" not in selected_data:
            return dash.no_update, False
        # Continue with audio processing
    elif not selected_data or "range" not in selected_data:
        return dash.no_update, False

    # Get the range from the selection
    range_x = selected_data["range"]["x"]
    first = range_x[0]
    second = range_x[1]
    time_window = uwacan.TimeWindow(
        start=min(first, second) + "Z",
        stop=max(first, second) + "Z",
    )

    # Check if time window is larger than 1 minute
    window_duration = time_window.duration / 60
    if window_duration > 1 and trigger_id != 'confirm-large-audio':
        return dash.no_update, True

    return range_x, False

@app.callback(
    Output('audio-player', 'src'),
    [Input('audio-time-window-store', 'data'),
     Input('file-selector', 'value')],
    prevent_initial_call=True
)
def process_audio(audio_time_window, selected_file):
    if not audio_time_window:
        return dash.no_update

    # Get the range from the store
    first = audio_time_window[0]
    second = audio_time_window[1]
    time_window = uwacan.TimeWindow(
        start=min(first, second) + "Z",
        stop=max(first, second) + "Z",
    )

    selected_file = Path(selected_file)
    with (selected_file / "metadata.toml").open("rb") as f:
        log = tomllib.load(f)
    sensor = uwacan.sensor(
        log["label"],
        sensitivity=log["hardware"]["sensitivity"],
        position=log["deployment"]["position"],
        depth=log["deployment"]["hydrophone_depth"]
    )
    rec_name = log["hardware"]["recorder"]

    # Load the recording
    if "SoundTrap" in rec_name:
        recording_cls = uwacan.recordings.SoundTrap
    elif "RTsys" in rec_name:
        recording_cls = uwacan.recordings.SylenceLP
    recording_cls.allowable_interrupt = 60  # For this analysis we don't really care.
    recording = recording_cls.read_folder(selected_file / "timedata", sensor=sensor)
    recording = recording.subwindow(start=time_window.start, stop=time_window.stop)

    # Get audio data and normalize
    audio = recording.raw_data()
    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio)) * 2)

    # Create WAV file in memory
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            # Set parameters
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(recording.samplerate)  # Use recording's sample rate

            # Convert float32 to int16
            audio_int16 = (audio * 32767).astype(np.int16)

            # Write audio data
            wav_file.writeframes(audio_int16.tobytes())

        # Get the WAV data and encode it
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_src = f'data:audio/wav;base64,{audio_base64}'

    return audio_src

if __name__ == '__main__':
    app.run(debug=True)
