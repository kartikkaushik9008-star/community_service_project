from __future__ import annotations

import hashlib
from datetime import datetime

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from audio_classification import (
    ClassifierUnavailableError,
    load_esc50_model,
    load_yamnet,
    predict_esc50,
    predict_yamnet,
)
from audio_processing import (
    AudioProcessingError,
    apply_audio_effects,
    decode_audio,
    encode_wav,
    mix_audio_tracks,
)


st.set_page_config(
    page_title="Auralis | Audio intelligence",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .auralis-hero {
        padding: 1.7rem 2rem;
        margin: 0.3rem 0 1.2rem;
        border: 1px solid rgba(95, 220, 245, .25);
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(17, 40, 62, .96), rgba(38, 25, 67, .92));
        box-shadow: 0 12px 40px rgba(0, 0, 0, .22);
    }
    .auralis-kicker {
        color: #70e5f4;
        letter-spacing: .14em;
        text-transform: uppercase;
        font-size: .75rem;
        font-weight: 700;
    }
    .auralis-hero h1 {
        margin: .35rem 0;
        font-size: clamp(1.8rem, 3.2vw, 2.8rem);
    }
    .auralis-hero p {
        margin: 0;
        color: #c1ccdc;
        max-width: 760px;
    }
    .auralis-note {
        border-left: 3px solid #65ddeb;
        padding: .65rem .9rem;
        background: rgba(37, 67, 87, .35);
        border-radius: 0 10px 10px 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(19, 29, 43, .64);
        backdrop-filter: blur(10px);
        border-color: rgba(101, 221, 235, .18);
        transition: border-color .2s ease, transform .2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(101, 221, 235, .38);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PAGES = [
    "Home dashboard",
    "AI sound detection",
    "Environmental monitoring",
    "Audio enhancement studio",
    "VR music studio",
    "SDG impact",
]
AUDIO_TYPES = ["wav", "mp3", "ogg", "flac"]


@st.cache_resource(show_spinner="Loading the original ESC-50 model...")
def cached_esc50_model():
    return load_esc50_model()


@st.cache_resource(show_spinner="Loading YAMNet from TensorFlow Hub...")
def cached_yamnet():
    return load_yamnet()


@st.cache_data(show_spinner=False, max_entries=8)
def decode_cached(audio_bytes: bytes, filename: str):
    decoded = decode_audio(audio_bytes, filename)
    return decoded.samples, decoded.sample_rate, decoded.source_sample_rate


def audio_fingerprint(filename: str, data: bytes) -> str:
    return hashlib.sha256(filename.encode("utf-8") + data).hexdigest()


def hero(title: str, description: str, kicker: str = "AURALIS · AUDIO INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="auralis-hero">
            <div class="auralis-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def themed_figure(figure: go.Figure, height: int = 300) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c5cfdf"},
        height=height,
        margin={"l": 22, "r": 18, "t": 32, "b": 35},
    )
    return figure


def render_audio_visuals(samples: np.ndarray, sample_rate: int):
    st.subheader("Signal observatory")
    waveform_tab, spectrum_tab, mel_tab = st.tabs(
        ["Waveform", "Frequency spectrum", "Mel spectrogram"]
    )
    with waveform_tab:
        stride = max(1, samples.size // 6000)
        plotted = samples[::stride]
        seconds = np.arange(plotted.size) * stride / sample_rate
        figure = go.Figure(
            go.Scattergl(
                x=seconds,
                y=plotted,
                mode="lines",
                line={"color": "#55d9e8", "width": 1},
                name="Amplitude",
            )
        )
        figure.update_layout(title="Amplitude over time", xaxis_title="Seconds")
        st.plotly_chart(themed_figure(figure, 260), width="stretch")
    with spectrum_tab:
        fft_size = min(65536, max(2048, 2 ** int(np.ceil(np.log2(samples.size)))))
        spectrum = np.abs(np.fft.rfft(samples, n=fft_size))
        frequencies = np.fft.rfftfreq(fft_size, 1 / sample_rate)
        figure = go.Figure(
            go.Scatter(
                x=frequencies,
                y=20 * np.log10(np.maximum(spectrum, 1e-8)),
                mode="lines",
                line={"color": "#a98bff", "width": 1.2},
            )
        )
        figure.update_layout(
            title="Frequency energy", xaxis_title="Hz", yaxis_title="Relative dB"
        )
        st.plotly_chart(themed_figure(figure, 260), width="stretch")
    with mel_tab:
        mel = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=64
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        figure = go.Figure(
            go.Heatmap(
                z=mel_db,
                x=librosa.frames_to_time(
                    np.arange(mel_db.shape[1]), sr=sample_rate, hop_length=256
                ),
                y=librosa.mel_frequencies(n_mels=64, fmin=0, fmax=sample_rate / 2),
                colorscale=[
                    [0, "#111827"],
                    [0.45, "#542b87"],
                    [0.78, "#306eaa"],
                    [1, "#69e5ef"],
                ],
                colorbar={"title": "dB"},
            )
        )
        figure.update_layout(
            title="Mel-frequency energy", xaxis_title="Seconds", yaxis_title="Hz"
        )
        st.plotly_chart(themed_figure(figure, 330), width="stretch")


def render_predictions(result: dict, model_name: str):
    predictions = result["predictions"]
    if result["low_confidence"]:
        st.warning(
            "Low confidence or close competing scores: treat these as candidate "
            "sound classes, not confirmed detections."
        )
    else:
        st.caption(
            "Model scores are relative estimates, not calibrated probabilities "
            "or proof that an event occurred."
        )
    top = predictions[0]
    metrics = st.columns(3)
    metrics[0].metric("Leading candidate", top["label"].replace("_", " ").title())
    metrics[1].metric("Model score", f"{top['confidence']:.1%}")
    metrics[2].metric("Inference time", f"{result['analysis_seconds']:.2f} s")
    chart = go.Figure(
        go.Bar(
            x=[item["confidence"] * 100 for item in predictions],
            y=[item["label"].replace("_", " ").title() for item in predictions],
            orientation="h",
            marker={"color": ["#65ddeb", "#8675fa", "#5a9fe8", "#7163bb", "#516175"]},
            text=[f"{item['confidence']:.1%}" for item in predictions],
            textposition="outside",
        )
    )
    chart.update_layout(
        title="Top five class scores",
        xaxis_title="Relative model score (%)",
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(themed_figure(chart, 320), width="stretch")
    if result["segments"]:
        timeline = go.Figure(
            go.Scatter(
                x=[item["start_seconds"] for item in result["segments"]],
                y=[item["label"].replace("_", " ").title() for item in result["segments"]],
                mode="markers",
                marker={
                    "size": 14,
                    "color": [
                        item["confidence"] for item in result["segments"]
                    ],
                    "colorscale": [[0, "#6753a8"], [1, "#65ddeb"]],
                    "showscale": True,
                    "colorbar": {"title": "Score"},
                },
                customdata=[
                    [item["confidence"]] for item in result["segments"]
                ],
                hovertemplate=(
                    "%{y}<br>Start: %{x:.2f} s"
                    "<br>Score: %{customdata[0]:.1%}<extra></extra>"
                ),
            )
        )
        timeline.update_layout(
            title=f"Segment event timeline · {model_name}",
            xaxis_title="Segment start (seconds)",
            yaxis_title="Top segment candidate",
        )
        st.plotly_chart(themed_figure(timeline, 260), width="stretch")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rank": index + 1,
                    "Candidate sound": item["label"].replace("_", " ").title(),
                    "Model score": f"{item['confidence']:.1%}",
                }
                for index, item in enumerate(predictions)
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def add_history(filename: str, result: dict, model_name: str, duration: float):
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "sound": result["predictions"][0]["label"],
        "score": result["predictions"][0]["confidence"],
        "model": model_name,
        "duration": duration,
        "segments": len(result["segments"]),
    }
    st.session_state.analysis_history.insert(0, record)
    st.session_state.analysis_history = st.session_state.analysis_history[:50]


def home_dashboard():
    history = st.session_state.analysis_history
    unique_categories = len({entry["sound"] for entry in history})
    hero(
        "Listen deeper. Understand your soundscape.",
        "An audio analysis and augmentation workspace for environmental sound "
        "awareness, learning, and future immersive music production.",
    )
    metrics = st.columns(3)
    metrics[0].metric("Audio files analyzed", len(history), border=True)
    metrics[1].metric("Detected sound categories", unique_categories, border=True)
    metrics[2].metric("Model options", "Local CNN + optional YAMNet")

    left, right = st.columns([1.25, 1])
    with left:
        with st.container(border=True):
            st.subheader("Recent analysis")
            if history:
                st.dataframe(
                    pd.DataFrame(history[:8]).rename(
                        columns={
                            "time": "Analyzed",
                            "file": "Audio file",
                            "sound": "Top candidate",
                            "score": "Score",
                            "model": "Backend",
                            "duration": "Duration (s)",
                            "segments": "Segments",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption(
                    "Your analysis history is session-only. Analyze a recording "
                    "to populate these metrics."
                )
    with right:
        with st.container(border=True):
            st.subheader("Community-service focus")
            st.write(
                "Explore how environmental audio can support learning, "
                "community soundscape awareness, and wildlife listening "
                "activities. The project demonstrates tools; it does not claim "
                "measured conservation or public-health outcomes."
            )
            st.caption("Aligned with SDGs 4, 9, 11, and 15.")
    st.subheader("Workspace")
    st.write(
        "Use the navigation in the sidebar to analyze WAV/MP3 recordings, "
        "monitor environmental categories, shape audio, or compose a multi-track "
        "VR-ready mix."
    )


def sound_detection():
    hero(
        "AI sound detection",
        "Upload a recording or capture a short sound in your browser. "
        "The signal is converted to mono, resampled, segmented, and scored.",
    )
    model_choice = st.selectbox(
        "Classification backend",
        ["ESC-50 CNN · local checkpoint", "YAMNet · 521 AudioSet classes"],
        help=(
            "YAMNet requires optional TensorFlow packages and internet access "
            "for its first model download."
        ),
    )
    upload_tab, record_tab = st.tabs(["Upload audio", "Record audio"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=AUDIO_TYPES,
            help="WAV and MP3 are supported; OGG and FLAC are also accepted.",
            key="analysis_upload",
        )
    with record_tab:
        recording = st.audio_input("Record a sound using your microphone")

    source = uploaded if uploaded is not None else recording
    if source is None:
        st.info("Upload an audio file or record a clip to begin.")
        return
    filename = source.name if uploaded is not None else "microphone_recording.wav"
    audio_bytes = source.getvalue()
    try:
        samples, sample_rate, source_rate = decode_cached(audio_bytes, filename)
    except AudioProcessingError as exc:
        st.error(str(exc))
        return
    duration = samples.size / sample_rate
    st.audio(audio_bytes, format=source.type or "audio/wav")
    info = st.columns(3)
    info[0].metric("Duration", f"{duration:.2f} s")
    info[1].metric("Original sample rate", f"{source_rate:,} Hz")
    info[2].metric("Analysis format", f"Mono · {sample_rate:,} Hz")
    render_audio_visuals(samples, sample_rate)

    if st.button("Run sound analysis", type="primary", width="stretch"):
        try:
            with st.spinner("Scoring audio segments..."):
                threshold = st.session_state.confidence_threshold
                if model_choice.startswith("ESC-50"):
                    model, labels = cached_esc50_model()
                    result = predict_esc50(
                        samples, sample_rate, model, labels, threshold
                    )
                    model_name = "ESC-50 CNN"
                else:
                    model, labels = cached_yamnet()
                    result = predict_yamnet(
                        samples,
                        sample_rate,
                        model,
                        labels,
                        threshold,
                    )
                    model_name = "YAMNet"
            fingerprint = audio_fingerprint(filename, audio_bytes)
            st.session_state.latest_analysis = {
                "fingerprint": fingerprint,
                "filename": filename,
                "model": model_name,
                "result": result,
                "duration": duration,
            }
            add_history(filename, result, model_name, duration)
        except ClassifierUnavailableError as exc:
            st.error(str(exc))
        except (AudioProcessingError, ValueError, RuntimeError) as exc:
            st.error(f"Could not analyze this recording: {exc}")

    latest = st.session_state.latest_analysis
    if latest and latest["fingerprint"] == audio_fingerprint(filename, audio_bytes):
        st.subheader("Classification result")
        st.caption(f"Backend: {latest['model']} · Not a definitive identification.")
        render_predictions(latest["result"], latest["model"])
    if duration < 1.0:
        st.caption(
            "Short recordings contain limited context and may produce especially "
            "uncertain scores."
        )


def _environment_group(label: str) -> str:
    normalized = label.lower().replace("_", " ")
    if any(word in normalized for word in ("bird", "crow", "rooster", "hen")):
        return "Bird sounds"
    if any(
        word in normalized
        for word in ("dog", "cat", "pig", "cow", "frog", "sheep", "animal")
    ):
        return "Animal sounds"
    if any(
        word in normalized
        for word in ("car", "engine", "siren", "train", "airplane", "helicopter")
    ):
        return "Traffic and transport"
    if any(word in normalized for word in ("rain", "thunder", "wind", "wave", "water")):
        return "Weather and water"
    if any(
        word in normalized
        for word in ("people", "crowd", "footsteps", "laugh", "cough", "clap")
    ):
        return "Human activity"
    if any(word in normalized for word in ("saw", "construction", "drill")):
        return "Construction and tools"
    return "Other sound classes"


def environmental_monitoring():
    hero(
        "Environmental sound monitoring",
        "Review model estimates as prompts for listening and community "
        "soundscape activities—not as verified measurements.",
    )
    st.warning(
        "Predictions are AI estimates. Do not treat them as definitive wildlife "
        "species identification, an acoustic survey, or an environmental "
        "compliance measurement."
    )
    history = st.session_state.analysis_history
    if not history:
        st.info("Analyze recordings first; monitoring statistics use actual session results.")
        return
    frame = pd.DataFrame(history)
    frame["environment_group"] = frame["sound"].map(_environment_group)
    group_counts = (
        frame["environment_group"].value_counts().rename_axis("Sound category").reset_index(name="Analyses")
    )
    st.subheader("Session analysis statistics")
    st.caption("Counts below are analyses in this browser session, not ambient prevalence.")
    st.bar_chart(
        group_counts,
        x="Sound category",
        y="Analyses",
        color="#58cfe3",
        alt="Counts of analyzed clips grouped by their top candidate sound class",
    )
    st.dataframe(
        frame[["time", "file", "sound", "score", "model"]].rename(
            columns={
                "time": "Analyzed",
                "file": "Audio file",
                "sound": "Top candidate",
                "score": "Model score",
                "model": "Backend",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Categories include birds and animals, transport, weather and water, "
        "human activity, construction/tools, and other model classes."
    )


def audio_enhancement():
    hero(
        "Audio enhancement studio",
        "Shape a copy of your recording with level, echo, reverb, pitch, speed, "
        "and filter controls. Your uploaded source remains unchanged.",
    )
    uploaded = st.file_uploader(
        "Load a source recording",
        type=AUDIO_TYPES,
        key="enhancement_upload",
    )
    if uploaded is None:
        st.info("Upload a WAV, MP3, OGG, or FLAC recording to open the effect rack.")
        return
    try:
        samples, sample_rate, _ = decode_cached(uploaded.getvalue(), uploaded.name)
    except AudioProcessingError as exc:
        st.error(str(exc))
        return
    st.audio(uploaded.getvalue(), format=uploaded.type or "audio/wav")
    with st.form("audio_effects_form"):
        first, second = st.columns(2)
        with first:
            gain_db = st.slider("Volume · gain (dB)", -18.0, 18.0, 0.0, 0.5)
            echo = st.slider("Echo amount", 0.0, 0.8, 0.0, 0.05)
            reverb = st.slider("Reverb amount", 0.0, 0.8, 0.0, 0.05)
        with second:
            pitch = st.slider("Pitch shift (semitones)", -12.0, 12.0, 0.0, 1.0)
            speed = st.slider("Playback speed", 0.5, 2.0, 1.0, 0.05)
            filter_type = st.selectbox(
                "Audio filter", ["Off", "Low-pass", "High-pass", "Band-pass"]
            )
        submitted = st.form_submit_button(
            "Apply effects to a copy", type="primary", width="stretch"
        )
    fingerprint = audio_fingerprint(uploaded.name, uploaded.getvalue())
    if submitted:
        try:
            processed = apply_audio_effects(
                samples,
                sample_rate,
                gain_db=gain_db,
                echo=echo,
                reverb=reverb,
                pitch_steps=pitch,
                speed=speed,
                filter_type=filter_type,
            )
            st.session_state.processed_audio = {
                "fingerprint": fingerprint,
                "samples": processed,
                "sample_rate": sample_rate,
            }
        except (AudioProcessingError, ValueError) as exc:
            st.error(f"Could not apply these effects: {exc}")
    processed = st.session_state.processed_audio
    if processed and processed["fingerprint"] == fingerprint:
        wav = encode_wav(processed["samples"], processed["sample_rate"])
        st.subheader("Processed preview")
        st.audio(wav, format="audio/wav")
        st.download_button(
            "Download processed WAV",
            data=wav,
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_enhanced.wav",
            mime="audio/wav",
            type="primary",
        )
    with st.expander("Mix multiple recordings"):
        render_audio_mixer(
            "enhancement_mix",
            "Layer tracks for a combined WAV preview.",
            "auralis_enhancement_mix.wav",
        )


def render_audio_mixer(key_prefix: str, description: str, export_name: str):
    st.caption(description)
    files = st.file_uploader(
        "Add audio tracks",
        type=AUDIO_TYPES,
        accept_multiple_files=True,
        key=f"{key_prefix}_tracks",
        help="All tracks are resampled to mono 22.05 kHz before mixing.",
    )
    if not files:
        st.caption("Upload two or more recordings to create a layered mix.")
        return
    track_samples = []
    volumes = []
    names = []
    for index, audio_file in enumerate(files):
        try:
            samples, _, _ = decode_cached(audio_file.getvalue(), audio_file.name)
        except AudioProcessingError as exc:
            st.error(f"{audio_file.name}: {exc}")
            return
        track_samples.append(samples)
        names.append(audio_file.name)
        volumes.append(
            st.slider(
                f"Track {index + 1} · {audio_file.name}",
                0.0,
                1.0,
                0.7,
                0.05,
                key=f"{key_prefix}_volume_{index}_{audio_file.name}",
            )
        )
    mix_state_key = f"{key_prefix}_mix"
    names_state_key = f"{key_prefix}_mix_names"
    st.session_state.setdefault(mix_state_key, None)
    st.session_state.setdefault(names_state_key, [])
    if st.button(
        "Render track mix",
        type="primary",
        width="stretch",
        key=f"{key_prefix}_render",
    ):
        try:
            st.session_state[mix_state_key] = mix_audio_tracks(track_samples, volumes)
            st.session_state[names_state_key] = names
        except AudioProcessingError as exc:
            st.error(f"Could not render this mix: {exc}")
    mix = st.session_state[mix_state_key]
    if mix is not None:
        wav = encode_wav(mix, 22050)
        st.subheader("Mix preview")
        st.caption(f"Tracks: {', '.join(st.session_state[names_state_key])}")
        st.audio(wav, format="audio/wav")
        st.download_button(
            "Export mix as WAV",
            data=wav,
            file_name=export_name,
            mime="audio/wav",
            type="primary",
            key=f"{key_prefix}_download",
        )


def vr_music_studio():
    hero(
        "VR music studio",
        "Layer environmental recordings into a spatial-audio-ready stereo or "
        "mono mix. Track levels are independent; export is a standard WAV for "
        "later Unity or Unreal Engine import.",
    )
    st.info(
        "Try birds + wind + rain, traffic + crowd + footsteps, or waves + "
        "wind + seagulls."
    )
    render_audio_mixer(
        "vr_studio",
        "Each track has an independent level; export a standard WAV for a future game-engine workflow.",
        "auralis_vr_environment_mix.wav",
    )


def sdg_impact():
    hero(
        "Sound for shared progress",
        "A community-service learning platform that connects audio AI, "
        "environmental listening, and accessible creative technology.",
        "SDG IMPACT · LEARNING & AWARENESS",
    )
    goals = [
        (
            "SDG 4 · Quality education",
            "Use interactive sound analysis and visualizations to support "
            "hands-on learning about audio signals, machine learning, and "
            "responsible interpretation of AI results.",
        ),
        (
            "SDG 9 · Industry, innovation and infrastructure",
            "Demonstrate an extensible Python audio workflow that connects "
            "preprocessing, alternative model backends, analysis, and creative "
            "audio tools.",
        ),
        (
            "SDG 11 · Sustainable cities and communities",
            "Invite students and community groups to explore everyday "
            "soundscapes and discuss noise awareness using recordings they "
            "choose to analyze.",
        ),
        (
            "SDG 15 · Life on land",
            "Support educational listening activities around birds and other "
            "animal sounds while making clear that automated labels are not "
            "verified species observations.",
        ),
    ]
    for title, description in goals:
        with st.container(border=True):
            st.subheader(title)
            st.write(description)
    st.info(
        "These are intended educational and awareness contributions. The "
        "project does not claim measured improvements in conservation, urban "
        "noise, education outcomes, or community health."
    )


st.session_state.setdefault("analysis_history", [])
st.session_state.setdefault("latest_analysis", None)
st.session_state.setdefault("processed_audio", None)
st.session_state.setdefault("confidence_threshold", 0.60)

with st.sidebar:
    st.markdown("## Auralis")
    st.caption("Audio-event analysis & augmentation")
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.slider(
        "Low-confidence warning threshold",
        min_value=0.10,
        max_value=0.95,
        step=0.05,
        key="confidence_threshold",
        help="An alert threshold only; model scores are not calibrated probabilities.",
    )
    st.caption("ESC-50 · 50 output classes")
    st.caption("YAMNet · optional · 521 AudioSet classes")
    st.caption("Analysis history is kept in this browser session only.")

if page == "Home dashboard":
    home_dashboard()
elif page == "AI sound detection":
    sound_detection()
elif page == "Environmental monitoring":
    environmental_monitoring()
elif page == "Audio enhancement studio":
    audio_enhancement()
elif page == "VR music studio":
    vr_music_studio()
else:
    sdg_impact()
