import io
import math
import random
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import streamlit as st

st.set_page_config(page_title="AFROGROOVE AI", page_icon="🥁", layout="wide")

SR = 44100
STEPS_PER_BAR = 16
GM_NOTES = {
    "Kick": 36,
    "Snare / Clap": 39,
    "Closed Hat": 42,
    "Open Hat": 46,
    "Shaker": 82,
    "Percussion": 75,
    "Log Drum": 45,
}

def analyze_sample(data):
    y, sr = librosa.load(io.BytesIO(data), sr=22050, mono=True)
    if len(y) < sr // 2:
        raise ValueError("Please upload an audio sample that is at least half a second long.")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    if not np.isfinite(tempo_value) or tempo_value < 35 or tempo_value > 240:
        tempo_value = 100.0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="frames")
    onset_count = int(len(onset_frames))
    duration = len(y) / sr
    return {
        "tempo": max(40.0, min(220.0, tempo_value)),
        "duration": duration,
        "onsets": onset_count,
        "sample_rate": sr,
    }

def choose_pattern(genre, feel, bars, density, swing, seed):
    rng = random.Random(seed)
    tracks = {name: [] for name in GM_NOTES}
    # Pattern templates use 16 steps per 4/4 bar. They are starting points, not fixed rules.
    if genre == "Afrobeats":
        kick_base = [0, 6, 10, 14]
        snare_base = [4, 12]
        hat_base = list(range(0, 16, 2))
        perc_base = [3, 7, 11, 15]
        log_base = []
    elif genre == "Amapiano":
        kick_base = [0, 8, 10]
        snare_base = [4, 12]
        hat_base = [2, 6, 10, 14]
        perc_base = [3, 7, 11, 15]
        log_base = [6, 14]
    elif genre == "Afro-house":
        kick_base = [0, 4, 8, 12]
        snare_base = [4, 12]
        hat_base = list(range(0, 16, 2))
        perc_base = [3, 7, 11, 15]
        log_base = []
    else:  # Afro-fusion
        kick_base = [0, 6, 8, 14]
        snare_base = [4, 12]
        hat_base = [2, 6, 10, 14]
        perc_base = [3, 7, 11, 15]
        log_base = []

    if feel == "Minimal":
        kick_base = kick_base[:max(2, len(kick_base)-1)]
        hat_base = hat_base[::2]
        perc_base = perc_base[::2]
        log_base = log_base[:1]
    elif feel == "High energy":
        hat_base = list(range(16))
        perc_base = sorted(set(perc_base + [5, 9, 13]))
        if genre == "Amapiano":
            log_base = sorted(set(log_base + [2, 10]))

    density_factor = density / 100.0
    for bar in range(bars):
        for step in kick_base:
            if rng.random() < 0.65 + 0.35 * density_factor:
                tracks["Kick"].append((bar * 16 + step, rng.randint(104, 122)))
        for step in snare_base:
            if rng.random() < 0.9:
                tracks["Snare / Clap"].append((bar * 16 + step, rng.randint(82, 108)))
        for step in hat_base:
            if rng.random() < (0.25 + 0.75 * density_factor):
                tracks["Closed Hat"].append((bar * 16 + step, rng.randint(48, 88)))
        for step in perc_base:
            if rng.random() < (0.2 + 0.75 * density_factor):
                tracks["Percussion"].append((bar * 16 + step, rng.randint(45, 88)))
        for step in log_base:
            if rng.random() < (0.35 + 0.6 * density_factor):
                tracks["Log Drum"].append((bar * 16 + step, rng.randint(80, 112)))
        # Light shaker layer; higher density means more subdivisions.
        shaker_steps = range(16) if density >= 70 else range(0, 16, 2)
        for step in shaker_steps:
            if rng.random() < 0.35 * density_factor:
                tracks["Shaker"].append((bar * 16 + step, rng.randint(35, 66)))
        # Open hats are occasional accents, not constant.
        for step in [6, 14]:
            if rng.random() < 0.25 * density_factor:
                tracks["Open Hat"].append((bar * 16 + step, rng.randint(55, 82)))

    # Apply controlled swing to off-grid 16th-step events, while preserving the bar grid.
    # Swing is represented by shifting odd 16th subdivisions later by a fraction of a 16th.
    return tracks

def events_to_midi(tracks, bpm, swing, bars):
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    seconds_per_beat = 60.0 / bpm
    step_seconds = seconds_per_beat / 4.0
    swing_delay = (swing / 100.0) * step_seconds * 0.42
    for name, events in tracks.items():
        inst = pretty_midi.Instrument(program=0, is_drum=True, name=name)
        for step, velocity in events:
            start = step * step_seconds
            if step % 2 == 1:
                start += swing_delay
            end = start + min(0.12, step_seconds * 0.7)
            inst.notes.append(pretty_midi.Note(
                velocity=int(max(1, min(127, velocity))),
                pitch=GM_NOTES[name],
                start=float(start),
                end=float(end),
            ))
        midi.instruments.append(inst)
    return midi

def synth_kick(t, vel):
    f = 48 + 75 * np.exp(-t * 28)
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase) * np.exp(-t * 12)
    click = np.random.default_rng(4).normal(0, 1, len(t)) * np.exp(-t * 150) * 0.12
    return (body + click) * (vel / 127) * 0.8

def synth_snare(t, vel):
    rng = np.random.default_rng(12)
    noise = rng.normal(0, 1, len(t))
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 32)
    return (noise * np.exp(-t * 24) * 0.42 + tone * 0.25) * (vel / 127) * 0.65

def synth_hat(t, vel, open_hat=False, seed=23):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(t))
    decay = 12 if open_hat else 75
    # A simple high-pass-ish difference emphasizes bright noise.
    bright = np.concatenate([[0.0], np.diff(noise)])
    return bright * np.exp(-t * decay) * (vel / 127) * (0.14 if open_hat else 0.11)

def synth_log(t, vel):
    f = 115 + 100 * np.exp(-t * 18)
    phase = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(phase) * np.exp(-t * 9) * (vel / 127) * 0.55

def render_wav(tracks, bpm, swing, bars):
    beat = 60.0 / bpm
    step = beat / 4.0
    total_seconds = bars * 4 * beat
    audio = np.zeros(int(math.ceil(total_seconds * SR)), dtype=np.float32)
    rng = np.random.default_rng(99)
    for name, events in tracks.items():
        for idx, velocity in events:
            start = idx * step
            if idx % 2 == 1:
                start += (swing / 100.0) * step * 0.42
            if name == "Kick":
                dur = 0.42
            elif name == "Snare / Clap":
                dur = 0.28
            elif name == "Closed Hat":
                dur = 0.10
            elif name == "Open Hat":
                dur = 0.34
            elif name == "Shaker":
                dur = 0.13
            elif name == "Log Drum":
                dur = 0.48
            else:
                dur = 0.20
            n = min(int(dur * SR), len(audio) - int(start * SR))
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float32) / SR
            if name == "Kick":
                sound = synth_kick(t, velocity)
            elif name == "Snare / Clap":
                sound = synth_snare(t, velocity)
            elif name == "Closed Hat":
                sound = synth_hat(t, velocity, False, int(idx + 1))
            elif name == "Open Hat":
                sound = synth_hat(t, velocity, True, int(idx + 3))
            elif name == "Shaker":
                sound = synth_hat(t, velocity, False, int(idx + 7)) * 0.7
            elif name == "Log Drum":
                sound = synth_log(t, velocity)
            else:
                sound = synth_hat(t, velocity, False, int(idx + 11)) * 0.8
            offset = int(start * SR)
            audio[offset:offset+n] += sound[:n].astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / max(1.0, peak / 0.92)
    return audio

st.title("🥁 AFROGROOVE AI")
st.caption("Starter prototype • Sample-aware genre selection • MIDI + WAV export")
st.warning(
    "Prototype note: tempo/onset analysis is real, but genre-aware pattern generation is rule-based. "
    "It does not yet understand harmony or separate existing drums from a mixed sample. WAV output uses "
    "simple synthesized drum sounds, not a commercial sample library."
)

uploaded = st.file_uploader("Load a WAV, MP3, AIFF or other librosa-supported audio file", type=["wav", "mp3", "aiff", "aif", "flac", "ogg"])
left, right = st.columns(2)
with left:
    genre = st.selectbox("Target genre", ["Afrobeats", "Amapiano", "Afro-house", "Afro-fusion"])
    feel = st.selectbox("Groove feel", ["Main groove", "Minimal", "High energy"])
    bars = st.selectbox("Pattern length", [1, 2, 4, 8], index=2)
with right:
    bpm_manual = st.number_input("Tempo (BPM)", min_value=40, max_value=220, value=105, step=1)
    use_detected = st.checkbox("Use detected tempo after analysis", value=True)
    density = st.slider("Pattern density", 0, 100, 60)
    swing = st.slider("Swing", 0, 100, 22)
seed = st.number_input("Variation seed", min_value=1, max_value=999999, value=42, step=1)

sample_info = None
if uploaded:
    try:
        sample_info = analyze_sample(uploaded.getvalue())
        st.success(
            f"Sample loaded • Duration: {sample_info['duration']:.2f}s • "
            f"Estimated tempo: {sample_info['tempo']:.1f} BPM • "
            f"Detected onsets: {sample_info['onsets']}"
        )
        st.audio(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not analyze this file: {e}")

if st.button("Generate drum pattern", type="primary", use_container_width=True):
    bpm = float(sample_info["tempo"] if uploaded and sample_info and use_detected else bpm_manual)
    tracks = choose_pattern(genre, feel, bars, density, swing, int(seed))
    midi = events_to_midi(tracks, bpm, swing, bars)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        midi.write(tmp.name)
        midi_path = tmp.name
    midi_data = Path(midi_path).read_bytes()
    Path(midi_path).unlink(missing_ok=True)

    wav = render_wav(tracks, bpm, swing, bars)
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav, SR, format="WAV", subtype="PCM_24")
    wav_data = wav_buffer.getvalue()

    st.session_state["result"] = {
        "bpm": bpm, "tracks": tracks, "midi": midi_data, "wav": wav_data,
        "genre": genre, "bars": bars,
    }

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Generated groove")
    st.write(f"{result['genre']} • {result['bpm']:.1f} BPM • {result['bars']} bars")
    st.audio(result["wav"], format="audio/wav")
    st.download_button("Download MIDI", data=result["midi"], file_name="afrogroove_pattern.mid", mime="audio/midi", use_container_width=True)
    st.download_button("Download drum-loop WAV", data=result["wav"], file_name="afrogroove_drums.wav", mime="audio/wav", use_container_width=True)
    st.markdown("#### Track event counts")
    st.write({name: len(events) for name, events in result["tracks"].items()})
    st.info("Import the MIDI into FL Studio and assign your own drum samples. The WAV is a separate synthesized drum loop; it is not mixed with your uploaded sample.")
