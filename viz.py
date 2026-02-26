import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import soundfile as sf
from basic_pitch.inference import predict
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt


@dataclass
class Band:
    name: str
    y0: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ensure_audio(audio_path: Path, sr: int, cache_dir: Path) -> Tuple[np.ndarray, int, float]:
    wav_path = cache_dir / f"mix_{sr}.wav"
    if not wav_path.exists():
        data, in_sr = sf.read(str(audio_path), always_2d=True)
        if in_sr != sr:
            try:
                import librosa
            except ImportError as exc:
                raise RuntimeError("librosa is required for resampling when sample rates differ") from exc
            data = librosa.resample(data.T, orig_sr=in_sr, target_sr=sr).T
        sf.write(str(wav_path), data, sr)
    audio, out_sr = sf.read(str(wav_path), always_2d=True)
    duration = len(audio) / out_sr
    return audio.astype(np.float32), out_sr, duration


def run_demucs(audio_wav: Path, cache_dir: Path) -> Dict[str, Path]:
    stems_dir = cache_dir / "stems"
    model_name = "htdemucs"
    if not stems_dir.exists():
        stems_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            model_name,
            "--out",
            str(stems_dir),
            str(audio_wav),
        ]
        subprocess.run(cmd, check=True)

    base_name = audio_wav.stem
    candidate = stems_dir / model_name / base_name
    if not candidate.exists():
        matches = list(stems_dir.glob(f"**/{base_name}"))
        if not matches:
            raise FileNotFoundError("Could not locate Demucs output directory")
        candidate = matches[0]

    stems = {
        "drums": candidate / "drums.wav",
        "bass": candidate / "bass.wav",
        "vocals": candidate / "vocals.wav",
        "other": candidate / "other.wav",
    }
    for stem, path in stems.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing stem {stem}: {path}")
    return stems


def rms_db_per_frame(sig: np.ndarray, hop: int, n_frames: int, eps: float = 1e-8) -> np.ndarray:
    out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = sig[i * hop : (i + 1) * hop]
        if frame.size == 0:
            break
        rms = np.sqrt(np.mean(frame**2))
        out[i] = 20.0 * np.log10(rms + eps)
    return out


def robust_normalize_db(db: np.ndarray) -> np.ndarray:
    p5 = np.percentile(db, 5)
    p95 = np.percentile(db, 95)
    if p95 <= p5:
        return np.zeros_like(db)
    norm = (db - p5) / (p95 - p5)
    return np.clip(norm, 0.0, 1.0)


def smooth_1pole(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def bandpass(sig: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, sig)


def highpass(sig: np.ndarray, sr: int, low_hz: float) -> np.ndarray:
    sos = butter(4, low_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, sig)


def midi_from_contour_bins(n_bins: int) -> np.ndarray:
    # Basic Pitch contour uses 3 bins per semitone from MIDI 21..109 (exclusive upper edge for 264 bins).
    return 21.0 + np.arange(n_bins, dtype=np.float32) / 3.0


def run_basic_pitch(stem_path: Path, cache_dir: Path) -> Dict[str, np.ndarray]:
    cache_path = cache_dir / f"basic_pitch_{stem_path.stem}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {"contour": data["contour"], "time": data["time"]}

    model_output, _, _ = predict(str(stem_path))
    if "contour" not in model_output:
        raise RuntimeError("Basic Pitch output does not contain contour")

    contour = model_output["contour"].astype(np.float32)
    if contour.ndim != 2:
        raise RuntimeError(f"Unexpected contour shape: {contour.shape}")

    stem_audio, stem_sr = sf.read(str(stem_path), always_2d=True)
    stem_duration = len(stem_audio) / stem_sr
    t = np.linspace(0.0, stem_duration, contour.shape[0], endpoint=False, dtype=np.float32)

    np.savez_compressed(cache_path, contour=contour, time=t)
    return {"contour": contour, "time": t}


def resample_contour_to_master(contour: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if contour.shape[0] < 2:
        return np.zeros((len(dst_t), contour.shape[1]), dtype=np.float32)
    f = interp1d(src_t, contour, axis=0, bounds_error=False, fill_value=0.0)
    return f(dst_t).astype(np.float32)


def select_midi_range(contour: np.ndarray, midi_min: float, midi_max: float) -> np.ndarray:
    midi = midi_from_contour_bins(contour.shape[1])
    sel = (midi >= midi_min) & (midi <= midi_max)
    if not np.any(sel):
        return np.zeros((contour.shape[0], 1), dtype=np.float32)
    return contour[:, sel]


def compute_bands(height: int) -> Dict[str, Band]:
    ratios = [0.10, 0.25, 0.15, 0.25, 0.10, 0.10]
    names = ["loudness", "vocals", "bass", "other", "kick", "drums_other"]
    sep = 1
    total_sep = sep * (len(ratios) - 1)
    usable = max(1, height - total_sep)
    raw_heights = [r * usable for r in ratios]
    heights = [max(1, int(round(h))) for h in raw_heights]

    delta = usable - sum(heights)
    i = 0
    while delta != 0:
        idx = i % len(heights)
        if delta > 0:
            heights[idx] += 1
            delta -= 1
        elif heights[idx] > 1:
            heights[idx] -= 1
            delta += 1
        i += 1

    bands: Dict[str, Band] = {}
    y = 0
    for name, h in zip(names, heights):
        bands[name] = Band(name=name, y0=y, y1=y + h)
        y += h + sep
    return bands


def draw_value_dot(img: np.ndarray, band: Band, value: float, x: int = 0) -> None:
    v = float(np.clip(value, 0.0, 1.0))
    y = band.y1 - 1 - int(round(v * (band.height - 1)))
    y = int(np.clip(y, band.y0, band.y1 - 1))
    img[y, x] = (255, 255, 255)
    if y > band.y0:
        img[y - 1, x] = (200, 200, 200)
    if y < band.y1 - 1:
        img[y + 1, x] = (200, 200, 200)


def draw_saliency_column(img: np.ndarray, band: Band, sal: np.ndarray, gamma: float = 0.5, x: int = 0) -> None:
    if sal.size == 0:
        return
    h = band.height
    if h <= 1:
        return
    src_y = np.linspace(0, sal.size - 1, h)
    resampled = np.interp(src_y, np.arange(sal.size), sal)
    values = np.power(np.clip(resampled, 0.0, 1.0), gamma)
    brightness = np.clip(np.round(values * 255.0), 0, 255).astype(np.uint8)
    for i in range(h):
        y = band.y1 - 1 - i
        b = int(brightness[i])
        img[y, x] = (b, b, b)


def render_video(
    out_video: Path,
    width: int,
    height: int,
    fps: int,
    times: np.ndarray,
    loudness: np.ndarray,
    vocals: np.ndarray,
    bass: np.ndarray,
    other: np.ndarray,
    kick: np.ndarray,
    drums_other: np.ndarray,
) -> Path:
    temp_video = out_video.with_name(out_video.stem + "_silent.mp4")
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open cv2.VideoWriter")

    bands = compute_bands(height)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(times)

    for i in range(n):
        img[:, 1:, :] = img[:, :-1, :]
        img[:, 0, :] = 0

        draw_value_dot(img, bands["loudness"], loudness[i], x=0)
        draw_saliency_column(img, bands["vocals"], vocals[i], x=0)
        draw_saliency_column(img, bands["bass"], bass[i], x=0)
        draw_saliency_column(img, bands["other"], other[i], x=0)
        draw_value_dot(img, bands["kick"], kick[i], x=0)
        draw_value_dot(img, bands["drums_other"], drums_other[i], x=0)

        writer.write(img)

    writer.release()
    return temp_video


def mux_audio(video_no_audio: Path, original_audio: Path, out_video: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_no_audio),
        "-i",
        str(original_audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_visualization(
    audio_path: str,
    out_video_path: str,
    width_px: int = 1920,
    height_px: int = 1080,
    fps_out: int = 60,
    analysis_hz: int = 60,
    sample_rate: int = 44100,
    workdir: str = ".viz_cache",
) -> None:
    if fps_out != analysis_hz:
        raise ValueError("For this prototype, fps_out must equal analysis_hz")

    audio = Path(audio_path).resolve()
    out_video = Path(out_video_path).resolve()
    cache_root = Path(workdir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    key = file_hash(audio)
    cache_dir = cache_root / key
    cache_dir.mkdir(parents=True, exist_ok=True)

    mix_audio, sr, duration = ensure_audio(audio, sample_rate, cache_dir)
    mono_mix = mix_audio.mean(axis=1)

    stems = run_demucs(cache_dir / f"mix_{sample_rate}.wav", cache_dir)

    hop = int(round(sr / analysis_hz))
    n_frames = int(math.floor(duration * analysis_hz))
    times = np.arange(n_frames, dtype=np.float32) / analysis_hz

    loud_db = rms_db_per_frame(mono_mix, hop, n_frames)
    loudness = smooth_1pole(robust_normalize_db(loud_db), alpha=0.25)

    drums, _ = sf.read(str(stems["drums"]), always_2d=True)
    drums_mono = drums.mean(axis=1).astype(np.float32)
    kick_sig = bandpass(drums_mono, sr, 30.0, 120.0)
    other_drums_sig = highpass(drums_mono, sr, 120.0)

    kick = smooth_1pole(robust_normalize_db(rms_db_per_frame(kick_sig, hop, n_frames)), alpha=0.2)
    drums_other = smooth_1pole(robust_normalize_db(rms_db_per_frame(other_drums_sig, hop, n_frames)), alpha=0.2)

    vocals_out = run_basic_pitch(stems["vocals"], cache_dir)
    bass_out = run_basic_pitch(stems["bass"], cache_dir)
    other_out = run_basic_pitch(stems["other"], cache_dir)

    vocals_contour = resample_contour_to_master(vocals_out["contour"], vocals_out["time"], times)
    bass_contour = resample_contour_to_master(bass_out["contour"], bass_out["time"], times)
    other_contour = resample_contour_to_master(other_out["contour"], other_out["time"], times)

    vocals_sel = select_midi_range(vocals_contour, 48, 84)
    bass_sel = select_midi_range(bass_contour, 28, 52)
    other_sel = select_midi_range(other_contour, 36, 96)

    silent_video = render_video(
        out_video,
        width_px,
        height_px,
        fps_out,
        times,
        loudness,
        vocals_sel,
        bass_sel,
        other_sel,
        kick,
        drums_other,
    )

    mux_audio(silent_video, audio, out_video)

    meta = {
        "audio": str(audio),
        "output": str(out_video),
        "width": width_px,
        "height": height_px,
        "fps": fps_out,
        "analysis_hz": analysis_hz,
        "sample_rate": sample_rate,
        "cache": str(cache_dir),
    }
    (cache_dir / "last_run.json").write_text(json.dumps(meta, indent=2))



def _parse_kv_args(argv: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            result[key] = "true"
            i += 1
        else:
            result[key] = argv[i + 1]
            i += 2
    return result


if __name__ == "__main__":
    args = _parse_kv_args(sys.argv[1:])
    if "audio" not in args or "out" not in args:
        raise SystemExit(
            "Usage: python viz.py --audio in.mp3 --out out.mp4 --width 1920 --height 1080 --fps 60 --sr 44100"
        )

    run_visualization(
        audio_path=args["audio"],
        out_video_path=args["out"],
        width_px=int(args.get("width", 1920)),
        height_px=int(args.get("height", 1080)),
        fps_out=int(args.get("fps", 60)),
        analysis_hz=int(args.get("analysis_hz", args.get("fps", 60))),
        sample_rate=int(args.get("sr", 44100)),
        workdir=args.get("workdir", ".viz_cache"),
    )
