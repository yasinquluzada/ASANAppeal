from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRANSCRIPT_PATH_KEYS = (
    "transcript_path",
    "transcript_uri",
    "transcript_file",
)
TRANSCRIPT_TEXT_KEYS = (
    "transcript_text",
    "transcript",
    "transcript_excerpt",
)
LOCAL_PATH_KEYS = (
    "local_object_path",
    "storage_path",
    "object_path",
)
DEFAULT_FRAME_SAMPLE_COUNT = max(1, int(os.getenv("ASAN_VIDEO_FRAME_SAMPLES", "3")))
DEFAULT_TRANSCRIPT_MAX_CHARS = max(
    120, int(os.getenv("ASAN_VIDEO_TRANSCRIPT_MAX_CHARS", "800"))
)
FFMPEG_PATH = os.getenv("ASAN_FFMPEG_PATH", "/opt/homebrew/bin/ffmpeg")
FFPROBE_PATH = os.getenv("ASAN_FFPROBE_PATH", "/opt/homebrew/bin/ffprobe")


@dataclass(frozen=True)
class VideoFrameSample:
    offset_seconds: float
    image_bytes: bytes
    mime_type: str = "image/jpeg"


@dataclass(frozen=True)
class VideoEvidenceContext:
    path: str | None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    has_audio: bool = False
    transcript_text: str | None = None
    transcript_source: str | None = None
    frame_samples: list[VideoFrameSample] = field(default_factory=list)

    def as_metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if self.duration_seconds is not None:
            metadata["video_duration_seconds"] = f"{self.duration_seconds:.2f}"
        if self.width is not None and self.height is not None:
            metadata["video_resolution"] = f"{self.width}x{self.height}"
        if self.frame_count is not None:
            metadata["video_frame_count"] = str(self.frame_count)
        metadata["video_audio_track"] = "present" if self.has_audio else "absent"
        if self.transcript_text:
            metadata["transcript_text"] = self.transcript_text
        if self.transcript_source:
            metadata["transcript_source"] = self.transcript_source
        return metadata

    def summary_lines(self) -> list[str]:
        lines = []
        if self.path:
            lines.append(f"video_path={self.path}")
        if self.duration_seconds is not None:
            lines.append(f"video_duration_seconds={self.duration_seconds:.2f}")
        if self.width is not None and self.height is not None:
            lines.append(f"video_resolution={self.width}x{self.height}")
        if self.frame_count is not None:
            lines.append(f"video_frame_count={self.frame_count}")
        lines.append(f"video_audio_track={'present' if self.has_audio else 'absent'}")
        if self.transcript_text:
            transcript = self.transcript_text.replace("\n", " ").strip()
            lines.append(f"video_transcript_excerpt={transcript}")
        if self.transcript_source:
            lines.append(f"video_transcript_source={self.transcript_source}")
        if self.frame_samples:
            offsets = ", ".join(f"{sample.offset_seconds:.2f}s" for sample in self.frame_samples)
            lines.append(f"video_frame_samples={offsets}")
        return lines


def ffmpeg_available() -> bool:
    return Path(FFMPEG_PATH).exists()


def ffprobe_available() -> bool:
    return Path(FFPROBE_PATH).exists()


def resolve_local_media_path(item: Any) -> Path | None:
    metadata = getattr(item, "metadata", None) or {}
    candidates: list[str] = []
    for key in LOCAL_PATH_KEYS:
        value = metadata.get(key)
        if value:
            candidates.append(str(value))
    uri = getattr(item, "uri", None)
    filename = getattr(item, "filename", None)
    if isinstance(uri, str) and uri and not uri.startswith(("http://", "https://", "data:")):
        candidates.append(uri.replace("file://", "", 1))
    if filename:
        candidates.append(str(filename))

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def video_context_from_item(
    item: Any,
    *,
    frame_sample_count: int = DEFAULT_FRAME_SAMPLE_COUNT,
    transcript_max_chars: int = DEFAULT_TRANSCRIPT_MAX_CHARS,
) -> VideoEvidenceContext | None:
    kind = getattr(item, "kind", None)
    kind_value = getattr(kind, "value", kind)
    if kind_value != "video":
        return None

    path = resolve_local_media_path(item)
    transcript_text, transcript_source = _transcript_from_item(
        item,
        video_path=path,
        max_chars=transcript_max_chars,
    )
    if path is None:
        if transcript_text is None:
            return None
        return VideoEvidenceContext(
            path=None,
            transcript_text=transcript_text,
            transcript_source=transcript_source,
        )
    return _probe_video_path(
        str(path),
        path.stat().st_mtime_ns,
        path.stat().st_size,
        transcript_text or "",
        transcript_source or "",
        max(1, frame_sample_count),
    )


def video_context_from_path(
    path: Path,
    *,
    frame_sample_count: int = DEFAULT_FRAME_SAMPLE_COUNT,
    transcript_text: str | None = None,
    transcript_source: str | None = None,
) -> VideoEvidenceContext:
    resolved = path.expanduser().resolve()
    return _probe_video_path(
        str(resolved),
        resolved.stat().st_mtime_ns,
        resolved.stat().st_size,
        transcript_text or "",
        transcript_source or "",
        max(1, frame_sample_count),
    )


@lru_cache(maxsize=64)
def _probe_video_path(
    path_str: str,
    mtime_ns: int,
    size_bytes: int,
    transcript_text: str,
    transcript_source: str,
    frame_sample_count: int,
) -> VideoEvidenceContext:
    del mtime_ns, size_bytes
    path = Path(path_str)
    payload = _ffprobe_json(path)
    streams = payload.get("streams", []) if isinstance(payload.get("streams"), list) else []
    format_payload = payload.get("format", {}) if isinstance(payload.get("format"), dict) else {}

    video_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
        {},
    )
    audio_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"),
        {},
    )

    duration = _coerce_float(
        format_payload.get("duration")
        or video_stream.get("duration")
        or audio_stream.get("duration")
    )
    width = _coerce_int(video_stream.get("width"))
    height = _coerce_int(video_stream.get("height"))
    frame_count = _coerce_int(video_stream.get("nb_frames"))
    frame_samples = _extract_video_frames(path, duration_seconds=duration, count=frame_sample_count)

    return VideoEvidenceContext(
        path=str(path),
        duration_seconds=duration,
        width=width,
        height=height,
        frame_count=frame_count,
        has_audio=bool(audio_stream),
        transcript_text=transcript_text or None,
        transcript_source=transcript_source or None,
        frame_samples=frame_samples,
    )


def _ffprobe_json(path: Path) -> dict[str, Any]:
    if not ffprobe_available():
        raise RuntimeError("ffprobe is not available for local video processing.")
    completed = subprocess.run(
        [
            FFPROBE_PATH,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout.decode("utf-8") or "{}")


def _extract_video_frames(
    path: Path,
    *,
    duration_seconds: float | None,
    count: int,
) -> list[VideoFrameSample]:
    if not ffmpeg_available():
        return []

    offsets = _sample_offsets(duration_seconds, count)
    frames: list[VideoFrameSample] = []
    for offset in offsets:
        completed = subprocess.run(
            [
                FFMPEG_PATH,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{offset:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout:
            frames.append(VideoFrameSample(offset_seconds=offset, image_bytes=completed.stdout))
    return frames


def _sample_offsets(duration_seconds: float | None, count: int) -> list[float]:
    if duration_seconds is None or duration_seconds <= 0.25 or count <= 1:
        return [0.0]
    samples = {0.0, max(0.0, duration_seconds * 0.5), max(0.0, duration_seconds - 0.2)}
    if count > 3:
        step = duration_seconds / max(1, count - 1)
        for index in range(count):
            samples.add(round(step * index, 3))
    ordered = sorted(samples)
    return ordered[:count]


def _transcript_from_item(
    item: Any,
    *,
    video_path: Path | None,
    max_chars: int,
) -> tuple[str | None, str | None]:
    metadata = getattr(item, "metadata", None) or {}
    for key in TRANSCRIPT_TEXT_KEYS:
        value = metadata.get(key)
        if value:
            return _clean_transcript_text(str(value), max_chars=max_chars), key
    for key in TRANSCRIPT_PATH_KEYS:
        value = metadata.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists() and path.is_file():
                return _read_transcript_file(path, max_chars=max_chars), str(path.resolve())
    if video_path is not None:
        for suffix in (".txt", ".vtt", ".srt"):
            sidecar = video_path.with_suffix(suffix)
            if sidecar.exists() and sidecar.is_file():
                return _read_transcript_file(sidecar, max_chars=max_chars), str(sidecar.resolve())
    return None, None


def _read_transcript_file(path: Path, *, max_chars: int) -> str:
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".vtt", ".srt"}:
        raw_text = _strip_caption_artifacts(raw_text)
    return _clean_transcript_text(raw_text, max_chars=max_chars)


def _strip_caption_artifacts(raw_text: str) -> str:
    cleaned_lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper() == "WEBVTT":
            continue
        if "-->" in stripped:
            continue
        if stripped.isdigit():
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def _clean_transcript_text(raw_text: str, *, max_chars: int) -> str:
    compact = " ".join(raw_text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def probe_uploaded_video_bytes(
    data: bytes,
    *,
    suffix: str = ".mp4",
    frame_sample_count: int = DEFAULT_FRAME_SAMPLE_COUNT,
) -> VideoEvidenceContext:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(data)
        temp_file.flush()
        return video_context_from_path(
            Path(temp_file.name),
            frame_sample_count=frame_sample_count,
        )
