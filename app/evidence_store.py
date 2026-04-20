from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mimetypes
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.models.domain import EvidenceItem, EvidenceKind, EvidencePrivacyState, StoredEvidence

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}
VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
}
TEXT_MIME_TYPES = {"text/plain"}
DEFAULT_EVIDENCE_ROOT = "./.data/evidence"


@dataclass(frozen=True)
class EvidenceRootChoice:
    requested_root: str
    resolved_root: str
    used_fallback: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class MediaValidationResult:
    mime_type: str
    width: int | None = None
    height: int | None = None
    extension: str | None = None


def _normalize_root(root_path: str) -> Path:
    path = Path(root_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _candidate_roots(requested_root: str) -> list[EvidenceRootChoice]:
    normalized_requested = requested_root.strip() or DEFAULT_EVIDENCE_ROOT
    explicit = EvidenceRootChoice(
        requested_root=normalized_requested,
        resolved_root=str(_normalize_root(normalized_requested)),
        used_fallback=False,
        fallback_reason=None,
    )
    candidates: list[EvidenceRootChoice] = [explicit]
    fallback_specs = [
        ("./.data/evidence", "project_data_dir"),
        (
            str(Path.home() / ".local" / "share" / "asanappeal" / "evidence"),
            "user_data_dir",
        ),
        (str(Path(tempfile.gettempdir()) / "asanappeal" / "evidence"), "temp_dir"),
    ]
    seen = {explicit.resolved_root}
    for candidate_path, reason in fallback_specs:
        resolved = str(_normalize_root(candidate_path))
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(
            EvidenceRootChoice(
                requested_root=normalized_requested,
                resolved_root=resolved,
                used_fallback=True,
                fallback_reason=reason,
            )
        )
    return candidates


def _safe_filename(filename: str | None, *, default_stem: str) -> str:
    raw_name = (filename or "").strip()
    if not raw_name:
        return default_stem
    candidate = Path(raw_name).name
    allowed = {"-", "_", "."}
    sanitized = "".join(ch if ch.isalnum() or ch in allowed else "-" for ch in candidate)
    sanitized = sanitized.strip(".-") or default_stem
    return sanitized


def _extension_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    guessed = mimetypes.guess_extension(mime_type, strict=False)
    return guessed or ".bin"


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    normalized = content_type.split(";", 1)[0].strip().lower()
    aliases = {
        "image/jpg": "image/jpeg",
        "application/octet-stream": "application/octet-stream",
    }
    return aliases.get(normalized, normalized)


def load_stored_evidence_by_id(evidence_id: str, requested_root: str) -> StoredEvidence | None:
    for candidate in _candidate_roots(requested_root):
        manifest_path = Path(candidate.resolved_root) / "manifests" / f"{evidence_id}.json"
        if not manifest_path.exists():
            continue
        try:
            return StoredEvidence.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Stored evidence manifest %s could not be parsed.", manifest_path)
            continue
    return None


class LocalEvidenceStore:
    def __init__(
        self,
        root: str,
        *,
        requested_root: str | None = None,
        max_bytes: int = 25 * 1024 * 1024,
        signed_url_ttl_seconds: int = 900,
        signing_secret: str = "asanappeal-local-evidence-signing-secret",
        thumbnail_max_size: int = 512,
        used_fallback: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        normalized_root = _normalize_root(root)
        self.root = normalized_root
        self.requested_root = requested_root or root
        self.max_bytes = max_bytes
        self.signed_url_ttl_seconds = signed_url_ttl_seconds
        self.signing_secret = signing_secret.encode("utf-8")
        self.thumbnail_max_size = thumbnail_max_size
        self.used_fallback = used_fallback
        self.fallback_reason = fallback_reason

        self.objects_dir = self.root / "objects"
        self.manifests_dir = self.root / "manifests"
        self.thumbnails_dir = self.root / "thumbnails"
        for directory in (self.root, self.objects_dir, self.manifests_dir, self.thumbnails_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def build_hardened(
        cls,
        root: str,
        *,
        max_bytes: int = 25 * 1024 * 1024,
        signed_url_ttl_seconds: int = 900,
        signing_secret: str = "asanappeal-local-evidence-signing-secret",
        thumbnail_max_size: int = 512,
    ) -> LocalEvidenceStore:
        failures: list[str] = []
        for candidate in _candidate_roots(root):
            try:
                store = cls(
                    candidate.resolved_root,
                    requested_root=candidate.requested_root,
                    max_bytes=max_bytes,
                    signed_url_ttl_seconds=signed_url_ttl_seconds,
                    signing_secret=signing_secret,
                    thumbnail_max_size=thumbnail_max_size,
                    used_fallback=candidate.used_fallback,
                    fallback_reason=candidate.fallback_reason,
                )
                if candidate.used_fallback:
                    logger.warning(
                        "Evidence store used fallback root %s because %s was not usable.",
                        candidate.resolved_root,
                        candidate.requested_root,
                    )
                return store
            except OSError as exc:
                failures.append(f"{candidate.resolved_root}: {exc}")
        failure_summary = "; ".join(failures) if failures else "no evidence roots tried"
        raise RuntimeError(f"Evidence store startup failed for all candidate roots: {failure_summary}")

    def diagnostics(self) -> dict[str, object]:
        object_count = len(list(self.manifests_dir.glob("*.json")))
        thumbnail_count = len(list(self.thumbnails_dir.glob("*.png")))
        total_size = sum(path.stat().st_size for path in self.objects_dir.rglob("*") if path.is_file())
        return {
            "evidence_storage_mode": "enabled",
            "evidence_root": str(self.root),
            "evidence_requested_root": self.requested_root,
            "evidence_fallback_used": self.used_fallback,
            "evidence_fallback_reason": self.fallback_reason,
            "evidence_max_bytes": self.max_bytes,
            "evidence_signed_url_ttl_seconds": self.signed_url_ttl_seconds,
            "evidence_thumbnail_max_size": self.thumbnail_max_size,
            "evidence_object_count": object_count,
            "evidence_thumbnail_count": thumbnail_count,
            "evidence_total_bytes": total_size,
        }

    def ingest_bytes(
        self,
        *,
        data: bytes,
        kind: EvidenceKind,
        filename: str | None = None,
        description: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        privacy: EvidencePrivacyState | None = None,
    ) -> StoredEvidence:
        if not data:
            raise ValueError("Evidence upload body was empty.")
        if len(data) > self.max_bytes:
            raise ValueError(
                f"Evidence upload exceeded the maximum allowed size of {self.max_bytes} bytes."
            )

        sha256 = hashlib.sha256(data).hexdigest()
        normalized_content_type = _normalize_content_type(content_type)
        validation = self._validate_media(
            kind=kind,
            data=data,
            content_type=normalized_content_type,
        )
        evidence_id = uuid.uuid4().hex[:16]
        safe_name = _safe_filename(filename, default_stem=f"evidence-{evidence_id}")
        extension = Path(safe_name).suffix or validation.extension or _extension_for_mime(validation.mime_type)
        if not safe_name.endswith(extension):
            safe_name = f"{safe_name}{extension}"

        object_path = self.objects_dir / sha256[:2] / f"{evidence_id}{extension}"
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(data)

        thumbnail_path: Path | None = None
        if kind == EvidenceKind.image:
            thumbnail_path = self._write_thumbnail(evidence_id=evidence_id, source_data=data)

        evidence = StoredEvidence(
            evidence_id=evidence_id,
            kind=kind,
            filename=safe_name,
            description=description,
            mime_type=validation.mime_type,
            size_bytes=len(data),
            sha256=sha256,
            width=validation.width,
            height=validation.height,
            object_path=str(object_path),
            thumbnail_path=str(thumbnail_path) if thumbnail_path is not None else None,
            metadata=metadata or {},
            privacy=privacy or EvidencePrivacyState(),
        )
        self._write_manifest(evidence)
        return evidence

    def get(self, evidence_id: str) -> StoredEvidence | None:
        manifest_path = self.manifests_dir / f"{evidence_id}.json"
        if not manifest_path.exists():
            return None
        return StoredEvidence.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    def build_evidence_item(self, evidence: StoredEvidence, *, metadata_url: str | None = None) -> EvidenceItem:
        return EvidenceItem(
            kind=evidence.kind,
            evidence_id=evidence.evidence_id,
            uri=metadata_url,
            filename=evidence.filename,
            description=evidence.description,
            mime_type=evidence.mime_type,
            size_bytes=evidence.size_bytes,
            sha256=evidence.sha256,
            thumbnail_available=evidence.thumbnail_path is not None,
            width=evidence.width,
            height=evidence.height,
            metadata=evidence.metadata,
            privacy=evidence.privacy,
        )

    def build_signed_params(
        self,
        evidence: StoredEvidence,
        *,
        variant: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, str]:
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=ttl_seconds or self.signed_url_ttl_seconds
        )
        expires = str(int(expires_at.timestamp()))
        message = f"{evidence.evidence_id}:{variant}:{expires}:{evidence.sha256}".encode("utf-8")
        signature = hmac.new(self.signing_secret, msg=message, digestmod="sha256").hexdigest()
        return {
            "expires": expires,
            "signature": signature,
            "expires_at": expires_at.isoformat(),
        }

    def verify_signed_params(
        self,
        evidence: StoredEvidence,
        *,
        variant: str,
        expires: str,
        signature: str,
    ) -> None:
        try:
            expires_int = int(expires)
        except ValueError as exc:  # pragma: no cover - surfaced as HTTP 403
            raise PermissionError("Invalid signed URL expiry.") from exc
        now_timestamp = int(datetime.now(timezone.utc).timestamp())
        if expires_int < now_timestamp:
            raise PermissionError("Signed URL has expired.")

        expected = hmac.new(
            self.signing_secret,
            msg=f"{evidence.evidence_id}:{variant}:{expires}:{evidence.sha256}".encode("utf-8"),
            digestmod="sha256",
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise PermissionError("Invalid signed URL signature.")

    def object_file(self, evidence: StoredEvidence) -> Path:
        return Path(evidence.object_path)

    def thumbnail_file(self, evidence: StoredEvidence) -> Path | None:
        if evidence.thumbnail_path is None:
            return None
        return Path(evidence.thumbnail_path)

    def delete(self, evidence_id: str) -> bool:
        evidence = self.get(evidence_id)
        if evidence is None:
            return False
        object_file = self.object_file(evidence)
        if object_file.exists():
            object_file.unlink()
            parent = object_file.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        thumbnail_file = self.thumbnail_file(evidence)
        if thumbnail_file is not None and thumbnail_file.exists():
            thumbnail_file.unlink()
        manifest_path = self.manifests_dir / f"{evidence_id}.json"
        if manifest_path.exists():
            manifest_path.unlink()
        return True

    def _write_manifest(self, evidence: StoredEvidence) -> None:
        manifest_path = self.manifests_dir / f"{evidence.evidence_id}.json"
        manifest_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")

    def _validate_media(
        self,
        *,
        kind: EvidenceKind,
        data: bytes,
        content_type: str | None,
    ) -> MediaValidationResult:
        if kind == EvidenceKind.image:
            return self._validate_image(data, content_type)
        if kind == EvidenceKind.video:
            return self._validate_video(data, content_type)
        return self._validate_text(data, content_type)

    def _validate_image(self, data: bytes, content_type: str | None) -> MediaValidationResult:
        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
                mime_type = _normalize_content_type(Image.MIME.get(image.format))
                if mime_type not in IMAGE_MIME_TYPES:
                    raise ValueError("Unsupported image MIME type.")
                if content_type not in {None, "application/octet-stream", mime_type}:
                    raise ValueError(
                        f"Uploaded image content type {content_type} does not match detected type {mime_type}."
                    )
                return MediaValidationResult(
                    mime_type=mime_type or "application/octet-stream",
                    width=int(image.width),
                    height=int(image.height),
                    extension=_extension_for_mime(mime_type or "application/octet-stream"),
                )
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded image could not be decoded safely.") from exc

    def _validate_video(self, data: bytes, content_type: str | None) -> MediaValidationResult:
        mime_type: str | None = None
        if len(data) >= 12 and data[4:8] == b"ftyp":
            brand = data[8:12]
            mime_type = "video/quicktime" if brand == b"qt  " else "video/mp4"
        elif data.startswith(b"\x1a\x45\xdf\xa3"):
            mime_type = "video/webm"
        if mime_type not in VIDEO_MIME_TYPES:
            raise ValueError("Unsupported or unrecognized video upload.")
        if content_type not in {None, "application/octet-stream", mime_type}:
            raise ValueError(
                f"Uploaded video content type {content_type} does not match detected type {mime_type}."
            )
        return MediaValidationResult(
            mime_type=mime_type,
            extension=_extension_for_mime(mime_type),
        )

    def _validate_text(self, data: bytes, content_type: str | None) -> MediaValidationResult:
        if b"\x00" in data:
            raise ValueError("Text evidence may not contain NUL bytes.")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text evidence must be valid UTF-8.") from exc
        if content_type not in {None, "application/octet-stream", "text/plain"}:
            raise ValueError(
                f"Uploaded text content type {content_type} is not supported for text evidence."
            )
        return MediaValidationResult(
            mime_type="text/plain",
            extension=".txt",
        )

    def _write_thumbnail(self, *, evidence_id: str, source_data: bytes) -> Path:
        thumbnail_path = self.thumbnails_dir / f"{evidence_id}.png"
        with Image.open(BytesIO(source_data)) as image:
            image.load()
            thumbnail = image.copy()
            thumbnail.thumbnail((self.thumbnail_max_size, self.thumbnail_max_size))
            if thumbnail.mode not in {"RGB", "RGBA"}:
                thumbnail = thumbnail.convert("RGB")
            thumbnail.save(thumbnail_path, format="PNG")
        return thumbnail_path
