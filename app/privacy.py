from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError

from app.config import Settings
from app.models.api import InstitutionResponseInput, ProcessCaseRequest, SubmissionInput
from app.models.domain import (
    CasePrivacyState,
    CaseRecord,
    EvidenceItem,
    EvidenceKind,
    EvidencePrivacyState,
    ExplanationNote,
    PrivacyImageRegion,
    PrivacyTextFinding,
    StoredEvidence,
    VerificationDecision,
)

EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d[\d()\-\s]{7,}\d))")
ID_PATTERN = re.compile(r"\b(?:[A-Z]{0,3}\d{6,}|\d{7,})\b")
UNIT_PATTERN = re.compile(
    r"\b(?:apartment|apt|flat|unit|suite|floor|door|entrance|building|block|bldg)\s*[-:#]?\s*[A-Za-z0-9/-]+\b",
    re.IGNORECASE,
)
COORDINATE_PAIR_PATTERN = re.compile(
    r"(-?\d{1,2}\.\d{4,})\s*,\s*(-?\d{1,3}\.\d{4,})"
)

MASK_FILL = (0, 0, 0)


@dataclass(frozen=True)
class PreparedEvidenceUpload:
    data: bytes
    filename: str | None
    description: str | None
    content_type: str | None
    privacy: EvidencePrivacyState


@dataclass(frozen=True)
class PrivacyExportArtifact:
    case_id: str
    exported_at: str
    export_path: str
    archive_size_bytes: int
    evidence_count: int
    audit_event_count: int


@dataclass(frozen=True)
class PrivacyRetentionRun:
    executed_at: str
    cases_scanned: int
    cases_privacy_deleted: int
    evidence_deleted: int
    affected_case_ids: list[str]
    records: list["PrivacyRetentionRecord"]


@dataclass(frozen=True)
class PrivacyRetentionRecord:
    case_id: str
    before_case: CaseRecord
    after_case: CaseRecord
    deleted_evidence_ids: list[str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _preview(value: str) -> str:
    if len(value) <= 8:
        return value
    return f"{value[:3]}...{value[-2:]}"


def _replace_matches(
    text: str,
    *,
    pattern: re.Pattern[str],
    pii_type: str,
    replacement: str,
    field_path: str,
) -> tuple[str, list[PrivacyTextFinding]]:
    findings: list[PrivacyTextFinding] = []

    def repl(match: re.Match[str]) -> str:
        findings.append(
            PrivacyTextFinding(
                pii_type=pii_type,
                field_path=field_path,
                match_preview=_preview(match.group(0)),
                replacement=replacement,
            )
        )
        return replacement

    return pattern.sub(repl, text), findings


def _minimize_location(text: str, field_path: str) -> tuple[str, list[PrivacyTextFinding], bool]:
    updated = text
    findings: list[PrivacyTextFinding] = []
    minimized = False

    def coords_repl(match: re.Match[str]) -> str:
        nonlocal minimized
        minimized = True
        lat = round(float(match.group(1)), 3)
        lon = round(float(match.group(2)), 3)
        findings.append(
            PrivacyTextFinding(
                pii_type="coordinate_precision",
                field_path=field_path,
                match_preview=_preview(match.group(0)),
                replacement=f"{lat:.3f}, {lon:.3f}",
            )
        )
        return f"{lat:.3f}, {lon:.3f}"

    updated = COORDINATE_PAIR_PATTERN.sub(coords_repl, updated)
    without_units, unit_findings = _replace_matches(
        updated,
        pattern=UNIT_PATTERN,
        pii_type="address_unit",
        replacement="",
        field_path=field_path,
    )
    if unit_findings:
        minimized = True
        findings.extend(unit_findings)
    updated = without_units

    normalized = re.sub(r"\b\d+[A-Za-z/-]*\b", "", updated)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = normalized.strip(" ,.-")
    if normalized and normalized != text:
        minimized = True
        findings.append(
            PrivacyTextFinding(
                pii_type="address_precision",
                field_path=field_path,
                match_preview=_preview(text),
                replacement=f"{normalized} area",
            )
        )
        updated = f"{normalized} area"
    else:
        updated = normalized or text
    return updated, findings, minimized


def _sanitize_free_text(
    text: str | None,
    *,
    field_path: str,
    minimize_address: bool = False,
) -> tuple[str | None, list[PrivacyTextFinding], bool]:
    if not text:
        return text, [], False

    updated = text
    findings: list[PrivacyTextFinding] = []
    redaction_applied = False

    for pattern, pii_type, replacement in (
        (EMAIL_PATTERN, "email", "[redacted-email]"),
        (PHONE_PATTERN, "phone", "[redacted-phone]"),
        (ID_PATTERN, "identifier", "[redacted-id]"),
    ):
        updated, pattern_findings = _replace_matches(
            updated,
            pattern=pattern,
            pii_type=pii_type,
            replacement=replacement,
            field_path=field_path,
        )
        if pattern_findings:
            findings.extend(pattern_findings)
            redaction_applied = True

    if minimize_address:
        updated, address_findings, minimized = _minimize_location(updated, field_path)
        if minimized:
            findings.extend(address_findings)
            redaction_applied = True

    return updated, findings, redaction_applied


def _iter_string_fields(payload: Any, field_path: str = "") -> list[tuple[str, str]]:
    if isinstance(payload, str):
        return [(field_path, payload)]
    if isinstance(payload, list):
        items: list[tuple[str, str]] = []
        for index, value in enumerate(payload):
            items.extend(_iter_string_fields(value, f"{field_path}[{index}]"))
        return items
    if isinstance(payload, dict):
        items: list[tuple[str, str]] = []
        for key, value in payload.items():
            next_path = f"{field_path}.{key}" if field_path else str(key)
            items.extend(_iter_string_fields(value, next_path))
        return items
    return []


def _component_regions(mask: list[bool], width: int, height: int, *, min_pixels: int) -> list[tuple[int, int, int, int, int]]:
    visited = bytearray(width * height)
    regions: list[tuple[int, int, int, int, int]] = []
    for index, active in enumerate(mask):
        if not active or visited[index]:
            continue
        stack = [index]
        visited[index] = 1
        count = 0
        min_x = max_x = index % width
        min_y = max_y = index // width
        while stack:
            current = stack.pop()
            x = current % width
            y = current // width
            count += 1
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    next_index = ny * width + nx
                    if mask[next_index] and not visited[next_index]:
                        visited[next_index] = 1
                        stack.append(next_index)
        if count >= min_pixels:
            regions.append((min_x, min_y, max_x, max_y, count))
    return regions


def _scale_region(
    region: tuple[int, int, int, int, int],
    *,
    scan_width: int,
    scan_height: int,
    image_width: int,
    image_height: int,
    pii_type: str,
    detector: str,
    confidence: float,
) -> PrivacyImageRegion:
    min_x, min_y, max_x, max_y, _count = region
    scale_x = image_width / scan_width
    scale_y = image_height / scan_height
    left = max(0, int(min_x * scale_x))
    top = max(0, int(min_y * scale_y))
    right = min(image_width, int((max_x + 1) * scale_x))
    bottom = min(image_height, int((max_y + 1) * scale_y))
    width = max(1, right - left)
    height = max(1, bottom - top)
    padding_x = max(2, width // 8)
    padding_y = max(2, height // 8)
    left = max(0, left - padding_x)
    top = max(0, top - padding_y)
    right = min(image_width, right + padding_x)
    bottom = min(image_height, bottom + padding_y)
    return PrivacyImageRegion(
        pii_type=pii_type,
        left=left,
        top=top,
        width=max(1, right - left),
        height=max(1, bottom - top),
        detector=detector,
        confidence=confidence,
    )


def _detect_face_regions(image: Image.Image) -> list[PrivacyImageRegion]:
    scan = image.copy()
    scan.thumbnail((180, 180))
    scan = scan.convert("RGB")
    width, height = scan.size
    pixels = scan.load()
    mask: list[bool] = []
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            mask.append(
                red > 95
                and green > 40
                and blue > 20
                and max(red, green, blue) - min(red, green, blue) > 15
                and abs(red - green) > 15
                and red > green
                and red > blue
            )
    regions: list[PrivacyImageRegion] = []
    for region in _component_regions(mask, width, height, min_pixels=max(18, (width * height) // 400)):
        min_x, min_y, max_x, max_y, count = region
        region_width = max_x - min_x + 1
        region_height = max_y - min_y + 1
        aspect = region_width / max(1, region_height)
        area_ratio = count / max(1, width * height)
        if 0.55 <= aspect <= 1.65 and 0.003 <= area_ratio <= 0.25:
            regions.append(
                _scale_region(
                    region,
                    scan_width=width,
                    scan_height=height,
                    image_width=image.width,
                    image_height=image.height,
                    pii_type="face_candidate",
                    detector="skin-tone-component",
                    confidence=0.78,
                )
            )
    return regions


def _detect_plate_regions(image: Image.Image) -> list[PrivacyImageRegion]:
    scan = image.copy()
    scan.thumbnail((220, 220))
    scan = scan.convert("RGB")
    width, height = scan.size
    pixels = scan.load()
    mask: list[bool] = []
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            brightness = (red + green + blue) / 3
            chroma = max(red, green, blue) - min(red, green, blue)
            mask.append(brightness >= 165 and chroma <= 35)
    regions: list[PrivacyImageRegion] = []
    for region in _component_regions(mask, width, height, min_pixels=max(20, (width * height) // 500)):
        min_x, min_y, max_x, max_y, count = region
        region_width = max_x - min_x + 1
        region_height = max_y - min_y + 1
        aspect = region_width / max(1, region_height)
        area_ratio = count / max(1, width * height)
        if 1.8 <= aspect <= 6.5 and 0.002 <= area_ratio <= 0.12:
            regions.append(
                _scale_region(
                    region,
                    scan_width=width,
                    scan_height=height,
                    image_width=image.width,
                    image_height=image.height,
                    pii_type="license_plate_candidate",
                    detector="bright-rectangle-component",
                    confidence=0.81,
                )
            )
    return regions


class PrivacyService:
    def __init__(self, settings: Settings, evidence_store) -> None:
        self.settings = settings
        self.evidence_store = evidence_store

    def diagnostics(self) -> dict[str, object]:
        return {
            "privacy_enabled": True,
            "privacy_redaction_backend": "regex+pillow-mask",
            "privacy_case_retention_days": self.settings.privacy_case_retention_days,
            "privacy_evidence_retention_days": self.settings.privacy_evidence_retention_days,
            "privacy_export_dir": self.settings.privacy_export_dir,
        }

    def _case_delete_after(self) -> datetime:
        return _utc_now() + timedelta(days=self.settings.privacy_case_retention_days)

    def _evidence_delete_after(self) -> datetime:
        return _utc_now() + timedelta(days=self.settings.privacy_evidence_retention_days)

    def prepare_evidence_upload(
        self,
        *,
        data: bytes,
        kind: EvidenceKind,
        filename: str | None,
        description: str | None,
        content_type: str | None,
    ) -> PreparedEvidenceUpload:
        sanitized_description, description_findings, description_redacted = _sanitize_free_text(
            description,
            field_path="evidence.description",
            minimize_address=True,
        )
        privacy = EvidencePrivacyState(
            retention_delete_after=self._evidence_delete_after(),
        )
        privacy.text_findings.extend(description_findings)
        privacy.address_minimized = any(
            finding.pii_type in {"address_unit", "address_precision", "coordinate_precision"}
            for finding in description_findings
        )

        prepared_data = data
        prepared_content_type = content_type
        prepared_filename = filename
        if kind == EvidenceKind.text:
            text_payload = data.decode("utf-8")
            redacted_text, text_findings, text_redacted = _sanitize_free_text(
                text_payload,
                field_path="evidence.text",
                minimize_address=True,
            )
            prepared_data = (redacted_text or "").encode("utf-8")
            privacy.text_findings.extend(text_findings)
            privacy.redaction_applied = description_redacted or text_redacted
            privacy.pii_detected = bool(privacy.text_findings)
            privacy.address_minimized = privacy.address_minimized or any(
                finding.pii_type in {"address_unit", "address_precision", "coordinate_precision"}
                for finding in text_findings
            )
            privacy.source_sha256 = hashlib.sha256(data).hexdigest()
            return PreparedEvidenceUpload(
                data=prepared_data,
                filename=prepared_filename,
                description=sanitized_description,
                content_type=prepared_content_type,
                privacy=privacy,
            )

        if kind == EvidenceKind.image:
            try:
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    mask_regions = [
                        *_detect_face_regions(image),
                        *_detect_plate_regions(image),
                    ]
                    privacy.image_regions.extend(mask_regions)
                    if mask_regions:
                        masked = image.copy()
                        draw = ImageDraw.Draw(masked)
                        for region in mask_regions:
                            draw.rectangle(
                                (
                                    region.left,
                                    region.top,
                                    region.left + region.width,
                                    region.top + region.height,
                                ),
                                fill=MASK_FILL,
                            )
                        buffer = BytesIO()
                        format_name = image.format or "PNG"
                        save_format = "PNG" if format_name.upper() == "GIF" else format_name
                        masked.save(buffer, format=save_format)
                        prepared_data = buffer.getvalue()
                        prepared_content_type = Image.MIME.get(save_format, content_type or "image/png")
                        if prepared_filename:
                            prepared_filename = f"{Path(prepared_filename).stem}-redacted"
                        privacy.redaction_applied = True
                        privacy.pii_detected = True
                        privacy.source_sha256 = hashlib.sha256(data).hexdigest()
                    else:
                        privacy.redaction_applied = description_redacted
                        privacy.pii_detected = bool(description_findings)
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError("Uploaded image could not be decoded safely.") from exc
        else:
            privacy.redaction_applied = description_redacted
            privacy.pii_detected = bool(description_findings)

        return PreparedEvidenceUpload(
            data=prepared_data,
            filename=prepared_filename,
            description=sanitized_description,
            content_type=prepared_content_type,
            privacy=privacy,
        )

    def _sanitize_evidence_item(self, item: EvidenceItem, index: int) -> tuple[EvidenceItem, list[PrivacyTextFinding], bool]:
        description, findings, redacted = _sanitize_free_text(
            item.description,
            field_path=f"submission.evidence[{index}].description",
            minimize_address=True,
        )
        updated_privacy = item.privacy.model_copy(
            update={
                "pii_detected": item.privacy.pii_detected or bool(findings),
                "redaction_applied": item.privacy.redaction_applied or redacted,
                "text_findings": [*item.privacy.text_findings, *findings],
                "retention_delete_after": item.privacy.retention_delete_after
                or self._evidence_delete_after(),
            }
        )
        return (
            item.model_copy(update={"description": description, "privacy": updated_privacy}),
            findings,
            redacted,
        )

    def sanitize_process_request(
        self,
        payload: ProcessCaseRequest,
    ) -> tuple[dict[str, object], CasePrivacyState]:
        text_findings: list[PrivacyTextFinding] = []
        redacted_fields: list[str] = []
        masked_evidence_ids: list[str] = []

        citizen_text, citizen_findings, citizen_redacted = _sanitize_free_text(
            payload.submission.citizen_text,
            field_path="submission.citizen_text",
            minimize_address=True,
        )
        if citizen_redacted:
            redacted_fields.append("submission.citizen_text")
        text_findings.extend(citizen_findings)

        location_hint, location_findings, location_redacted = _sanitize_free_text(
            payload.submission.location_hint,
            field_path="submission.location_hint",
            minimize_address=True,
        )
        if location_redacted:
            redacted_fields.append("submission.location_hint")
        text_findings.extend(location_findings)

        sanitized_evidence: list[EvidenceItem] = []
        for index, item in enumerate(payload.submission.evidence):
            sanitized_item, findings, redacted = self._sanitize_evidence_item(item, index)
            if redacted:
                redacted_fields.append(f"submission.evidence[{index}]")
            if sanitized_item.privacy.image_regions:
                masked_evidence_ids.append(sanitized_item.evidence_id or f"evidence-{index}")
            text_findings.extend(findings)
            sanitized_evidence.append(sanitized_item)

        sanitized_submission = SubmissionInput(
            citizen_text=citizen_text or "",
            language=payload.submission.language,
            location_hint=location_hint,
            time_hint=payload.submission.time_hint,
            evidence=sanitized_evidence,
        )

        sanitized_response = None
        if payload.institution_response is not None:
            response_text, response_findings, response_redacted = _sanitize_free_text(
                payload.institution_response.response_text,
                field_path="institution_response.response_text",
                minimize_address=True,
            )
            if response_redacted:
                redacted_fields.append("institution_response.response_text")
            text_findings.extend(response_findings)
            response_location, response_location_findings, response_location_redacted = _sanitize_free_text(
                payload.institution_response.location_hint,
                field_path="institution_response.location_hint",
                minimize_address=True,
            )
            if response_location_redacted:
                redacted_fields.append("institution_response.location_hint")
            text_findings.extend(response_location_findings)
            sanitized_response_evidence: list[EvidenceItem] = []
            for index, item in enumerate(payload.institution_response.evidence):
                sanitized_item, findings, redacted = self._sanitize_evidence_item(item, index)
                if redacted:
                    redacted_fields.append(f"institution_response.evidence[{index}]")
                if sanitized_item.privacy.image_regions:
                    masked_evidence_ids.append(
                        sanitized_item.evidence_id or f"institution-response-evidence-{index}"
                    )
                text_findings.extend(findings)
                sanitized_response_evidence.append(sanitized_item)
            sanitized_response = InstitutionResponseInput(
                response_text=response_text or "",
                location_hint=response_location,
                evidence=sanitized_response_evidence,
            )

        sanitized_request = ProcessCaseRequest(
            submission=sanitized_submission,
            institution_response=sanitized_response,
        ).model_dump(mode="json", exclude_unset=True, exclude_none=True)
        privacy_state = CasePrivacyState(
            pii_detected=bool(text_findings or masked_evidence_ids),
            redaction_applied=bool(redacted_fields or masked_evidence_ids),
            address_minimized=any(
                finding.pii_type in {"address_unit", "address_precision", "coordinate_precision"}
                for finding in text_findings
            ),
            redacted_field_paths=sorted(set(redacted_fields)),
            text_findings=text_findings,
            masked_evidence_ids=sorted(set(masked_evidence_ids)),
            evidence_delete_after=self._evidence_delete_after(),
            case_delete_after=self._case_delete_after(),
        )
        return sanitized_request, privacy_state

    def redact_case_record(
        self,
        case: CaseRecord,
        *,
        base_privacy: CasePrivacyState | None = None,
    ) -> CaseRecord:
        privacy = (base_privacy or case.privacy).model_copy(deep=True)
        findings = list(privacy.text_findings)
        redacted_fields = list(privacy.redacted_field_paths)

        def sanitize(value: str | None, field_path: str, *, minimize_address: bool = True) -> str | None:
            sanitized, new_findings, changed = _sanitize_free_text(
                value,
                field_path=field_path,
                minimize_address=minimize_address,
            )
            if changed:
                redacted_fields.append(field_path)
            findings.extend(new_findings)
            return sanitized

        structured_issue = case.structured_issue.model_copy(
            update={
                "summary": sanitize(case.structured_issue.summary, "case.structured_issue.summary"),
                "extracted_signals": [
                    sanitize(signal, f"case.structured_issue.extracted_signals[{index}]", minimize_address=False) or ""
                    for index, signal in enumerate(case.structured_issue.extracted_signals)
                ],
                "missing_information": [
                    sanitize(
                        signal,
                        f"case.structured_issue.missing_information[{index}]",
                        minimize_address=True,
                    )
                    or ""
                    for index, signal in enumerate(case.structured_issue.missing_information)
                ],
            }
        )
        routing = case.routing.model_copy(
            update={
                "rationale": sanitize(case.routing.rationale, "case.routing.rationale") or case.routing.rationale,
            }
        )
        priority = case.priority.model_copy(
            update={
                "reasons": [
                    sanitize(reason, f"case.priority.reasons[{index}]", minimize_address=True) or ""
                    for index, reason in enumerate(case.priority.reasons)
                ]
            }
        )
        draft = case.draft.model_copy(
            update={
                "title": sanitize(case.draft.title, "case.draft.title") or case.draft.title,
                "body": sanitize(case.draft.body, "case.draft.body") or case.draft.body,
                "citizen_review_checklist": [
                    sanitize(item, f"case.draft.citizen_review_checklist[{index}]") or ""
                    for index, item in enumerate(case.draft.citizen_review_checklist)
                ],
            }
        )
        verification = None
        if case.verification is not None:
            verification = VerificationDecision(
                same_place=case.verification.same_place,
                issue_resolved=case.verification.issue_resolved,
                mismatch_flags=[
                    sanitize(
                        flag,
                        f"case.verification.mismatch_flags[{index}]",
                        minimize_address=False,
                    )
                    or ""
                    for index, flag in enumerate(case.verification.mismatch_flags)
                ],
                summary=sanitize(case.verification.summary, "case.verification.summary")
                or case.verification.summary,
                confidence=case.verification.confidence,
            )
        explanation = ExplanationNote(
            summary=sanitize(case.explanation.summary, "case.explanation.summary")
            or case.explanation.summary,
            next_action=sanitize(case.explanation.next_action, "case.explanation.next_action")
            or case.explanation.next_action,
            detailed_rationale=[
                sanitize(item, f"case.explanation.detailed_rationale[{index}]") or ""
                for index, item in enumerate(case.explanation.detailed_rationale)
            ],
            risk_flags=[
                sanitize(item, f"case.explanation.risk_flags[{index}]", minimize_address=False) or ""
                for index, item in enumerate(case.explanation.risk_flags)
            ],
        )
        human_review = case.human_review.model_copy(
            update={
                "reasons": [
                    sanitize(item, f"case.human_review.reasons[{index}]") or ""
                    for index, item in enumerate(case.human_review.reasons)
                ]
            }
        )
        sanitized_last_response = case.last_institution_response
        if isinstance(case.last_institution_response, dict):
            sanitized_last_response = self._sanitize_json_payload(
                case.last_institution_response,
                root="case.last_institution_response",
            )
        updated_privacy = privacy.model_copy(
            update={
                "pii_detected": privacy.pii_detected or bool(findings or privacy.masked_evidence_ids),
                "redaction_applied": privacy.redaction_applied or bool(redacted_fields),
                "address_minimized": privacy.address_minimized
                or any(
                    finding.pii_type
                    in {"address_unit", "address_precision", "coordinate_precision"}
                    for finding in findings
                ),
                "redacted_field_paths": sorted(set(redacted_fields)),
                "text_findings": findings,
                "case_delete_after": privacy.case_delete_after or self._case_delete_after(),
                "evidence_delete_after": privacy.evidence_delete_after or self._evidence_delete_after(),
            }
        )
        return case.model_copy(
            update={
                "submission_excerpt": sanitize(case.submission_excerpt, "case.submission_excerpt")
                or case.submission_excerpt,
                "structured_issue": structured_issue,
                "routing": routing,
                "priority": priority,
                "draft": draft,
                "verification": verification,
                "explanation": explanation,
                "human_review": human_review,
                "last_institution_response": sanitized_last_response,
                "privacy": updated_privacy,
            }
        )

    def _sanitize_json_payload(self, payload: Any, *, root: str) -> Any:
        if isinstance(payload, dict):
            return {
                key: self._sanitize_json_payload(value, root=f"{root}.{key}")
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [
                self._sanitize_json_payload(value, root=f"{root}[{index}]")
                for index, value in enumerate(payload)
            ]
        if isinstance(payload, str):
            sanitized, _findings, _changed = _sanitize_free_text(
                payload,
                field_path=root,
                minimize_address=True,
            )
            return sanitized
        return payload

    def sanitize_audit_payload(self, payload: dict[str, object]) -> dict[str, object]:
        return self._sanitize_json_payload(payload, root="audit") or {}

    def export_case_bundle(
        self,
        *,
        case: CaseRecord,
        request_payload: dict[str, object] | None,
        audit_events: list[Any],
    ) -> PrivacyExportArtifact:
        export_root = Path(self.settings.privacy_export_dir).expanduser()
        if not export_root.is_absolute():
            export_root = Path(__file__).resolve().parent.parent / export_root
        export_root.mkdir(parents=True, exist_ok=True)
        exported_at = _utc_now()
        archive_path = export_root / f"{case.case_id}-privacy-export-{exported_at.strftime('%Y%m%d%H%M%S')}.zip"
        sanitized_case = self.redact_case_record(case)
        sanitized_request = self._sanitize_json_payload(request_payload or {}, root="original_request")
        sanitized_audit = [
            {
                **event.model_dump(mode="json"),
                "payload": self.sanitize_audit_payload(event.payload),
            }
            for event in audit_events
        ]
        evidence_ids = self.extract_evidence_ids(request_payload or {})
        if isinstance(case.last_institution_response, dict):
            evidence_ids.update(self.extract_evidence_ids(case.last_institution_response))

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "case.json",
                sanitized_case.model_dump_json(indent=2),
            )
            archive.writestr(
                "original_request.json",
                json.dumps(sanitized_request, ensure_ascii=False, indent=2),
            )
            archive.writestr(
                "audit_log.json",
                json.dumps(sanitized_audit, ensure_ascii=False, indent=2),
            )
            for evidence_id in sorted(evidence_ids):
                evidence = self.evidence_store.get(evidence_id)
                if evidence is None:
                    continue
                object_file = self.evidence_store.object_file(evidence)
                if object_file.exists():
                    archive.write(object_file, arcname=f"evidence/{evidence.filename}")
                if evidence.thumbnail_path:
                    thumbnail_file = self.evidence_store.thumbnail_file(evidence)
                    if thumbnail_file is not None and thumbnail_file.exists():
                        archive.write(
                            thumbnail_file,
                            arcname=f"evidence/thumbnails/{evidence_id}.png",
                        )
                archive.writestr(
                    f"evidence/manifests/{evidence_id}.json",
                    evidence.model_dump_json(indent=2),
                )
        return PrivacyExportArtifact(
            case_id=case.case_id,
            exported_at=exported_at.isoformat(),
            export_path=str(archive_path),
            archive_size_bytes=archive_path.stat().st_size,
            evidence_count=len(evidence_ids),
            audit_event_count=len(audit_events),
        )

    def extract_evidence_ids(self, payload: Any) -> set[str]:
        evidence_ids: set[str] = set()
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key == "evidence" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and item.get("evidence_id"):
                            evidence_ids.add(str(item["evidence_id"]))
                else:
                    evidence_ids.update(self.extract_evidence_ids(value))
        elif isinstance(payload, list):
            for item in payload:
                evidence_ids.update(self.extract_evidence_ids(item))
        return evidence_ids

    def privacy_delete_case(
        self,
        *,
        case: CaseRecord,
        request_payload: dict[str, object] | None,
        note: str | None = None,
    ) -> tuple[CaseRecord, dict[str, object], list[str]]:
        evidence_ids = self.extract_evidence_ids(request_payload or {})
        if isinstance(case.last_institution_response, dict):
            evidence_ids.update(self.extract_evidence_ids(case.last_institution_response))
        deleted_ids: list[str] = []
        for evidence_id in sorted(evidence_ids):
            if self.evidence_store.delete(evidence_id):
                deleted_ids.append(evidence_id)

        deleted_at = _utc_now()
        redacted_case = self.redact_case_record(case)
        updated_privacy = redacted_case.privacy.model_copy(
            update={
                "deleted_at": deleted_at,
                "deleted_evidence_ids": sorted(set([*redacted_case.privacy.deleted_evidence_ids, *deleted_ids])),
            }
        )
        tombstone_case = redacted_case.model_copy(
            update={
                "submission_excerpt": "[privacy deleted]",
                "last_institution_response": None,
                "privacy": updated_privacy,
                "operations": redacted_case.operations.model_copy(
                    update={
                        "final_disposition": "privacy_deleted",
                        "final_disposition_reason": note or "Privacy deletion workflow executed.",
                        "disposition_updated_at": deleted_at,
                    }
                ),
            }
        )
        tombstone_request = {
            "case_id": case.case_id,
            "privacy_deleted_at": deleted_at.isoformat(),
            "deleted_evidence_ids": deleted_ids,
            "note": note,
        }
        return tombstone_case, tombstone_request, deleted_ids

    def enforce_retention(self, *, repository) -> PrivacyRetentionRun:
        now = _utc_now()
        scanned = 0
        deleted_cases = 0
        evidence_deleted = 0
        affected_case_ids: list[str] = []
        records: list[PrivacyRetentionRecord] = []
        for item in repository.list_cases(limit=10000):
            case = repository.get_case(str(item["case_id"]))
            if case is None:
                continue
            scanned += 1
            privacy = case.privacy or CasePrivacyState()
            case_due = privacy.case_delete_after is not None and privacy.case_delete_after <= now
            evidence_due = (
                privacy.evidence_delete_after is not None
                and privacy.evidence_delete_after <= now
                and privacy.deleted_at is None
            )
            if not case_due and not evidence_due:
                continue
            request_payload = repository.get_case_request_payload(case.case_id)
            tombstone_case, tombstone_request, deleted_ids = self.privacy_delete_case(
                case=case,
                request_payload=request_payload,
                note="Retention policy executed.",
            )
            repository.save_case(tombstone_case, tombstone_request)
            deleted_cases += 1
            evidence_deleted += len(deleted_ids)
            affected_case_ids.append(case.case_id)
            records.append(
                PrivacyRetentionRecord(
                    case_id=case.case_id,
                    before_case=case,
                    after_case=tombstone_case,
                    deleted_evidence_ids=deleted_ids,
                )
            )
        return PrivacyRetentionRun(
            executed_at=now.isoformat(),
            cases_scanned=scanned,
            cases_privacy_deleted=deleted_cases,
            evidence_deleted=evidence_deleted,
            affected_case_ids=affected_case_ids,
            records=records,
        )
