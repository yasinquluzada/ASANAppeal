from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from PIL import ImageChops, ImageStat

from app.config import Settings
from app.image_understanding import analyze_image_item, open_evidence_image_rgb
from app.models.api import InstitutionResponseInput, SubmissionInput
from app.models.domain import StructuredIssue, VerificationDecision, VerificationLabel


LOCATION_STOPWORDS = {
    "the",
    "near",
    "next",
    "beside",
    "behind",
    "front",
    "of",
    "at",
    "on",
    "district",
    "block",
    "zone",
    "area",
}

LOCATION_ABBREVIATIONS = {
    "st": "street",
    "st.": "street",
    "rd": "road",
    "rd.": "road",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "sq": "square",
    "sq.": "square",
    "ctr": "center",
    "ctr.": "center",
    "hwy": "highway",
    "hwy.": "highway",
    "no": "number",
    "no.": "number",
}

COORDINATE_PATTERN = re.compile(r"(-?\d{1,2}\.\d+)\s*[,;/ ]\s*(-?\d{1,3}\.\d+)")

RESOLVED_WORDS = {
    "fixed",
    "repaired",
    "resolved",
    "completed",
    "cleaned",
    "restored",
    "removed",
    "replaced",
    "patched",
    "sealed",
    "paved",
    "drained",
    "pruned",
    "trimmed",
    "installed",
}

UNRESOLVED_WORDS = {
    "pending",
    "scheduled",
    "awaiting",
    "not fixed",
    "inspection",
    "will review",
    "will inspect",
    "temporary",
    "barrier installed",
    "under review",
    "planned",
}

POLICY_BY_CATEGORY = {
    "road_damage": {
        "institution": "ASAN Road Maintenance Agency",
        "visual_threshold": 0.12,
        "visual_tags": {"damaged_surface_candidate"},
        "keywords": {"patched", "paved", "sealed", "repaired", "fixed"},
        "requires_response_evidence": True,
    },
    "street_lighting": {
        "institution": "City Lighting Department",
        "visual_threshold": 0.1,
        "visual_tags": {"poor_lighting_visual", "night_scene"},
        "keywords": {"restored", "fixed", "replaced"},
        "requires_response_evidence": True,
    },
    "water_infrastructure": {
        "institution": "Water and Sewer Authority",
        "visual_threshold": 0.1,
        "visual_tags": {"water_surface", "pooling_water"},
        "keywords": {"drained", "fixed", "repaired", "resolved"},
        "requires_response_evidence": True,
    },
    "waste_management": {
        "institution": "Municipal Sanitation Department",
        "visual_threshold": 0.08,
        "visual_tags": set(),
        "keywords": {"cleaned", "removed", "resolved", "completed"},
        "requires_response_evidence": False,
    },
    "tree_maintenance": {
        "institution": "Parks and Greenery Department",
        "visual_threshold": 0.08,
        "visual_tags": {"branch_clutter", "vegetation"},
        "keywords": {"removed", "trimmed", "pruned", "completed"},
        "requires_response_evidence": True,
    },
    "signage_safety": {
        "institution": "Traffic Management Authority",
        "visual_threshold": 0.08,
        "visual_tags": {"poor_lighting_visual", "night_scene"},
        "keywords": {"replaced", "restored", "fixed", "resolved"},
        "requires_response_evidence": True,
    },
    "public_transport": {
        "institution": "Transit Operations Agency",
        "visual_threshold": 0.08,
        "visual_tags": set(),
        "keywords": {"replaced", "restored", "fixed", "resolved"},
        "requires_response_evidence": False,
    },
    "general_public_service": {
        "institution": "ASAN Operations Triage Desk",
        "visual_threshold": 0.08,
        "visual_tags": set(),
        "keywords": {"completed", "resolved", "fixed"},
        "requires_response_evidence": False,
    },
}


@dataclass(frozen=True)
class VisualVerificationMatch:
    scene_score: float
    issue_change_score: float
    summary: str


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9.,/ -]+", " ", normalized)
    tokens = []
    for token in normalized.split():
        tokens.append(LOCATION_ABBREVIATIONS.get(token, token))
    return " ".join(tokens)


def _location_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value)
        if len(token) > 2 and token not in LOCATION_STOPWORDS
    }


def _extract_coordinates_from_text(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    match = COORDINATE_PATTERN.search(value)
    if not match:
        return None
    latitude = float(match.group(1))
    longitude = float(match.group(2))
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return latitude, longitude
    return None


def _extract_coordinates_from_evidence(submission: SubmissionInput | InstitutionResponseInput) -> tuple[float, float] | None:
    for item in submission.evidence:
        for latitude_key in ("lat", "latitude"):
            for longitude_key in ("lon", "lng", "longitude"):
                if latitude_key in item.metadata and longitude_key in item.metadata:
                    try:
                        latitude = float(item.metadata[latitude_key])
                        longitude = float(item.metadata[longitude_key])
                    except ValueError:
                        continue
                    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                        return latitude, longitude
    return None


def _best_coordinates(submission: SubmissionInput | InstitutionResponseInput) -> tuple[float, float] | None:
    return _extract_coordinates_from_evidence(submission) or _extract_coordinates_from_text(
        submission.location_hint
    )


def _haversine_meters(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, coord_a)
    lat2, lon2 = map(math.radians, coord_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(hav))


def _image_pair_match(
    original_item,
    response_item,
    category: str,
    settings: Settings,
) -> VisualVerificationMatch | None:
    original_observation = analyze_image_item(original_item, settings)
    response_observation = analyze_image_item(response_item, settings)
    original_rgb = open_evidence_image_rgb(original_item, settings)
    response_rgb = open_evidence_image_rgb(response_item, settings)
    if (
        original_observation is None
        or response_observation is None
        or original_rgb is None
        or response_rgb is None
    ):
        return None

    difference = ImageChops.difference(original_rgb, response_rgb)
    diff_mean = sum(ImageStat.Stat(difference).mean) / 3
    pixel_similarity = max(0.0, 1.0 - min(diff_mean / 120.0, 1.0))
    tag_union = set(original_observation.tags) | set(response_observation.tags)
    tag_overlap = (
        len(set(original_observation.tags) & set(response_observation.tags)) / len(tag_union)
        if tag_union
        else 0.0
    )
    aspect_original = original_observation.width / max(original_observation.height, 1)
    aspect_response = response_observation.width / max(response_observation.height, 1)
    aspect_similarity = max(0.0, 1.0 - min(abs(aspect_original - aspect_response) / 0.8, 1.0))
    edge_similarity = max(
        0.0,
        1.0 - min(abs(original_observation.edge_density - response_observation.edge_density) / 0.16, 1.0),
    )
    color_similarity = 1.0 if original_observation.dominant_color == response_observation.dominant_color else 0.45
    scene_score = (
        0.28 * pixel_similarity
        + 0.22 * tag_overlap
        + 0.18 * edge_similarity
        + 0.17 * aspect_similarity
        + 0.15 * color_similarity
    )

    problem_tags = POLICY_BY_CATEGORY.get(category, POLICY_BY_CATEGORY["general_public_service"])["visual_tags"]
    original_problem = len(problem_tags & set(original_observation.tags))
    response_problem = len(problem_tags & set(response_observation.tags))
    issue_change_score = max(
        0.0,
        min(
            1.0,
            (original_problem - response_problem) * 0.35
            + (original_observation.center_dark_fraction - response_observation.center_dark_fraction) * 1.8
            + (original_observation.dark_fraction - response_observation.dark_fraction) * 0.8,
        ),
    )
    summary = (
        f"visual scene score {scene_score:.2f}; "
        f"issue change score {issue_change_score:.2f}; "
        f"pixel similarity {pixel_similarity:.2f}"
    )
    return VisualVerificationMatch(
        scene_score=round(scene_score, 2),
        issue_change_score=round(issue_change_score, 2),
        summary=summary,
    )


def _best_visual_match(
    original_submission: SubmissionInput,
    institution_response: InstitutionResponseInput,
    category: str,
    settings: Settings,
) -> VisualVerificationMatch | None:
    original_images = [item for item in original_submission.evidence if item.kind.value == "image"][:3]
    response_images = [item for item in institution_response.evidence if item.kind.value == "image"][:3]
    best: VisualVerificationMatch | None = None
    for original_item in original_images:
        for response_item in response_images:
            match = _image_pair_match(original_item, response_item, category, settings)
            if match is None:
                continue
            if best is None or (
                match.scene_score + match.issue_change_score
                > best.scene_score + best.issue_change_score
            ):
                best = match
    return best


def _compare_locations(
    original_submission: SubmissionInput,
    institution_response: InstitutionResponseInput,
    visual_match: VisualVerificationMatch | None,
) -> tuple[VerificationLabel, float, list[str], list[str]]:
    original_location = _normalize_text(original_submission.location_hint)
    response_location = _normalize_text(institution_response.location_hint)
    original_tokens = _location_tokens(original_location)
    response_tokens = _location_tokens(response_location)
    token_overlap = (
        len(original_tokens & response_tokens) / len(original_tokens | response_tokens)
        if original_tokens and response_tokens
        else 0.0
    )
    sequence_ratio = (
        SequenceMatcher(None, original_location, response_location, autojunk=False).ratio()
        if original_location and response_location
        else 0.0
    )
    original_coordinates = _best_coordinates(original_submission)
    response_coordinates = _best_coordinates(institution_response)
    coordinate_score: float | None = None
    coordinate_distance: float | None = None
    if original_coordinates and response_coordinates:
        coordinate_distance = _haversine_meters(original_coordinates, response_coordinates)
        if coordinate_distance <= 150:
            coordinate_score = 0.98
        elif coordinate_distance <= 400:
            coordinate_score = 0.78
        elif coordinate_distance <= 1000:
            coordinate_score = 0.46
        else:
            coordinate_score = 0.05

    weighted_score = 0.0
    total_weight = 0.0
    for score, weight in (
        (token_overlap, 0.35),
        (sequence_ratio, 0.3),
        (coordinate_score, 0.2),
        (visual_match.scene_score if visual_match else None, 0.15),
    ):
        if score is None:
            continue
        weighted_score += score * weight
        total_weight += weight
    place_score = weighted_score / total_weight if total_weight else 0.0

    mismatch_flags: list[str] = []
    summary_parts: list[str] = [
        f"text overlap {token_overlap:.2f}",
        f"sequence ratio {sequence_ratio:.2f}",
    ]
    if coordinate_distance is not None:
        summary_parts.append(f"coordinate distance {coordinate_distance:.0f}m")
    if visual_match is not None:
        summary_parts.append(f"visual scene {visual_match.scene_score:.2f}")

    if not response_location and response_coordinates is None:
        same_place = VerificationLabel.uncertain
        mismatch_flags.append("weak_location_match")
    elif coordinate_score is not None and coordinate_score <= 0.05:
        same_place = VerificationLabel.no
        mismatch_flags.append("coordinate_mismatch")
    elif (
        place_score >= 0.66
        or (coordinate_score is not None and coordinate_score >= 0.9)
        or (token_overlap >= 0.5 and sequence_ratio >= 0.8)
        or (original_tokens and response_tokens and original_tokens.issubset(response_tokens))
        or (original_tokens and response_tokens and response_tokens.issubset(original_tokens))
    ):
        same_place = VerificationLabel.yes
    elif place_score <= 0.28:
        same_place = VerificationLabel.no
        mismatch_flags.append("location_mismatch")
    else:
        same_place = VerificationLabel.uncertain
        mismatch_flags.append("weak_location_match")

    if visual_match is not None and visual_match.scene_score < 0.28 and same_place != VerificationLabel.yes:
        mismatch_flags.append("visual_scene_mismatch")
    return same_place, round(place_score, 2), sorted(set(mismatch_flags)), summary_parts


def _resolution_language_scores(response_text: str, category: str) -> tuple[float, float, list[str], list[str]]:
    normalized = _normalize_text(response_text)
    policy = POLICY_BY_CATEGORY.get(category, POLICY_BY_CATEGORY["general_public_service"])
    resolved_hits = sorted(
        {word for word in RESOLVED_WORDS | set(policy["keywords"]) if word in normalized}
    )
    unresolved_hits = sorted({word for word in UNRESOLVED_WORDS if word in normalized})
    resolved_score = min(1.0, 0.22 * len(resolved_hits))
    unresolved_score = min(1.0, 0.35 * len(unresolved_hits))
    return resolved_score, unresolved_score, resolved_hits, unresolved_hits


def _resolve_issue_state(
    *,
    category: str,
    same_place: VerificationLabel,
    response_has_evidence: bool,
    visual_match: VisualVerificationMatch | None,
    resolved_score: float,
    unresolved_score: float,
) -> tuple[VerificationLabel, list[str], list[str]]:
    policy = POLICY_BY_CATEGORY.get(category, POLICY_BY_CATEGORY["general_public_service"])
    mismatch_flags: list[str] = []
    summary_parts: list[str] = [
        f"policy owner {policy['institution']}",
    ]

    if same_place == VerificationLabel.no:
        mismatch_flags.append("resolution_policy_not_satisfied")
        return VerificationLabel.uncertain, mismatch_flags, summary_parts

    visual_support = False
    if visual_match is not None:
        summary_parts.append(visual_match.summary)
        if visual_match.issue_change_score >= float(policy["visual_threshold"]):
            visual_support = True

    if unresolved_score >= 0.35:
        mismatch_flags.append("pending_resolution_language")
        mismatch_flags.append("resolution_policy_not_satisfied")
        return VerificationLabel.no, mismatch_flags, summary_parts

    if policy["requires_response_evidence"] and not response_has_evidence:
        mismatch_flags.append("missing_response_evidence")

    if visual_match is not None and not visual_support and same_place != VerificationLabel.no:
        mismatch_flags.append("issue_visual_change_insufficient")

    language_support = resolved_score >= 0.22
    strong_language = resolved_score >= 0.44

    if same_place == VerificationLabel.yes and (visual_support or strong_language) and (
        response_has_evidence or not policy["requires_response_evidence"]
    ):
        return VerificationLabel.yes, sorted(set(mismatch_flags)), summary_parts

    if same_place == VerificationLabel.yes and language_support and (
        response_has_evidence or not policy["requires_response_evidence"]
    ):
        return VerificationLabel.yes, sorted(set(mismatch_flags)), summary_parts

    if same_place == VerificationLabel.uncertain and visual_support and strong_language:
        return VerificationLabel.uncertain, sorted(set(mismatch_flags)), summary_parts

    mismatch_flags.append("resolution_policy_not_satisfied")
    if resolved_score <= 0:
        mismatch_flags.append("weak_resolution_language")
    return VerificationLabel.uncertain, sorted(set(mismatch_flags)), summary_parts


def verify_resolution_advanced(
    original_submission: SubmissionInput,
    structured_issue: StructuredIssue,
    institution_response: InstitutionResponseInput,
    *,
    settings: Settings | None = None,
) -> VerificationDecision:
    settings = settings or Settings()
    category = structured_issue.category
    visual_match = _best_visual_match(
        original_submission,
        institution_response,
        category,
        settings,
    )
    same_place, place_score, place_flags, place_summary = _compare_locations(
        original_submission,
        institution_response,
        visual_match,
    )
    resolved_score, unresolved_score, resolved_hits, unresolved_hits = _resolution_language_scores(
        institution_response.response_text,
        category,
    )
    issue_resolved, resolution_flags, resolution_summary = _resolve_issue_state(
        category=category,
        same_place=same_place,
        response_has_evidence=bool(institution_response.evidence),
        visual_match=visual_match,
        resolved_score=resolved_score,
        unresolved_score=unresolved_score,
    )

    mismatch_flags = sorted(set(place_flags + resolution_flags))
    confidence_inputs = [0.38 + 0.42 * place_score]
    if visual_match is not None:
        confidence_inputs.append(0.32 + 0.5 * max(visual_match.scene_score, visual_match.issue_change_score))
    if issue_resolved != VerificationLabel.uncertain:
        confidence_inputs.append(0.72 if issue_resolved == VerificationLabel.yes else 0.68)
    if same_place != VerificationLabel.uncertain:
        confidence_inputs.append(0.7 if same_place == VerificationLabel.yes else 0.66)
    confidence = sum(confidence_inputs) / len(confidence_inputs)
    confidence -= 0.05 * max(0, len(mismatch_flags) - 1)
    confidence = max(0.08, min(confidence, 0.98))

    summary_parts = [
        f"Same-place check: {same_place.value}.",
        f"Issue-resolved check: {issue_resolved.value}.",
        "Location signals: " + ", ".join(place_summary) + ".",
        "Resolution policy: " + ", ".join(resolution_summary) + ".",
    ]
    if resolved_hits:
        summary_parts.append(f"Completion language detected: {', '.join(resolved_hits)}.")
    if unresolved_hits:
        summary_parts.append(f"Pending language detected: {', '.join(unresolved_hits)}.")

    return VerificationDecision(
        same_place=same_place,
        issue_resolved=issue_resolved,
        mismatch_flags=mismatch_flags,
        summary=" ".join(summary_parts),
        confidence=round(confidence, 2),
    )
