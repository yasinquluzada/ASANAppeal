from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

from app.config import Settings
from app.evidence_store import load_stored_evidence_by_id
from app.models.api import SubmissionInput
from app.models.domain import EvidenceItem, EvidenceKind


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageObservation:
    evidence_id: str | None
    width: int
    height: int
    brightness: float
    contrast: float
    saturation: float
    edge_density: float
    dark_fraction: float
    center_dark_fraction: float
    dominant_color: str
    tags: tuple[str, ...]
    summary: str
    suggested_category: str | None
    confidence: float


@dataclass(frozen=True)
class VisualContext:
    observation_count: int
    tags: tuple[str, ...]
    summary: str
    suggested_category: str | None
    confidence: float


def load_image_bytes(item: EvidenceItem, settings: Settings) -> bytes | None:
    if item.kind != EvidenceKind.image:
        return None

    if item.evidence_id:
        stored = load_stored_evidence_by_id(item.evidence_id, settings.evidence_root)
        if stored and Path(stored.object_path).exists():
            return Path(stored.object_path).read_bytes()

    if not item.uri:
        return None

    uri = item.uri.strip()
    if uri.startswith("data:image/") and "," in uri:
        try:
            return base64.b64decode(uri.split(",", 1)[1], validate=True)
        except Exception:
            return None

    path = Path(uri).expanduser()
    if path.exists() and path.is_file():
        return path.read_bytes()
    return None


def open_evidence_image_rgb(
    item: EvidenceItem,
    settings: Settings,
    *,
    size: tuple[int, int] = (128, 128),
) -> Image.Image | None:
    data = load_image_bytes(item, settings)
    if not data:
        return None
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
            if size:
                rgb = rgb.resize(size)
            return rgb
    except (UnidentifiedImageError, OSError):
        logger.warning("Image evidence could not be opened for RGB comparison.")
        return None


def _dominant_color_label(red: float, green: float, blue: float) -> str:
    if blue > red + 14 and blue > green + 10:
        return "blue"
    if green > red + 12 and green > blue + 8:
        return "green"
    if red > green + 12 and red > blue + 12:
        return "red"
    if abs(red - green) < 14 and abs(green - blue) < 14:
        return "gray"
    return "mixed"


def _suggest_from_tags(tags: set[str]) -> tuple[str | None, float]:
    if {"roadway_scene", "damaged_surface_candidate"} <= tags:
        return "road_damage", 0.81
    if {"water_surface", "pooling_water"} <= tags:
        return "water_infrastructure", 0.78
    if {"vegetation", "branch_clutter"} <= tags:
        return "tree_maintenance", 0.72
    if {"night_scene", "poor_lighting_visual"} <= tags:
        return "street_lighting", 0.66
    return None, 0.0


def _summarize_tags(tags: list[str]) -> str:
    if not tags:
        return "image evidence was present but produced limited visual cues"
    phrases = {
        "roadway_scene": "a roadway-like surface",
        "damaged_surface_candidate": "a damaged surface region",
        "water_surface": "standing or flowing water",
        "pooling_water": "pooling water",
        "vegetation": "dense vegetation",
        "branch_clutter": "branch or tree clutter",
        "night_scene": "a dark low-light scene",
        "poor_lighting_visual": "limited scene lighting",
        "strong_edges": "strong structural edges",
        "gray_surface": "gray paved or concrete texture",
    }
    parts = [phrases.get(tag, tag.replace("_", " ")) for tag in tags[:3]]
    return "visual cues suggest " + ", ".join(parts)


def analyze_image_item(item: EvidenceItem, settings: Settings) -> ImageObservation | None:
    data = load_image_bytes(item, settings)
    if not data:
        return None
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
            rgb.thumbnail((256, 256))
            gray = rgb.convert("L")
            hsv = rgb.convert("HSV")

            width, height = rgb.size
            pixels = max(width * height, 1)
            rgb_stat = ImageStat.Stat(rgb)
            gray_stat = ImageStat.Stat(gray)
            sat_stat = ImageStat.Stat(hsv.getchannel("S"))
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)

            red, green, blue = rgb_stat.mean
            brightness = float(gray_stat.mean[0])
            contrast = float(gray_stat.stddev[0])
            saturation = float(sat_stat.mean[0])
            edge_density = float(edge_stat.mean[0]) / 255.0
            histogram = gray.histogram()
            dark_fraction = sum(histogram[:48]) / pixels

            left = width // 4
            upper = height // 4
            right = max(left + 1, width - left)
            lower = max(upper + 1, height - upper)
            center_crop = gray.crop((left, upper, right, lower))
            center_histogram = center_crop.histogram()
            center_pixels = max(center_crop.size[0] * center_crop.size[1], 1)
            center_dark_fraction = sum(center_histogram[:56]) / center_pixels

            dominant_color = _dominant_color_label(red, green, blue)
            tags: set[str] = set()
            if dominant_color == "gray":
                tags.add("gray_surface")
            if width >= height * 1.15 and dominant_color in {"gray", "mixed"}:
                tags.add("roadway_scene")
            if brightness < 72 or dark_fraction > 0.45:
                tags.add("night_scene")
            if edge_density > 0.12:
                tags.add("strong_edges")
            if dominant_color == "blue":
                tags.add("water_surface")
            if dominant_color == "green":
                tags.add("vegetation")
            if dominant_color in {"gray", "mixed"} and center_dark_fraction > 0.18 and contrast > 28:
                tags.add("damaged_surface_candidate")
            if "water_surface" in tags and edge_density < 0.16:
                tags.add("pooling_water")
            if "vegetation" in tags and edge_density > 0.11:
                tags.add("branch_clutter")
            if "night_scene" in tags and edge_density < 0.1:
                tags.add("poor_lighting_visual")

            suggested_category, confidence = _suggest_from_tags(tags)
            ordered_tags = tuple(sorted(tags))
            summary = _summarize_tags(list(ordered_tags))
            return ImageObservation(
                evidence_id=item.evidence_id,
                width=width,
                height=height,
                brightness=round(brightness, 2),
                contrast=round(contrast, 2),
                saturation=round(saturation, 2),
                edge_density=round(edge_density, 4),
                dark_fraction=round(dark_fraction, 4),
                center_dark_fraction=round(center_dark_fraction, 4),
                dominant_color=dominant_color,
                tags=ordered_tags,
                summary=summary,
                suggested_category=suggested_category,
                confidence=round(confidence, 2),
            )
    except (UnidentifiedImageError, OSError):
        logger.warning("Image evidence could not be decoded for local visual reasoning.")
        return None


def analyze_submission_images(submission: SubmissionInput, settings: Settings) -> VisualContext | None:
    observations = [
        observation
        for observation in (
            analyze_image_item(item, settings)
            for item in submission.evidence[:3]
            if item.kind == EvidenceKind.image
        )
        if observation is not None
    ]
    if not observations:
        return None

    tags = sorted({tag for observation in observations for tag in observation.tags})
    category_scores: dict[str, float] = {}
    for observation in observations:
        if observation.suggested_category:
            category_scores[observation.suggested_category] = category_scores.get(
                observation.suggested_category,
                0.0,
            ) + observation.confidence

    suggested_category = None
    confidence = 0.0
    if category_scores:
        suggested_category = max(category_scores, key=category_scores.get)
        confidence = round(
            min(0.92, category_scores[suggested_category] / max(len(observations), 1)),
            2,
        )

    summary = "; ".join(observation.summary for observation in observations[:2])
    return VisualContext(
        observation_count=len(observations),
        tags=tuple(tags),
        summary=summary,
        suggested_category=suggested_category,
        confidence=confidence,
    )
