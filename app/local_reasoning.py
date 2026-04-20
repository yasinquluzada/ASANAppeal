from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from app.config import Settings
from app.image_understanding import VisualContext
from app.local_ml import TrainingExample, load_training_examples
from app.models.api import SubmissionInput
from app.models.domain import (
    DraftAppeal,
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
)
from app.providers.heuristic import (
    CATEGORY_KEYWORDS,
    _clip_confidence,
    _first_sentence,
    _normalize,
    _slug_to_title,
    _tokenize,
)


COMMON_TOKENS = {
    "issue",
    "reported",
    "reporting",
    "citizen",
    "public",
    "service",
    "location",
    "near",
    "main",
    "road",
    "street",
    "today",
    "area",
    "problem",
}


@dataclass(frozen=True)
class ReasoningMatch:
    label: str
    source: str
    text: str
    score: float
    shared_terms: tuple[str, ...]


class LocalCaseReasoner:
    """Corpus-backed local reasoning for no-key operation without Ollama."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.routing_examples = load_training_examples(self.settings, task="routing")
        self.priority_examples = load_training_examples(self.settings, task="priority")
        self.routing_df = self._document_frequency(self.routing_examples)
        self.priority_df = self._document_frequency(self.priority_examples)

    def _document_frequency(self, examples: list[TrainingExample]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for example in examples:
            counts.update(set(_tokenize(example.text)))
        return counts

    def _idf(self, token: str, *, df: Counter[str], total_docs: int) -> float:
        return 1.0 + math.log((1 + total_docs) / (1 + df.get(token, 0)))

    def _similarity(
        self,
        query_tokens: set[str],
        document_tokens: set[str],
        *,
        df: Counter[str],
        total_docs: int,
    ) -> tuple[float, tuple[str, ...]]:
        shared = tuple(sorted(query_tokens & document_tokens))
        if not shared:
            return 0.0, ()
        numerator = sum(self._idf(token, df=df, total_docs=total_docs) for token in shared)
        denominator_tokens = query_tokens | document_tokens
        denominator = sum(
            self._idf(token, df=df, total_docs=total_docs) for token in denominator_tokens
        ) or 1.0
        return numerator / denominator, shared

    def retrieve_matches(
        self,
        text: str,
        *,
        task: str,
        label_filter: str | None = None,
        limit: int = 3,
    ) -> list[ReasoningMatch]:
        if task == "priority":
            examples = self.priority_examples
            df = self.priority_df
        else:
            examples = self.routing_examples
            df = self.routing_df
        query_tokens = set(_tokenize(text))
        if not query_tokens:
            return []

        matches: list[ReasoningMatch] = []
        total_docs = max(len(examples), 1)
        for example in examples:
            if label_filter and example.label != label_filter:
                continue
            score, shared_terms = self._similarity(
                query_tokens,
                set(_tokenize(example.text)),
                df=df,
                total_docs=total_docs,
            )
            if score <= 0:
                continue
            matches.append(
                ReasoningMatch(
                    label=example.label,
                    source=example.source,
                    text=example.text,
                    score=round(score, 4),
                    shared_terms=shared_terms,
                )
            )
        matches.sort(
            key=lambda item: (item.score, len(item.shared_terms), len(item.text)),
            reverse=True,
        )
        return matches[:limit]

    def build_issue(
        self,
        submission: SubmissionInput,
        heuristic_issue: StructuredIssue,
        *,
        classifier_label: str | None = None,
        classifier_confidence: float | None = None,
        visual_context: VisualContext | None = None,
    ) -> StructuredIssue:
        combined = _normalize(
            " ".join(
                filter(
                    None,
                    [
                        submission.citizen_text,
                        submission.location_hint,
                        submission.time_hint,
                        " ".join(
                            item.description or item.filename or item.uri or ""
                            for item in submission.evidence
                        ),
                    ],
                )
            )
        )
        matches = self.retrieve_matches(combined, task="routing", limit=3)
        label_scores: Counter[str] = Counter()
        for match in matches:
            label_scores[match.label] += match.score

        category = heuristic_issue.category
        if classifier_label:
            if category == "general_public_service" or (
                classifier_confidence is not None
                and classifier_confidence >= max(0.67, heuristic_issue.confidence + 0.05)
            ):
                category = classifier_label
        if matches:
            top_label, top_score = label_scores.most_common(1)[0]
            second_score = label_scores.most_common(2)[1][1] if len(label_scores) > 1 else 0.0
            if category == "general_public_service" and top_score >= max(0.12, second_score + 0.03):
                category = top_label
        if visual_context and visual_context.suggested_category:
            if category == "general_public_service" or visual_context.confidence >= max(
                0.72,
                heuristic_issue.confidence + 0.06,
            ):
                category = visual_context.suggested_category

        retrieved_terms = [
            term
            for match in matches
            for term in match.shared_terms
            if term not in COMMON_TOKENS
        ]
        keyword_hits = [
            keyword
            for keyword in CATEGORY_KEYWORDS.get(category, ())
            if keyword in combined
        ]
        extracted_signals = sorted(
            set(heuristic_issue.extracted_signals)
            | set(keyword_hits)
            | set(retrieved_terms[:6])
            | set(visual_context.tags if visual_context else ())
        )[:8]

        summary_seed = _first_sentence(submission.citizen_text)
        if len(summary_seed.strip()) < 24:
            summary_seed = f"{_slug_to_title(category)} reported"
        summary = summary_seed.rstrip(".")
        if submission.location_hint and submission.location_hint.lower() not in summary.lower():
            summary += f" near {submission.location_hint}"
        if extracted_signals:
            summary += f". Indicators include {', '.join(extracted_signals[:3])}"
        if visual_context and visual_context.summary and visual_context.summary not in summary:
            summary += f". Visual evidence suggests {visual_context.summary.replace('visual cues suggest ', '')}"
        summary = summary.rstrip(".") + "."

        missing_information = list(heuristic_issue.missing_information)
        if submission.evidence and not any(item.description for item in submission.evidence):
            missing_information.append("Add short descriptions for uploaded evidence items.")
        if category == "general_public_service" and not matches:
            missing_information.append("Name the affected public asset or service more specifically.")
        if submission.evidence and visual_context is None:
            missing_information.append("Attach clearer image evidence or add short descriptions for each image.")

        confidence_inputs = [heuristic_issue.confidence]
        if classifier_confidence is not None:
            confidence_inputs.append(classifier_confidence if category == classifier_label else classifier_confidence * 0.8)
        if matches:
            confidence_inputs.append(min(0.92, 0.44 + matches[0].score))
        if visual_context and visual_context.confidence > 0:
            confidence_inputs.append(visual_context.confidence)

        return StructuredIssue(
            category=category,
            issue_type=_slug_to_title(category),
            summary=summary,
            extracted_signals=extracted_signals,
            missing_information=sorted(set(missing_information)),
            confidence=_clip_confidence(sum(confidence_inputs) / len(confidence_inputs)),
        )

    def build_draft(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        heuristic_draft: DraftAppeal,
        visual_context: VisualContext | None = None,
    ) -> DraftAppeal:
        combined = _normalize(
            " ".join(
                filter(
                    None,
                    [
                        submission.citizen_text,
                        structured_issue.summary,
                        submission.location_hint,
                        routing.category,
                        priority.level.value,
                    ],
                )
            )
        )
        matches = self.retrieve_matches(
            combined,
            task="routing",
            label_filter=structured_issue.category,
            limit=2,
        )
        evidence_sentence = (
            f"I attached {len(submission.evidence)} evidence item(s) documenting the problem."
            if submission.evidence
            else "No evidence was attached yet, so a field check may still be needed."
        )
        if visual_context and visual_context.summary:
            evidence_sentence += f" Visual analysis indicates {visual_context.summary.replace('visual cues suggest ', '')}."
        impact_sentence = " ".join(priority.reasons[:2]) or "The issue still requires operational review."
        precedent_sentence = ""
        if matches:
            precedent_sentence = (
                " Comparable local patterns were found for "
                + ", ".join(match.text for match in matches[:2])
                + "."
            )

        body = "\n\n".join(
            [
                f"I respectfully submit this appeal regarding {structured_issue.summary.rstrip('.')}.",
                (
                    f"The reported location is {submission.location_hint or 'the stated location'}"
                    + (
                        f", observed around {submission.time_hint}."
                        if submission.time_hint
                        else "."
                    )
                ),
                evidence_sentence,
                (
                    f"The matter aligns with {structured_issue.issue_type} and should be handled by "
                    f"{routing.institution} / {routing.department}. Operational impact: {impact_sentence}"
                    f"{precedent_sentence}"
                ),
                (
                    f"Requested action: please inspect, document findings, and take "
                    f"{priority.level.value}-priority action until the issue is fully resolved."
                ),
            ]
        )
        checklist = [
            "Verify the exact dispatch location and nearby landmark.",
            "Confirm the generated issue category matches the actual civic problem.",
            "Check that attached evidence clearly shows the reported condition.",
        ]
        checklist.extend(structured_issue.missing_information[:2])
        if priority.reasons:
            checklist.append(f"Check urgency assumptions: {priority.reasons[0]}")

        confidence = _clip_confidence(
            (heuristic_draft.confidence + structured_issue.confidence + routing.confidence + priority.confidence)
            / 4
        )
        return DraftAppeal(
            title=heuristic_draft.title,
            body=body,
            citizen_review_checklist=checklist[:5],
            confidence=confidence,
        )

    def build_explanation(
        self,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        human_review: HumanReviewTask,
        heuristic_explanation: ExplanationNote,
        verification: VerificationDecision | None = None,
        visual_context: VisualContext | None = None,
    ) -> ExplanationNote:
        combined = _normalize(
            " ".join(
                filter(
                    None,
                    [
                        structured_issue.summary,
                        structured_issue.category,
                        routing.category,
                        routing.institution,
                        priority.level.value,
                    ],
                )
            )
        )
        matches = self.retrieve_matches(
            combined,
            task="routing",
            label_filter=structured_issue.category,
            limit=2,
        )
        detailed_rationale = [
            f"Local corpus reasoning aligned this case with {structured_issue.issue_type} patterns.",
            f"Routing selected {routing.institution} with confidence {routing.confidence:.2f}.",
            f"Priority was assessed as {priority.level.value} with confidence {priority.confidence:.2f}.",
        ]
        if priority.reasons:
            detailed_rationale.extend(priority.reasons[:2])
        if matches:
            detailed_rationale.append(
                "Comparable local patterns: "
                + "; ".join(match.text for match in matches)
                + "."
            )
        else:
            detailed_rationale.append("Comparable local patterns were limited, so confidence remained conservative.")
        if visual_context and visual_context.summary:
            detailed_rationale.append(f"Image evidence analysis: {visual_context.summary}.")
        if verification is not None:
            detailed_rationale.append(verification.summary)

        risk_flags = sorted(
            set(heuristic_explanation.risk_flags)
            | set(human_review.reasons)
            | set(verification.mismatch_flags if verification else [])
        )
        if not matches:
            risk_flags.append("limited_local_pattern_support")
        next_action = (
            f"Send to {human_review.queue} for manual review."
            if human_review.needed
            else f"Dispatch to {routing.department}."
        )
        summary = (
            f"{structured_issue.issue_type} case routed to {routing.institution} with "
            f"{priority.level.value} priority using local corpus-backed reasoning."
        )
        return ExplanationNote(
            summary=summary,
            next_action=next_action,
            detailed_rationale=detailed_rationale[:6],
            risk_flags=sorted(set(risk_flags)),
        )
