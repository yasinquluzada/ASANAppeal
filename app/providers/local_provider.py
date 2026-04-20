from __future__ import annotations

from typing import Any

from app.config import Settings
from app.image_understanding import VisualContext, analyze_submission_images
from app.local_ml import LocalPriorityModel, LocalRoutingModel, export_reviewed_feedback
from app.local_reasoning import LocalCaseReasoner
from app.models.api import SubmissionInput
from app.models.domain import (
    DraftAppeal,
    ExplanationNote,
    HumanReviewTask,
    PriorityDecision,
    PriorityLevel,
    RoutingDecision,
    StructuredIssue,
    VerificationDecision,
)
from app.ollama_client import OllamaLocalClient
from app.providers.heuristic import (
    ROUTING_TABLE,
    HeuristicAIProvider,
    _clip_confidence,
    _slug_to_title,
)
from app.provenance import (
    HEURISTIC_RULESET_VERSION,
    IMAGE_REASONING_VERSION,
    LOCALFREE_DRAFT_TEMPLATE_PROMPT_VERSION,
    LOCALFREE_EXPLANATION_TEMPLATE_PROMPT_VERSION,
    LOCALFREE_INTAKE_TEMPLATE_PROMPT_VERSION,
    LOCALFREE_OLLAMA_DRAFT_PROMPT_VERSION,
    LOCALFREE_OLLAMA_EXPLANATION_PROMPT_VERSION,
    LOCALFREE_OLLAMA_INTAKE_PROMPT_VERSION,
    LOCAL_ML_THRESHOLD_SET_VERSION,
    LOCAL_RETRIEVAL_MODEL_VERSION,
    PRIORITY_CLASSIFIER_VERSION,
    PROMPT_VERSION_NOT_APPLICABLE,
    ROUTING_CLASSIFIER_VERSION,
    STRUCTURED_OUTPUT_SCHEMA_VERSION,
    THRESHOLD_VERSION_NOT_APPLICABLE,
    VERIFICATION_POLICY_VERSION,
    VERIFICATION_THRESHOLD_SET_VERSION,
)
from app.verification_reasoning import POLICY_BY_CATEGORY, verify_resolution_advanced


class LocalFreeProvider(HeuristicAIProvider):
    """Zero-cost local provider with trainable classifiers and optional Ollama."""

    def __init__(self, settings: Settings | None = None, llm_backend: str | None = None) -> None:
        self.settings = settings or Settings()
        self.local_llm_backend = llm_backend or self.settings.local_llm_backend
        self.routing_model = None
        self.priority_model = None
        self.local_reasoner = LocalCaseReasoner(self.settings)
        self.last_local_ml_retrain: dict[str, object] | None = None
        self._reload_models()
        self.ollama_client = None
        if self.local_llm_backend in {"ollama", "auto"}:
            self.ollama_client = OllamaLocalClient(
                base_url=self.settings.ollama_url,
                model=self.settings.ollama_model,
                timeout_seconds=self.settings.ollama_timeout_seconds,
            )
        model_parts = ["localfree"]
        if self.settings.local_ml_enabled:
            model_parts.append("naive-bayes-corpus")
        model_parts.append("retrieval-template")
        if self.ollama_client:
            model_parts.append(f"ollama:{self.settings.ollama_model}")
        self._model = "+".join(model_parts)

    def _reload_models(self) -> None:
        self.routing_model = (
            LocalRoutingModel(self.settings) if self.settings.local_ml_enabled else None
        )
        self.priority_model = (
            LocalPriorityModel(self.settings) if self.settings.local_ml_enabled else None
        )

    def retrain_local_models(self) -> dict[str, object]:
        if not self.settings.local_ml_enabled:
            raise RuntimeError("Local ML retraining requires ASAN_LOCAL_ML_ENABLED=true.")
        export_result = export_reviewed_feedback(self.settings)
        self._reload_models()
        summary = {
            "retrained": True,
            "exported_at": export_result.exported_at,
            "sqlite_path": export_result.sqlite_path,
            "feedback_dir": export_result.feedback_dir,
            "routing_export_path": export_result.routing.output_path,
            "routing_exported_examples": export_result.routing.exported_examples,
            "priority_export_path": export_result.priority.output_path,
            "priority_exported_examples": export_result.priority.exported_examples,
            "routing_training_examples": (
                self.routing_model.training_examples_count if self.routing_model else 0
            ),
            "priority_training_examples": (
                self.priority_model.training_examples_count if self.priority_model else 0
            ),
            "routing_training_sources": (
                self.routing_model.training_sources if self.routing_model else []
            ),
            "priority_training_sources": (
                self.priority_model.training_sources if self.priority_model else []
            ),
        }
        self.last_local_ml_retrain = summary
        return summary

    def _ollama_probe(self, *, force_refresh: bool = False):
        if self.ollama_client is None:
            return None
        return self.ollama_client.probe(force_refresh=force_refresh)

    def diagnostics(self, *, force_refresh: bool = False) -> dict[str, object]:
        probe = self._ollama_probe(force_refresh=force_refresh)
        llm_status = "disabled"
        dependency_ok = True
        if probe is not None:
            llm_status = probe.status
            dependency_ok = probe.dependency_ok
        return {
            "local_llm_backend": self.local_llm_backend,
            "local_llm_status": llm_status,
            "local_llm_dependency_ok": dependency_ok,
            "local_llm_dependency_required": probe is not None,
            "local_llm_base_url": probe.base_url if probe else None,
            "local_llm_model_requested": probe.model_requested if probe else None,
            "local_llm_server_reachable": probe.server_reachable if probe else False,
            "local_llm_model_available": probe.model_available if probe else False,
            "local_llm_available_model_count": probe.available_model_count if probe else 0,
            "routing_model": "naive-bayes" if self.routing_model else "disabled",
            "priority_model": "naive-bayes" if self.priority_model else "disabled",
            "local_reasoning_backend": "retrieval-template",
            "local_reasoning_routing_examples": len(self.local_reasoner.routing_examples),
            "local_reasoning_priority_examples": len(self.local_reasoner.priority_examples),
            "local_image_reasoning_backend": "pillow-vision-heuristics",
            "local_verification_backend": "geo-visual-policy",
            "routing_model_training_examples": (
                self.routing_model.training_examples_count if self.routing_model else 0
            ),
            "routing_model_training_sources": (
                self.routing_model.training_sources if self.routing_model else []
            ),
            "routing_model_label_counts": (
                self.routing_model.label_counts if self.routing_model else {}
            ),
            "priority_model_training_examples": (
                self.priority_model.training_examples_count if self.priority_model else 0
            ),
            "priority_model_training_sources": (
                self.priority_model.training_sources if self.priority_model else []
            ),
            "priority_model_label_counts": (
                self.priority_model.label_counts if self.priority_model else {}
            ),
            "local_ml_feedback_dir": self.settings.local_ml_feedback_dir,
            "local_ml_last_retrain": self.last_local_ml_retrain,
        }

    def _can_use_ollama(self) -> bool:
        if self.ollama_client is None:
            return False
        probe = self._ollama_probe()
        return bool(probe and probe.dependency_ok)

    def _validate_structured_issue(self, payload: dict[str, Any]) -> StructuredIssue | None:
        try:
            issue = StructuredIssue.model_validate(payload)
        except Exception:
            return None
        if issue.category not in ROUTING_TABLE:
            return None
        return issue

    def _validate_draft(self, payload: dict[str, Any]) -> DraftAppeal | None:
        try:
            return DraftAppeal.model_validate(payload)
        except Exception:
            return None

    def _validate_explanation(self, payload: dict[str, Any]) -> ExplanationNote | None:
        try:
            return ExplanationNote.model_validate(payload)
        except Exception:
            return None

    def analyze_submission(self, submission: SubmissionInput) -> StructuredIssue:
        heuristic_issue = super().analyze_submission(submission)
        visual_context = analyze_submission_images(submission, self.settings)
        visual_observation_count = visual_context.observation_count if visual_context else 0
        routing_prediction = (
            self.routing_model.predict(submission, heuristic_issue) if self.routing_model else None
        )
        retrieval_issue = self.local_reasoner.build_issue(
            submission,
            heuristic_issue,
            classifier_label=routing_prediction.label if routing_prediction else None,
            classifier_confidence=routing_prediction.confidence if routing_prediction else None,
            visual_context=visual_context,
        )
        if not self._can_use_ollama():
            self._record_stage_provenance(
                "intake",
                provider=type(self).__name__,
                engine="retrieval-template",
                model_name="local-retrieval-template",
                model_version=LOCAL_RETRIEVAL_MODEL_VERSION,
                prompt_version=LOCALFREE_INTAKE_TEMPLATE_PROMPT_VERSION,
                classifier_version=(
                    ROUTING_CLASSIFIER_VERSION if routing_prediction is not None else None
                ),
                threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
                thresholds={
                    "local_ml_min_confidence": self.settings.local_ml_min_confidence,
                    "visual_observation_count": visual_observation_count,
                },
                notes=[
                    "routing-classifier-signal-used" if routing_prediction is not None else "routing-classifier-unavailable",
                    "local-image-context-used" if visual_observation_count else "no-image-context",
                ],
            )
            return retrieval_issue

        categories = ", ".join(ROUTING_TABLE.keys())
        payload = self.ollama_client.analyze_submission(
            submission,
            system_prompt=(
                "You classify municipal civic complaints. "
                "Return strict JSON with the keys category, issue_type, summary, "
                "extracted_signals, missing_information, confidence. "
                f"category must be one of: {categories}."
            ),
            user_prompt=(
                "Citizen text:\n"
                f"{submission.citizen_text}\n\n"
                f"Location hint: {submission.location_hint or 'n/a'}\n"
                f"Time hint: {submission.time_hint or 'n/a'}\n"
                "Evidence descriptions:\n"
                + "\n".join(
                    f"- {item.description or item.filename or item.uri or 'evidence item'}"
                    for item in submission.evidence
                )
            ),
        )
        issue = self._validate_structured_issue(payload or {})
        if issue is None:
            self._record_stage_provenance(
                "intake",
                provider=type(self).__name__,
                engine="heuristic-rules",
                model_name="heuristic-intake",
                model_version=HEURISTIC_RULESET_VERSION,
                prompt_version=PROMPT_VERSION_NOT_APPLICABLE,
                classifier_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
                threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
                thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
                notes=["ollama-response-invalid", "fell-back-to-heuristic-intake"],
            )
            return heuristic_issue
        if issue.category == "general_public_service" and heuristic_issue.category != "general_public_service":
            self._record_stage_provenance(
                "intake",
                provider=type(self).__name__,
                engine="heuristic-rules",
                model_name="heuristic-intake",
                model_version=HEURISTIC_RULESET_VERSION,
                prompt_version=LOCALFREE_OLLAMA_INTAKE_PROMPT_VERSION,
                classifier_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
                threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
                thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
                notes=["ollama-generalized-category", "heuristic-category-retained"],
            )
            return heuristic_issue

        merged_issue = StructuredIssue(
            category=issue.category,
            issue_type=issue.issue_type or _slug_to_title(issue.category),
            summary=issue.summary or retrieval_issue.summary,
            extracted_signals=sorted(
                set(issue.extracted_signals) | set(retrieval_issue.extracted_signals)
            )[:8],
            missing_information=sorted(
                set(issue.missing_information) | set(retrieval_issue.missing_information)
            ),
            confidence=_clip_confidence(max(issue.confidence, retrieval_issue.confidence)),
        )
        self._record_stage_provenance(
            "intake",
            provider=type(self).__name__,
            engine="ollama-structured-merge",
            model_name=self.settings.ollama_model,
            model_version=self.settings.ollama_model,
            prompt_version=LOCALFREE_OLLAMA_INTAKE_PROMPT_VERSION,
            classifier_version=(
                ROUTING_CLASSIFIER_VERSION if routing_prediction is not None else STRUCTURED_OUTPUT_SCHEMA_VERSION
            ),
            threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
            thresholds={
                "local_ml_min_confidence": self.settings.local_ml_min_confidence,
                "visual_observation_count": visual_observation_count,
            },
            notes=["merged-with-retrieval-template"],
        )
        return merged_issue

    def route_issue(
        self, submission: SubmissionInput, structured_issue: StructuredIssue
    ) -> RoutingDecision:
        heuristic_routing = super().route_issue(submission, structured_issue)
        if self.routing_model is None:
            self._record_stage_provenance(
                "routing",
                provider=type(self).__name__,
                engine="heuristic-routing",
                model_name="heuristic-routing",
                model_version=HEURISTIC_RULESET_VERSION,
                classifier_version=None,
                threshold_set_version=THRESHOLD_VERSION_NOT_APPLICABLE,
                notes=["local-ml-disabled"],
            )
            return heuristic_routing

        prediction = self.routing_model.predict(submission, structured_issue)
        if prediction is None:
            self._record_stage_provenance(
                "routing",
                provider=type(self).__name__,
                engine="heuristic-routing",
                model_name="heuristic-routing",
                model_version=HEURISTIC_RULESET_VERSION,
                classifier_version=ROUTING_CLASSIFIER_VERSION,
                threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
                thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
                notes=["classifier-returned-no-prediction"],
            )
            return heuristic_routing

        should_override = (
            structured_issue.category == "general_public_service"
            or prediction.label == heuristic_routing.category
            or prediction.confidence >= max(
                0.82, heuristic_routing.confidence + self.settings.local_ml_min_confidence / 3
            )
        )
        if not should_override:
            self._record_stage_provenance(
                "routing",
                provider=type(self).__name__,
                engine="local-ml-assisted-routing",
                model_name="heuristic-routing",
                model_version=HEURISTIC_RULESET_VERSION,
                classifier_version=ROUTING_CLASSIFIER_VERSION,
                threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
                thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
                notes=["heuristic-routing-retained"],
            )
            return heuristic_routing

        institution, department = ROUTING_TABLE.get(
            prediction.label, ROUTING_TABLE["general_public_service"]
        )
        confidence = _clip_confidence(
            max(heuristic_routing.confidence, (heuristic_routing.confidence + prediction.confidence) / 2)
        )
        rationale = (
            f"Local routing classifier predicted {prediction.label} "
            f"with confidence {prediction.confidence:.2f}. "
            f"{heuristic_routing.rationale}"
        )
        routing = RoutingDecision(
            institution=institution,
            department=department,
            category=prediction.label,
            rationale=rationale,
            confidence=confidence,
        )
        self._record_stage_provenance(
            "routing",
            provider=type(self).__name__,
            engine="local-naive-bayes-classifier",
            model_name="routing-naive-bayes-corpus",
            model_version=ROUTING_CLASSIFIER_VERSION,
            classifier_version=ROUTING_CLASSIFIER_VERSION,
            threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
            thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
            notes=["classifier-override-applied"],
        )
        return routing

    def assess_priority(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
    ) -> PriorityDecision:
        heuristic_priority = super().assess_priority(submission, structured_issue, routing)
        if self.priority_model is None:
            self._record_stage_provenance(
                "priority",
                provider=type(self).__name__,
                engine="heuristic-priority",
                model_name="heuristic-priority",
                model_version=HEURISTIC_RULESET_VERSION,
                threshold_set_version=THRESHOLD_VERSION_NOT_APPLICABLE,
                notes=["local-ml-disabled"],
            )
            return heuristic_priority

        prediction = self.priority_model.predict(submission, structured_issue, routing)
        if prediction is None:
            self._record_stage_provenance(
                "priority",
                provider=type(self).__name__,
                engine="heuristic-priority",
                model_name="heuristic-priority",
                model_version=HEURISTIC_RULESET_VERSION,
                classifier_version=PRIORITY_CLASSIFIER_VERSION,
                threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
                thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
                notes=["classifier-returned-no-prediction"],
            )
            return heuristic_priority

        predicted_level = PriorityLevel(prediction.label)
        heuristic_rank = list(PriorityLevel).index(heuristic_priority.level)
        predicted_rank = list(PriorityLevel).index(predicted_level)
        should_override = (
            predicted_level == heuristic_priority.level
            or heuristic_priority.confidence < self.settings.local_ml_min_confidence
            or (predicted_rank > heuristic_rank and prediction.confidence >= 0.8)
        )
        if not should_override:
            self._record_stage_provenance(
                "priority",
                provider=type(self).__name__,
                engine="local-ml-assisted-priority",
                model_name="heuristic-priority",
                model_version=HEURISTIC_RULESET_VERSION,
                classifier_version=PRIORITY_CLASSIFIER_VERSION,
                threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
                thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
                notes=["heuristic-priority-retained"],
            )
            return heuristic_priority

        anchor_score = {
            PriorityLevel.low: 22,
            PriorityLevel.medium: 48,
            PriorityLevel.high: 74,
            PriorityLevel.critical: 92,
        }[predicted_level]
        score = round((heuristic_priority.score + anchor_score) / 2)
        if predicted_rank > heuristic_rank:
            score = max(score, anchor_score - 4)

        reasons = list(heuristic_priority.reasons)
        reasons.append(
            f"Local priority classifier predicted {predicted_level.value} urgency "
            f"with confidence {prediction.confidence:.2f}."
        )
        confidence = _clip_confidence(
            max(heuristic_priority.confidence, (heuristic_priority.confidence + prediction.confidence) / 2)
        )

        priority_decision = PriorityDecision(
            level=predicted_level,
            score=score,
            reasons=reasons,
            confidence=confidence,
            requires_human_review=(
                heuristic_priority.requires_human_review
                or predicted_level == PriorityLevel.critical
                or confidence < self.settings.local_ml_min_confidence
            ),
        )
        self._record_stage_provenance(
            "priority",
            provider=type(self).__name__,
            engine="local-naive-bayes-classifier",
            model_name="priority-naive-bayes-corpus",
            model_version=PRIORITY_CLASSIFIER_VERSION,
            classifier_version=PRIORITY_CLASSIFIER_VERSION,
            threshold_set_version=LOCAL_ML_THRESHOLD_SET_VERSION,
            thresholds={"local_ml_min_confidence": self.settings.local_ml_min_confidence},
            notes=["classifier-override-applied"],
        )
        return priority_decision

    def draft_appeal(
        self,
        submission: SubmissionInput,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
    ) -> DraftAppeal:
        heuristic_draft = super().draft_appeal(submission, structured_issue, routing, priority)
        visual_context = analyze_submission_images(submission, self.settings)
        visual_observation_count = visual_context.observation_count if visual_context else 0
        if not self._can_use_ollama():
            draft = self.local_reasoner.build_draft(
                submission,
                structured_issue,
                routing,
                priority,
                heuristic_draft,
                visual_context=visual_context,
            )
            self._record_stage_provenance(
                "draft",
                provider=type(self).__name__,
                engine="retrieval-template",
                model_name="local-retrieval-template",
                model_version=LOCAL_RETRIEVAL_MODEL_VERSION,
                prompt_version=LOCALFREE_DRAFT_TEMPLATE_PROMPT_VERSION,
                threshold_set_version=THRESHOLD_VERSION_NOT_APPLICABLE,
                thresholds={"visual_observation_count": visual_observation_count},
                notes=["local-draft-template"],
            )
            return draft

        payload = self.ollama_client.chat_json(
            system_prompt=(
                "You draft municipal complaint letters. "
                "Return strict JSON with the keys title, body, citizen_review_checklist, confidence."
            ),
            user_prompt=(
                f"Issue summary: {structured_issue.summary}\n"
                f"Category: {structured_issue.issue_type}\n"
                f"Location: {submission.location_hint or 'n/a'}\n"
                f"Time: {submission.time_hint or 'n/a'}\n"
                f"Route institution: {routing.institution}\n"
                f"Priority: {priority.level.value}\n"
                f"Evidence count: {len(submission.evidence)}"
            ),
        )
        draft = self._validate_draft(payload or {})
        if draft is None:
            self._record_stage_provenance(
                "draft",
                provider=type(self).__name__,
                engine="heuristic-rules",
                model_name="heuristic-draft",
                model_version=HEURISTIC_RULESET_VERSION,
                prompt_version=LOCALFREE_OLLAMA_DRAFT_PROMPT_VERSION,
                classifier_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
                notes=["ollama-response-invalid", "fell-back-to-heuristic-draft"],
            )
            return heuristic_draft
        merged_draft = DraftAppeal(
            title=draft.title or heuristic_draft.title,
            body=draft.body or heuristic_draft.body,
            citizen_review_checklist=(
                draft.citizen_review_checklist or heuristic_draft.citizen_review_checklist
            ),
            confidence=_clip_confidence(max(draft.confidence, heuristic_draft.confidence)),
        )
        self._record_stage_provenance(
            "draft",
            provider=type(self).__name__,
            engine="ollama-local-llm",
            model_name=self.settings.ollama_model,
            model_version=self.settings.ollama_model,
            prompt_version=LOCALFREE_OLLAMA_DRAFT_PROMPT_VERSION,
            classifier_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
            notes=["merged-with-heuristic-draft"],
        )
        return merged_draft

    def verify_resolution(
        self,
        original_submission,
        structured_issue,
        institution_response,
    ) -> VerificationDecision:
        verification = verify_resolution_advanced(
            original_submission,
            structured_issue,
            institution_response,
            settings=self.settings,
        )
        policy = POLICY_BY_CATEGORY.get(
            structured_issue.category,
            POLICY_BY_CATEGORY["general_public_service"],
        )
        self._record_stage_provenance(
            "verification",
            provider=type(self).__name__,
            engine="geo-visual-policy",
            model_name="geo-visual-verifier",
            model_version=VERIFICATION_POLICY_VERSION,
            threshold_set_version=VERIFICATION_THRESHOLD_SET_VERSION,
            thresholds={
                "same_place_yes_score": 0.66,
                "same_place_no_score": 0.28,
                "coordinate_exact_match_m": 150,
                "coordinate_near_match_m": 400,
                "policy_visual_threshold": float(policy["visual_threshold"]),
            },
            notes=[
                f"policy_owner={policy['institution']}",
                f"image_reasoning={IMAGE_REASONING_VERSION}",
            ],
        )
        return verification

    def explain_case(
        self,
        structured_issue: StructuredIssue,
        routing: RoutingDecision,
        priority: PriorityDecision,
        human_review: HumanReviewTask,
        verification: VerificationDecision | None = None,
    ) -> ExplanationNote:
        heuristic_explanation = super().explain_case(
            structured_issue,
            routing,
            priority,
            human_review,
            verification,
        )
        visual_context = None
        if any(
            signal in structured_issue.extracted_signals
            for signal in {"damaged_surface_candidate", "water_surface", "vegetation", "poor_lighting_visual"}
        ):
            visual_context = VisualContext(
                observation_count=1,
                tags=tuple(),
                summary="uploaded imagery contributed concrete visual cues",
                suggested_category=None,
                confidence=0.0,
            )
        if not self._can_use_ollama():
            explanation = self.local_reasoner.build_explanation(
                structured_issue,
                routing,
                priority,
                human_review,
                heuristic_explanation,
                verification,
                visual_context=visual_context,
            )
            self._record_stage_provenance(
                "explanation",
                provider=type(self).__name__,
                engine="retrieval-template",
                model_name="local-retrieval-template",
                model_version=LOCAL_RETRIEVAL_MODEL_VERSION,
                prompt_version=LOCALFREE_EXPLANATION_TEMPLATE_PROMPT_VERSION,
                notes=["local-explanation-template"],
            )
            return explanation

        payload = self.ollama_client.chat_json(
            system_prompt=(
                "You explain civic-case routing and priority decisions. "
                "Return strict JSON with the keys summary, next_action, detailed_rationale, risk_flags."
            ),
            user_prompt=(
                f"Issue: {structured_issue.summary}\n"
                f"Category: {structured_issue.category}\n"
                f"Routing institution: {routing.institution}\n"
                f"Priority: {priority.level.value}\n"
                f"Human review needed: {human_review.needed}\n"
                f"Verification summary: {verification.summary if verification else 'n/a'}"
            ),
        )
        explanation = self._validate_explanation(payload or {})
        if explanation is None:
            self._record_stage_provenance(
                "explanation",
                provider=type(self).__name__,
                engine="heuristic-rules",
                model_name="heuristic-explanation",
                model_version=HEURISTIC_RULESET_VERSION,
                prompt_version=LOCALFREE_OLLAMA_EXPLANATION_PROMPT_VERSION,
                classifier_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
                notes=["ollama-response-invalid", "fell-back-to-heuristic-explanation"],
            )
            return heuristic_explanation
        merged_explanation = ExplanationNote(
            summary=explanation.summary or heuristic_explanation.summary,
            next_action=explanation.next_action or heuristic_explanation.next_action,
            detailed_rationale=(
                explanation.detailed_rationale or heuristic_explanation.detailed_rationale
            ),
            risk_flags=sorted(set(explanation.risk_flags) | set(heuristic_explanation.risk_flags)),
        )
        self._record_stage_provenance(
            "explanation",
            provider=type(self).__name__,
            engine="ollama-local-llm",
            model_name=self.settings.ollama_model,
            model_version=self.settings.ollama_model,
            prompt_version=LOCALFREE_OLLAMA_EXPLANATION_PROMPT_VERSION,
            classifier_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
            notes=["merged-with-heuristic-explanation"],
        )
        return merged_explanation
