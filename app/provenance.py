from __future__ import annotations

from app.models.domain import CaseModelContext, DecisionProvenance


PROVENANCE_SCHEMA_VERSION = "case-provenance.v1"
PROMPT_VERSION_NOT_APPLICABLE = "not_applicable"
THRESHOLD_VERSION_NOT_APPLICABLE = "not_applicable"

HEURISTIC_RULESET_VERSION = "heuristic-rules.v2"
LOCAL_RETRIEVAL_MODEL_VERSION = "local-retrieval-template.v2"
ROUTING_CLASSIFIER_VERSION = "routing-naive-bayes-corpus.v2"
PRIORITY_CLASSIFIER_VERSION = "priority-naive-bayes-corpus.v2"
STRUCTURED_OUTPUT_SCHEMA_VERSION = "structured-json-schema.v1"
REVIEW_POLICY_VERSION = "review-routing-policy.v2"
VERIFICATION_POLICY_VERSION = "geo-visual-policy.v2"
IMAGE_REASONING_VERSION = "pillow-vision-heuristics.v2"
MANUAL_WORKFLOW_VERSION = "manual-workflow.v1"
SUBMISSION_GUARD_POLICY_VERSION = "submission-guard-policy.v1"

INTAKE_PROMPT_VERSION = "intake-structured-prompt.v1"
ROUTING_PROMPT_VERSION = "routing-structured-prompt.v1"
PRIORITY_PROMPT_VERSION = "priority-structured-prompt.v1"
DRAFT_PROMPT_VERSION = "draft-structured-prompt.v1"
VERIFICATION_PROMPT_VERSION = "verification-structured-prompt.v1"
EXPLANATION_PROMPT_VERSION = "explanation-structured-prompt.v1"

LOCALFREE_INTAKE_TEMPLATE_PROMPT_VERSION = "localfree-intake-template.v1"
LOCALFREE_DRAFT_TEMPLATE_PROMPT_VERSION = "localfree-draft-template.v1"
LOCALFREE_EXPLANATION_TEMPLATE_PROMPT_VERSION = "localfree-explanation-template.v1"
LOCALFREE_OLLAMA_INTAKE_PROMPT_VERSION = "localfree-ollama-intake.v1"
LOCALFREE_OLLAMA_DRAFT_PROMPT_VERSION = "localfree-ollama-draft.v1"
LOCALFREE_OLLAMA_EXPLANATION_PROMPT_VERSION = "localfree-ollama-explanation.v1"

HEURISTIC_THRESHOLD_SET_VERSION = "heuristic-thresholds.v1"
LOCAL_ML_THRESHOLD_SET_VERSION = "local-ml-thresholds.v1"
REVIEW_THRESHOLD_SET_VERSION = "review-thresholds.v1"
VERIFICATION_THRESHOLD_SET_VERSION = "verification-thresholds.v2"
MANUAL_WORKFLOW_THRESHOLD_SET_VERSION = "manual-workflow-thresholds.v1"
SUBMISSION_GUARD_THRESHOLD_SET_VERSION = "submission-guard-thresholds.v1"


def _combine_versions(values: list[str]) -> str:
    unique = sorted({value for value in values if value and value != PROMPT_VERSION_NOT_APPLICABLE})
    if not unique:
        return PROMPT_VERSION_NOT_APPLICABLE
    return "+".join(unique)


def build_case_model_context(
    *,
    provider: str,
    model_name: str,
    model_version: str,
    stage_provenance: dict[str, DecisionProvenance],
) -> CaseModelContext:
    stage_copy = {
        stage: provenance.model_copy(deep=True)
        for stage, provenance in stage_provenance.items()
    }
    return CaseModelContext(
        provider=provider or "unknown",
        model_name=model_name or "unknown",
        model_version=model_version or "unknown",
        provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
        prompt_bundle_version=_combine_versions(
            [provenance.prompt_version for provenance in stage_copy.values()]
        ),
        classifier_bundle_version=_combine_versions(
            [
                provenance.classifier_version or PROMPT_VERSION_NOT_APPLICABLE
                for provenance in stage_copy.values()
            ]
        ),
        threshold_set_version=_combine_versions(
            [provenance.threshold_set_version for provenance in stage_copy.values()]
        ),
        stage_provenance=stage_copy,
    )


def merge_case_model_context(
    existing: CaseModelContext | None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    stage_updates: dict[str, DecisionProvenance] | None = None,
) -> CaseModelContext:
    baseline = existing or CaseModelContext()
    merged = {
        stage: provenance.model_copy(deep=True)
        for stage, provenance in baseline.stage_provenance.items()
    }
    for stage, provenance in (stage_updates or {}).items():
        merged[stage] = provenance.model_copy(deep=True)
    return build_case_model_context(
        provider=provider or baseline.provider,
        model_name=model_name or baseline.model_name,
        model_version=model_version or baseline.model_version,
        stage_provenance=merged,
    )
