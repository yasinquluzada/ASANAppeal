const state = {
  activeTab: "citizen",
  selectedCaseId: null,
  reviewQueueItems: [],
  authToken: localStorage.getItem("asanappeal.authToken"),
  currentUser: null,
};

const elements = {
  authForm: document.getElementById("auth-form"),
  authUsername: document.getElementById("auth-username"),
  authPassword: document.getElementById("auth-password"),
  authLogout: document.getElementById("auth-logout"),
  authStatus: document.getElementById("auth-status"),
  healthStatus: document.getElementById("health-status"),
  healthDetail: document.getElementById("health-detail"),
  refreshAll: document.getElementById("refresh-all"),
  citizenForm: document.getElementById("citizen-form"),
  citizenFiles: document.getElementById("citizen-files"),
  citizenFileList: document.getElementById("citizen-file-list"),
  citizenResult: document.getElementById("citizen-result"),
  reviewFilters: document.getElementById("review-filters"),
  reviewQueueList: document.getElementById("review-queue-list"),
  reviewQueueMeta: document.getElementById("review-queue-meta"),
  reviewCaseTitle: document.getElementById("review-case-title"),
  reviewCaseDetail: document.getElementById("review-case-detail"),
  reviewCaseId: document.getElementById("review-case-id"),
  reviewOpenCase: document.getElementById("review-open-case"),
  reviewActorId: document.getElementById("review-actor-id"),
  reviewAssigneeId: document.getElementById("review-assignee-id"),
  reviewNote: document.getElementById("review-note"),
  refreshAnalytics: document.getElementById("refresh-analytics"),
  analyticsCards: document.getElementById("analytics-cards"),
  chartStatus: document.getElementById("chart-status"),
  chartPriority: document.getElementById("chart-priority"),
  chartInstitution: document.getElementById("chart-institution"),
  institutionForm: document.getElementById("institution-form"),
  institutionFiles: document.getElementById("institution-files"),
  institutionFileList: document.getElementById("institution-file-list"),
  institutionResult: document.getElementById("institution-result"),
};

function selectTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tab === tabName);
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.panel === tabName);
  });
}

async function apiRequest(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.authToken) {
    headers.set("Authorization", `Bearer ${state.authToken}`);
  }
  const response = await fetch(url, { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await response.json();
  } else {
    payload = await response.text();
  }
  if (!response.ok) {
    const detail = typeof payload === "object" && payload && "detail" in payload ? payload.detail : payload;
    if (response.status === 401) {
      clearAuthState();
    }
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

function setAuthState(token, user) {
  state.authToken = token;
  state.currentUser = user;
  if (token) {
    localStorage.setItem("asanappeal.authToken", token);
  } else {
    localStorage.removeItem("asanappeal.authToken");
  }
  renderAuthState();
}

function clearAuthState() {
  setAuthState(null, null);
}

function renderAuthState() {
  if (!state.currentUser) {
    elements.authStatus.textContent =
      "Sign in with a citizen, operator, reviewer, institution, or admin account to use the secured API.";
    return;
  }
  const parts = [
    `${state.currentUser.display_name} (${state.currentUser.role})`,
    `@${state.currentUser.username}`,
  ];
  if (state.currentUser.institution_slug) {
    parts.push(`institution: ${state.currentUser.institution_slug}`);
  }
  elements.authStatus.textContent = parts.join(" · ");
}

function setResult(element, value, { asJson = true } = {}) {
  if (asJson) {
    element.innerHTML = `<pre class="json-block">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
    return;
  }
  element.textContent = value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderSelectedFiles(input, listElement) {
  const files = Array.from(input.files || []);
  if (!files.length) {
    listElement.innerHTML = "<li>No files selected.</li>";
    return;
  }
  listElement.innerHTML = files
    .map((file) => `<li>${escapeHtml(file.name)} · ${Math.round(file.size / 1024)} KB</li>`)
    .join("");
}

function detectEvidenceKind(file) {
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("text/")) return "text";
  return "image";
}

async function uploadEvidenceFiles(fileList) {
  const evidenceItems = [];
  for (const file of Array.from(fileList || [])) {
    const params = new URLSearchParams({
      kind: detectEvidenceKind(file),
      filename: file.name,
      description: file.name,
    });
    const payload = await apiRequest(`/v1/evidence/upload?${params.toString()}`, {
      method: "POST",
      headers: {
        "Content-Type": file.type || "application/octet-stream",
      },
      body: await file.arrayBuffer(),
    });
    evidenceItems.push(payload.evidence_item);
  }
  return evidenceItems;
}

async function refreshHealth() {
  try {
    const health = await apiRequest("/health");
    elements.healthStatus.textContent = `${health.status.toUpperCase()} · ${health.active_provider}`;
    const repository = health.repository || health.repository_backend;
    elements.healthDetail.textContent = [
      `Repo: ${repository}`,
      health.local_llm_status ? `LLM: ${health.local_llm_status}` : null,
      health.evidence_object_count !== undefined ? `Evidence: ${health.evidence_object_count}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
  } catch (error) {
    elements.healthStatus.textContent = "UNAVAILABLE";
    elements.healthDetail.textContent = error.message;
  }
}

async function refreshCurrentUser() {
  if (!state.authToken) {
    renderAuthState();
    return null;
  }
  try {
    const user = await apiRequest("/v1/auth/me");
    state.currentUser = user;
    renderAuthState();
    return user;
  } catch (error) {
    clearAuthState();
    throw error;
  }
}

async function submitAuthForm(event) {
  event.preventDefault();
  try {
    const payload = await apiRequest("/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: elements.authUsername.value.trim(),
        password: elements.authPassword.value,
      }),
    });
    setAuthState(payload.access_token, payload.user);
    elements.authPassword.value = "";
    await Promise.allSettled([refreshReviewQueue(), refreshAnalytics()]);
  } catch (error) {
    elements.authStatus.textContent = error.message;
  }
}

async function logoutCurrentUser() {
  if (!state.authToken) {
    clearAuthState();
    return;
  }
  try {
    await apiRequest("/v1/auth/logout", { method: "POST" });
  } catch (error) {
    // Clearing local state keeps the portal consistent even if the token already expired.
  }
  clearAuthState();
}

function summarizeCase(casePayload) {
  return {
    case_id: casePayload.case_id,
    status: casePayload.status,
    category: casePayload.structured_issue.category,
    issue_type: casePayload.structured_issue.issue_type,
    queue: casePayload.human_review.queue,
    review_needed: casePayload.human_review.needed,
    institution: casePayload.routing.institution,
    department: casePayload.routing.department,
    priority: casePayload.priority.level,
    priority_score: casePayload.priority.score,
    draft_title: casePayload.draft.title,
    explanation: casePayload.explanation.summary,
  };
}

async function submitCitizenForm(event) {
  event.preventDefault();
  elements.citizenResult.textContent = "Uploading evidence and processing the submission…";
  try {
    const evidence = await uploadEvidenceFiles(elements.citizenFiles.files);
    const payload = {
      submission: {
        citizen_text: document.getElementById("citizen-text").value,
        language: document.getElementById("citizen-language").value,
        location_hint: document.getElementById("citizen-location").value || null,
        time_hint: document.getElementById("citizen-time").value || null,
        evidence,
      },
    };
    const result = await apiRequest("/v1/cases/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setResult(elements.citizenResult, {
      summary: summarizeCase(result.case),
      draft_body: result.case.draft.body,
      original_request: result.original_request,
    });
    state.selectedCaseId = result.case.case_id;
    elements.reviewCaseId.value = result.case.case_id;
    await refreshReviewQueue();
  } catch (error) {
    setResult(elements.citizenResult, error.message, { asJson: false });
  }
}

function reviewFilterUrl() {
  const params = new URLSearchParams({
    page: "1",
    page_size: "20",
    sort_by: document.getElementById("review-sort-filter").value,
  });
  const queue = document.getElementById("review-queue-filter").value;
  const assignment = document.getElementById("review-assignment-filter").value;
  const priority = document.getElementById("review-priority-filter").value;
  if (queue) params.set("review_queue", queue);
  if (assignment) params.set("assignment_state", assignment);
  if (priority) params.set("priority_level", priority);
  return `/v1/review-queue?${params.toString()}`;
}

function renderQueue(items) {
  if (!items.length) {
    elements.reviewQueueList.innerHTML = '<div class="empty-state">No cases match the current filters.</div>';
    return;
  }
  elements.reviewQueueList.innerHTML = items
    .map(
      (item) => `
        <button class="queue-item ${item.case_id === state.selectedCaseId ? "is-selected" : ""}" data-case-id="${item.case_id}" type="button">
          <div class="queue-item-top">
            <strong>${escapeHtml(item.issue_type)}</strong>
            <span class="pill priority-${escapeHtml(item.priority_level)}">${escapeHtml(item.priority_level)}</span>
          </div>
          <p>${escapeHtml(item.submission_excerpt)}</p>
          <div class="mini-grid">
            <span class="pill">${escapeHtml(item.review_queue)}</span>
            <span>${escapeHtml(item.assignee_id || "unassigned")}</span>
          </div>
        </button>
      `,
    )
    .join("");
  elements.reviewQueueList.querySelectorAll("[data-case-id]").forEach((button) => {
    button.addEventListener("click", () => openCase(button.dataset.caseId));
  });
}

async function refreshReviewQueue(event) {
  if (event) event.preventDefault();
  try {
    const queue = await apiRequest(reviewFilterUrl());
    state.reviewQueueItems = queue.items;
    elements.reviewQueueMeta.textContent = `${queue.meta.total_items} cases · sorted by ${queue.meta.sort_by}`;
    renderQueue(queue.items);
    if (state.selectedCaseId) {
      const stillPresent = queue.items.some((item) => item.case_id === state.selectedCaseId);
      if (!stillPresent) {
        await openCase(state.selectedCaseId);
      }
    }
  } catch (error) {
    elements.reviewQueueList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function formatHistory(casePayload) {
  const transitions = casePayload.operations.transition_history || [];
  const workflow = casePayload.operations.workflow_history || [];
  return {
    transitions,
    workflow,
  };
}

async function openCase(caseId) {
  if (!caseId) return;
  try {
    const [payload, auditLog] = await Promise.all([
      apiRequest(`/v1/cases/${encodeURIComponent(caseId)}`),
      apiRequest(`/v1/cases/${encodeURIComponent(caseId)}/audit-log`),
    ]);
    state.selectedCaseId = caseId;
    elements.reviewCaseId.value = caseId;
    elements.reviewCaseTitle.textContent = `${payload.case.case_id} · ${payload.case.structured_issue.issue_type}`;
    setResult(elements.reviewCaseDetail, {
      summary: summarizeCase(payload.case),
      human_review: payload.case.human_review,
      operations: payload.case.operations,
      history: formatHistory(payload.case),
      audit_log: auditLog.items,
      original_request: payload.original_request,
    });
    renderQueue(state.reviewQueueItems);
  } catch (error) {
    setResult(elements.reviewCaseDetail, error.message, { asJson: false });
  }
}

async function performWorkflowAction(action) {
  if (!state.selectedCaseId) {
    setResult(elements.reviewCaseDetail, "Open a case first.", { asJson: false });
    return;
  }
  try {
    const payload = await apiRequest(
      `/v1/cases/${encodeURIComponent(state.selectedCaseId)}/workflow-actions`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          actor_id: elements.reviewActorId.value || null,
          assignee_id: elements.reviewAssigneeId.value || null,
          note: elements.reviewNote.value || null,
        }),
      },
    );
    await openCase(state.selectedCaseId);
    await refreshReviewQueue();
    setResult(elements.reviewCaseDetail, {
      action_executed: action,
      summary: summarizeCase(payload.case),
      human_review: payload.case.human_review,
      operations: payload.case.operations,
      history: formatHistory(payload.case),
    });
  } catch (error) {
    setResult(elements.reviewCaseDetail, error.message, { asJson: false });
  }
}

function renderMetricCards(summary) {
  const cards = [
    ["Total Cases", summary.total_cases],
    ["Needs Review", summary.review_needed_cases],
    ["Distinct Statuses", Object.keys(summary.counts_by_status || {}).length],
    ["Distinct Institutions", Object.keys(summary.counts_by_institution || {}).length],
  ];
  elements.analyticsCards.innerHTML = cards
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <p class="panel-kicker">${escapeHtml(label)}</p>
          <p class="metric-card-value">${escapeHtml(String(value))}</p>
        </div>
      `,
    )
    .join("");
}

function renderBarChart(container, mapping) {
  const entries = Object.entries(mapping || {});
  if (!entries.length) {
    container.innerHTML = '<div class="empty-state">No data yet.</div>';
    return;
  }
  const max = Math.max(...entries.map(([, value]) => Number(value)));
  container.innerHTML = entries
    .map(([label, value]) => {
      const width = max > 0 ? (Number(value) / max) * 100 : 0;
      return `
        <div class="bar-row">
          <div>${escapeHtml(label)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${escapeHtml(String(value))}</strong>
        </div>
      `;
    })
    .join("");
}

async function refreshAnalytics() {
  try {
    const payload = await apiRequest("/v1/analytics/summary");
    renderMetricCards(payload.summary);
    renderBarChart(elements.chartStatus, payload.summary.counts_by_status);
    renderBarChart(elements.chartPriority, payload.summary.counts_by_priority);
    renderBarChart(elements.chartInstitution, payload.summary.counts_by_institution);
  } catch (error) {
    elements.analyticsCards.innerHTML = "";
    setResult(elements.chartStatus, error.message, { asJson: false });
    setResult(elements.chartPriority, error.message, { asJson: false });
    setResult(elements.chartInstitution, error.message, { asJson: false });
  }
}

async function submitInstitutionForm(event) {
  event.preventDefault();
  const caseId = document.getElementById("institution-case-id").value.trim();
  if (!caseId) {
    setResult(elements.institutionResult, "Case ID is required.", { asJson: false });
    return;
  }
  elements.institutionResult.textContent = "Uploading response evidence and verifying the case…";
  try {
    const evidence = await uploadEvidenceFiles(elements.institutionFiles.files);
    const payload = {
      actor_id: document.getElementById("institution-actor-id").value || null,
      note: document.getElementById("institution-note").value || null,
      institution_response: {
        response_text: document.getElementById("institution-response-text").value,
        location_hint: document.getElementById("institution-location").value || null,
        evidence,
      },
    };
    const result = await apiRequest(`/v1/cases/${encodeURIComponent(caseId)}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setResult(elements.institutionResult, {
      summary: summarizeCase(result.case),
      verification: result.case.verification,
      original_request: result.original_request,
    });
    state.selectedCaseId = result.case.case_id;
    await refreshReviewQueue();
  } catch (error) {
    setResult(elements.institutionResult, error.message, { asJson: false });
  }
}

function bindEvents() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => selectTab(button.dataset.tab));
  });
  elements.authForm.addEventListener("submit", submitAuthForm);
  elements.authLogout.addEventListener("click", logoutCurrentUser);
  elements.refreshAll.addEventListener("click", async () => {
    await Promise.allSettled([refreshHealth(), refreshReviewQueue(), refreshAnalytics()]);
    if (state.selectedCaseId) {
      await openCase(state.selectedCaseId);
    }
  });
  elements.citizenFiles.addEventListener("change", () =>
    renderSelectedFiles(elements.citizenFiles, elements.citizenFileList),
  );
  elements.institutionFiles.addEventListener("change", () =>
    renderSelectedFiles(elements.institutionFiles, elements.institutionFileList),
  );
  elements.citizenForm.addEventListener("submit", submitCitizenForm);
  elements.reviewFilters.addEventListener("submit", refreshReviewQueue);
  elements.reviewOpenCase.addEventListener("click", () => openCase(elements.reviewCaseId.value.trim()));
  document.querySelectorAll(".action-chip").forEach((button) => {
    button.addEventListener("click", () => performWorkflowAction(button.dataset.action));
  });
  elements.refreshAnalytics.addEventListener("click", refreshAnalytics);
  elements.institutionForm.addEventListener("submit", submitInstitutionForm);
}

async function boot() {
  bindEvents();
  elements.authUsername.value = "citizen.demo";
  renderSelectedFiles(elements.citizenFiles, elements.citizenFileList);
  renderSelectedFiles(elements.institutionFiles, elements.institutionFileList);
  await refreshHealth();
  if (state.authToken) {
    await Promise.allSettled([refreshCurrentUser(), refreshReviewQueue(), refreshAnalytics()]);
    return;
  }
  renderAuthState();
}

boot();
