"use client";

import { useCallback, useEffect, useState } from "react";
import {
  MDBBadge,
  MDBBtn,
  MDBCard,
  MDBCardBody,
  MDBCardHeader,
  MDBIcon,
  MDBSpinner,
} from "mdb-react-ui-kit";
import ChatPanel from "@/components/ChatPanel";
import DocumentsPanel from "@/components/DocumentsPanel";
import { useDocuments } from "@/hooks/useDocuments";
import { useAuth } from "@/components/AuthProvider";
import {
  evaluateQuestions,
  getCacheStats,
  getHealth,
  getProjectConfig,
  invalidateCache,
  issueAccessToken,
  type CacheStats,
  type ProjectConfig,
  type SystemHealth,
  type TokenResponse,
} from "@/lib/api";

type Section = "chat" | "documents" | "knowledge" | "users" | "evaluations" | "monitoring" | "production";

const sections: { id: Section; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "comments" },
  { id: "documents", label: "Documents", icon: "folder-open" },
  { id: "knowledge", label: "Knowledge", icon: "brain" },
  { id: "users", label: "Users & access", icon: "users" },
  { id: "evaluations", label: "Evaluations", icon: "flask" },
  { id: "monitoring", label: "Monitoring", icon: "chart-line" },
  { id: "production", label: "Production", icon: "sliders-h" },
];

function PanelTitle({ icon, title, subtitle }: { icon: string; title: string; subtitle: string }) {
  return (
    <div className="mb-4">
      <div className="d-flex align-items-center gap-2 mb-1">
        <MDBIcon fas icon={icon} className="text-primary" />
        <h2 className="h5 fw-bold mb-0">{title}</h2>
      </div>
      <p className="text-muted small mb-0">{subtitle}</p>
    </div>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <div>
      <div className="fw-semibold small mb-2">{title}</div>
      <pre className="dashboard-json">{JSON.stringify(value, null, 2)}</pre>
    </div>
  );
}

function KnowledgeSection() {
  const { documents, isLoading, error, refresh } = useDocuments();
  const indexedChunks = documents.reduce((sum, doc) => sum + (doc.chunk_count ?? 0), 0);
  const recentlyIndexed = [...documents].sort((a, b) => b.uploaded_at.localeCompare(a.uploaded_at)).slice(0, 5);

  return (
    <>
      <PanelTitle icon="brain" title="Knowledge training & indexing" subtitle="Uploaded documents are indexed automatically in the background." />
      <div className="row g-3 mb-4">
        <div className="col-sm-6">
          <div className="dashboard-stat">
            <span className="text-muted small">Indexed documents</span>
            <strong>{isLoading ? "…" : documents.length}</strong>
          </div>
        </div>
        <div className="col-sm-6">
          <div className="dashboard-stat">
            <span className="text-muted small">Indexed chunks</span>
            <strong>{isLoading ? "…" : indexedChunks.toLocaleString()}</strong>
          </div>
        </div>
      </div>
      <div className="alert alert-info small">
        <MDBIcon fas icon="info-circle" className="me-2" />
        There is no separate re-index endpoint in the current API. Add or remove files in Documents; upload starts background indexing automatically.
      </div>
      {error && <div className="alert alert-danger small">{error}</div>}
      <div className="d-flex align-items-center justify-content-between mb-2">
        <h3 className="h6 fw-bold mb-0">Latest indexed documents</h3>
        <MDBBtn color="link" size="sm" onClick={refresh} disabled={isLoading}>
          <MDBIcon fas icon="sync-alt" spin={isLoading} /> Refresh
        </MDBBtn>
      </div>
      {recentlyIndexed.length ? (
        <div className="list-group">
          {recentlyIndexed.map((doc) => (
            <div key={doc.id} className="list-group-item d-flex justify-content-between align-items-center">
              <span className="text-truncate me-3"><MDBIcon far icon="file-alt" className="text-muted me-2" />{doc.filename}</span>
              <MDBBadge color="success" pill>{doc.chunk_count ?? 0} chunks</MDBBadge>
            </div>
          ))}
        </div>
      ) : (
        <div className="dashboard-empty">No indexed documents yet. Upload documents to start building this project’s knowledge base.</div>
      )}
    </>
  );
}

function UsersSection() {
  const { accessToken, projects, selectedProjectId, setAccessToken, refreshProjects } = useAuth();
  const [userId, setUserId] = useState("");
  const [role, setRole] = useState("user");
  const [token, setToken] = useState<TokenResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const createToken = async (event: React.FormEvent) => {
    event.preventDefault();
    setError("");
    setToken(null);
    setLoading(true);
    try {
      const issued = await issueAccessToken(userId.trim(), role);
      setToken(issued);
      setAccessToken(issued.access_token);
      await refreshProjects();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not issue access token");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PanelTitle icon="users" title="Users & access" subtitle="Use this local-only tool to issue development tokens." />
      <div className="dashboard-stat mb-3">
        <span className="text-muted small">Current session</span>
        <strong className="fs-5">{accessToken ? "Authenticated" : "Unauthenticated development mode"}</strong>
        <span className="small text-muted">{projects.find((project) => project.id === selectedProjectId)?.name ?? "No project selected"}</span>
      </div>
      <div className="alert alert-warning small">
        <MDBIcon fas icon="exclamation-triangle" className="me-2" />
        Token issuance is described by the backend as development/testing only. Do not use this as production identity management.
      </div>
      <form onSubmit={createToken} className="row g-3 align-items-end mb-4">
        <div className="col-md-6">
          <label className="form-label small fw-semibold" htmlFor="token-user">User ID</label>
          <input id="token-user" className="form-control" value={userId} onChange={(event) => setUserId(event.target.value)} required placeholder="e.g. analyst-01" />
        </div>
        <div className="col-md-3">
          <label className="form-label small fw-semibold" htmlFor="token-role">Role</label>
          <select id="token-role" className="form-select" value={role} onChange={(event) => setRole(event.target.value)}>
            <option value="user">User</option>
            <option value="readonly">Read only</option>
            <option value="admin">Admin</option>
          </select>
        </div>
        <div className="col-md-3">
          <MDBBtn type="submit" color="primary" className="w-100" disabled={loading || !userId.trim()}>
            {loading ? <MDBSpinner size="sm" /> : "Issue dev token"}
          </MDBBtn>
        </div>
      </form>
      {error && <div className="alert alert-danger small">{error}</div>}
      {token && <div className="alert alert-success small">Development token issued and activated for this browser session. Keep the credential private.</div>}
      <div className="dashboard-empty mt-4">
        <strong>Not wired in this MVP:</strong> project member roster and membership changes have backend endpoints, but are not yet exposed in this dashboard.
      </div>
    </>
  );
}

function EvaluationSection() {
  const [questions, setQuestions] = useState("");
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const runEvaluation = async (event: React.FormEvent) => {
    event.preventDefault();
    const items = questions.split("\n").map((question) => question.trim()).filter(Boolean);
    if (!items.length) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await evaluateQuestions(items));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PanelTitle icon="flask" title="Evaluations" subtitle="Run the backend RAG evaluation against a question set (up to 50 questions)." />
      <form onSubmit={runEvaluation}>
        <label className="form-label small fw-semibold" htmlFor="evaluation-questions">Questions — one per line</label>
        <textarea id="evaluation-questions" className="form-control mb-3" rows={7} value={questions} onChange={(event) => setQuestions(event.target.value)} placeholder={"How do I request time off?\nWhat is the expense approval process?"} />
        <MDBBtn type="submit" color="primary" disabled={loading || !questions.trim()}>
          {loading ? <><MDBSpinner size="sm" className="me-2" />Running evaluation…</> : "Run evaluation"}
        </MDBBtn>
      </form>
      {error && <div className="alert alert-danger small mt-3">{error}</div>}
      {result !== null && <div className="mt-4"><JsonPanel title="Evaluation results" value={result} /></div>}
    </>
  );
}

function MonitoringSection() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [cache, setCache] = useState<CacheStats | null>(null);
  const [error, setError] = useState("");
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    const [healthResult, cacheResult] = await Promise.allSettled([getHealth(), getCacheStats()]);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (cacheResult.status === "fulfilled") setCache(cacheResult.value);
    const errors = [healthResult, cacheResult].filter((result) => result.status === "rejected");
    setError(errors.length ? "Some monitoring endpoints could not be reached." : "");
    setCheckedAt(new Date());
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const featureEntries = Object.entries(health?.features ?? {});

  return (
    <>
      <PanelTitle icon="chart-line" title="Monitoring" subtitle="Live service health and retrieval-cache counters from the available API." />
      {error && <div className="alert alert-warning small">{error}</div>}
      <div className="d-flex justify-content-between align-items-center mb-3">
        <span className="small text-muted">Auto-refreshes every 15 seconds{checkedAt ? ` · Updated ${checkedAt.toLocaleTimeString()}` : ""}</span>
        <MDBBtn color="link" size="sm" onClick={() => void refresh()}><MDBIcon fas icon="sync-alt" /> Refresh</MDBBtn>
      </div>
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="dashboard-stat">
            <span className="text-muted small">API health</span>
            <strong className="text-capitalize">{health?.status ?? "Unknown"}</strong>
            <span className="small text-muted">{health?.vector_db?.status ? `Vector database: ${health.vector_db.status}` : "Waiting for health check"}</span>
          </div>
        </div>
        <div className="col-md-6">
          <div className="dashboard-stat">
            <span className="text-muted small">Indexed chunks</span>
            <strong>{(health?.vector_db?.chunks ?? 0).toLocaleString()}</strong>
            <span className="small text-muted">{health?.vector_db?.backend ?? "Vector store"}</span>
          </div>
        </div>
      </div>
      <div className="mb-4">
        <h3 className="h6 fw-bold">Feature flags</h3>
        {featureEntries.length ? (
          <div className="d-flex flex-wrap gap-2">
            {featureEntries.map(([name, enabled]) => <MDBBadge key={name} color={enabled ? "success" : "light"} className={enabled ? "" : "text-dark"}>{name.replaceAll("_", " ")} · {enabled ? "on" : "off"}</MDBBadge>)}
          </div>
        ) : <div className="dashboard-empty">Feature status is not available.</div>}
      </div>
      {cache && <JsonPanel title="Cache statistics" value={cache} />}
    </>
  );
}

function ProductionSection() {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [configError, setConfigError] = useState("");
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState("");
  const [actionError, setActionError] = useState("");

  const loadConfig = useCallback(async () => {
    setLoading(true);
    try {
      setConfig(await getProjectConfig());
      setConfigError("");
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : "Configuration unavailable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadConfig(); }, [loadConfig]);

  const clearCache = async () => {
    setAction("");
    setActionError("");
    try {
      const result = await invalidateCache();
      setAction(`Cache invalidated: ${result.deleted} entries removed (${result.pattern}).`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Cache invalidation failed");
    }
  };

  return (
    <>
      <PanelTitle icon="sliders-h" title="Production management" subtitle="Read-only runtime configuration and the cache maintenance action exposed by the backend." />
      <div className="alert alert-info small">Settings are read-only here; configuration editing, deployment controls, and rollback APIs are not available.</div>
      <div className="d-flex justify-content-between align-items-center mb-2">
        <h3 className="h6 fw-bold mb-0">Runtime configuration</h3>
        <MDBBtn color="link" size="sm" onClick={() => void loadConfig()} disabled={loading}><MDBIcon fas icon="sync-alt" spin={loading} /> Refresh</MDBBtn>
      </div>
      {loading ? <MDBSpinner color="primary" /> : config ? <JsonPanel title="Current non-sensitive settings" value={config} /> : <div className="alert alert-warning small">{configError}</div>}
      <div className="border-top mt-4 pt-4">
        <h3 className="h6 fw-bold">Cache maintenance</h3>
        <p className="small text-muted">Clear cached responses when project knowledge or runtime behavior has changed.</p>
        <MDBBtn color="warning" onClick={clearCache}><MDBIcon fas icon="broom" className="me-2" />Invalidate cache</MDBBtn>
        {action && <div className="alert alert-success small mt-3 mb-0">{action}</div>}
        {actionError && <div className="alert alert-danger small mt-3 mb-0">{actionError}</div>}
      </div>
    </>
  );
}

export default function ProjectDashboard() {
  const [active, setActive] = useState<Section>("chat");
  const { projects, selectedProjectId, accessToken, projectsLoading, projectError } = useAuth();
  const selectedProject = projects.find((project) => project.id === selectedProjectId);

  return (
    <div className="dashboard-shell">
      <div className="dashboard-heading d-flex justify-content-between align-items-end gap-3 flex-wrap">
        <div>
          <div className="text-uppercase small text-primary fw-bold mb-1">Workspace</div>
          <h1 className="h3 fw-bold mb-1">Project dashboard</h1>
          <p className="text-muted mb-0">A focused view of your knowledge project and AI operations.</p>
        </div>
        <div className="dashboard-project-label">
          <MDBIcon fas icon="layer-group" className="text-primary me-2" />
          {selectedProject?.name ?? (projectsLoading ? "Loading projects…" : "Development workspace")}
          {!accessToken && <span className="badge text-bg-light ms-2">Unauthenticated dev mode</span>}
          {projectError && <span className="badge text-bg-warning ms-2">Projects unavailable</span>}
        </div>
      </div>

      <div className="dashboard-nav" role="tablist" aria-label="Project dashboard sections">
        {sections.map((section) => (
          <button
            key={section.id}
            type="button"
            role="tab"
            aria-selected={active === section.id}
            className={`dashboard-nav-item${active === section.id ? " active" : ""}`}
            onClick={() => setActive(section.id)}
          >
            <MDBIcon fas icon={section.icon} />
            <span>{section.label}</span>
          </button>
        ))}
      </div>

      <main className={`dashboard-content${active === "chat" || active === "documents" ? " dashboard-content-fill" : ""}`} role="tabpanel">
        {active === "chat" && <ChatPanel />}
        {active === "documents" && <DocumentsPanel />}
        {active === "knowledge" && <KnowledgeSection />}
        {active === "users" && <UsersSection />}
        {active === "evaluations" && <EvaluationSection />}
        {active === "monitoring" && <MonitoringSection />}
        {active === "production" && <ProductionSection />}
      </main>
    </div>
  );
}
