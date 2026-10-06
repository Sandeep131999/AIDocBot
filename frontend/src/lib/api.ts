export interface QueryRewrite {
  optimized: string;
  variations?: string[];
  key_entities?: string[];
  hyde_document?: string | null;
}

export interface Source {
  document: string;
  combined_score: number;
  vector_score: number;
  keyword_score: number;
  final_score: number;
  llm_score: number;
  explanation: string;
  llm_provider: string;
  employee_name?: string;
  query_rewrite?: QueryRewrite;
}

export interface ChatResponse {
  assistant_message: string;
  sources: Source[];
  used_model: string;
  content?: string;
}

export interface Document {
  id: string;
  filename: string;
  size: number;
  indexed: boolean;
  uploaded_at: string;
  status?: "indexed" | "processing" | "pending" | "error";
  chunk_count?: number;
  error?: string;
}

export interface DocumentsResponse {
  documents: Document[];
}

export interface SystemHealth {
  status: string;
  vector_db?: { status: string; chunks?: number; backend?: string; detail?: string };
  features?: Record<string, boolean>;
}

export interface CacheStats {
  [key: string]: string | number | boolean | null | undefined;
}

export interface ProjectConfig {
  [key: string]: string | number | boolean | string[] | null | undefined;
}

export interface EvaluationResponse {
  status: string;
  results: unknown;
}

export interface TokenResponse {
  access_token: string;
  token_type?: string;
  refresh_token?: string;
  [key: string]: unknown;
}

export interface Project {
  id: string;
  name: string;
}

let accessToken: string | null = null;
let projectId: string | null = null;

export function setApiCredentials(token: string | null, selectedProjectId: string | null): void {
  accessToken = token;
  projectId = selectedProjectId;
}

function requestHeaders(headers?: HeadersInit, projectScoped = true): Headers {
  const result = new Headers(headers);
  const token = accessToken ?? (typeof window !== "undefined" ? window.sessionStorage.getItem("aidocbot.accessToken") : null);
  const selectedProject = projectId ?? (typeof window !== "undefined" ? window.localStorage.getItem("aidocbot.projectId") : null);
  if (token) result.set("Authorization", `Bearer ${token}`);
  if (projectScoped && selectedProject) result.set("X-Project-ID", selectedProject);
  return result;
}

function apiFetch(url: string, init: RequestInit = {}, projectScoped = true): Promise<Response> {
  return fetch(url, { ...init, headers: requestHeaders(init.headers, projectScoped) });
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  provider?: string;
  latency?: number;
  used_model?: string;
  timestamp: string;
}

export async function sendMessage(query: string): Promise<ChatResponse> {
  const res = await apiFetch("http://127.0.0.1:8000/api/chat", {
    method: "POST",
    headers: {
      accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, top_k: 2 }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  const data = await res.json();
  return {
    assistant_message: data.answer ?? data.assistant_message ?? data.content ?? "No response",
    sources: (data.sources ?? []).map((source: Record<string, unknown>) => ({
      document: String(source.excerpt ?? source.document ?? ""),
      combined_score: Number(source.combined_score ?? 0),
      vector_score: Number(source.vector_score ?? 0),
      keyword_score: Number(source.keyword_score ?? 0),
      final_score: Number(source.final_score ?? 0),
      llm_score: Number(source.llm_score ?? 0),
      explanation: String(source.explanation ?? ""),
      llm_provider: String(data.provider_used ?? source.llm_provider ?? ""),
      employee_name: typeof source.filename === "string" ? source.filename : undefined,
    })),
    used_model: data.used_model ?? data.provider_used ?? "unknown",
  };
}

export async function listDocuments(): Promise<Document[]> {
  const res = await apiFetch("http://127.0.0.1:8000/api/documents", {
    method: "GET",
    headers: { accept: "application/json" },
  });

  if (!res.ok) {
    throw new Error(`Failed to load documents: ${res.status}`);
  }

  const data = await res.json();
  return (data.documents ?? []).map((doc: Record<string, unknown>) => {
    const filename = String(doc.filename ?? "Unknown document");
    return {
      id: filename,
      filename,
      size: Number(doc.size ?? 0),
      indexed: true,
      uploaded_at: String(doc.indexed_at ?? doc.uploaded_at ?? ""),
      status: "indexed" as const,
      chunk_count: Number(doc.chunks ?? doc.chunk_count ?? 0),
    };
  });
}

export async function uploadDocument(file: File): Promise<{ filename: string; indexed_chunks: number; message: string; job_id?: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await apiFetch("http://127.0.0.1:8000/api/documents/upload", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await apiFetch(`http://127.0.0.1:8000/api/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: { accept: "application/json" },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Delete failed: ${res.status}`);
  }
}

export async function getHealth(): Promise<SystemHealth> {
  const res = await apiFetch("http://127.0.0.1:8000/api/health", {
    method: "GET",
    headers: { accept: "application/json" },
  }, false);

  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }

  return res.json();
}

export async function getCacheStats(): Promise<CacheStats> {
  const res = await apiFetch("http://127.0.0.1:8000/api/cache/stats", { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`Cache stats unavailable: ${res.status}`);
  return res.json();
}

export async function invalidateCache(): Promise<{ deleted: number; pattern: string }> {
  const res = await apiFetch("http://127.0.0.1:8000/api/cache/invalidate", {
    method: "POST",
    headers: { accept: "application/json" },
  });
  if (!res.ok) throw new Error(`Cache invalidation failed: ${res.status}`);
  return res.json();
}

export async function getProjectConfig(): Promise<ProjectConfig> {
  const res = await apiFetch("http://127.0.0.1:8000/api/config", { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`Configuration unavailable: ${res.status}`);
  return res.json();
}

export async function evaluateQuestions(questions: string[]): Promise<EvaluationResponse> {
  const res = await apiFetch("http://127.0.0.1:8000/api/evaluate", {
    method: "POST",
    headers: { accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ questions, k: 5 }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Evaluation failed: ${res.status}`);
  }
  return res.json();
}

export async function issueAccessToken(userId: string, role: string): Promise<TokenResponse> {
  const res = await apiFetch("http://127.0.0.1:8000/auth/token", {
    method: "POST",
    headers: { accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, role }),
  }, false);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Token request failed: ${res.status}`);
  }
  return res.json();
}

export async function listProjects(): Promise<Project[]> {
  const res = await apiFetch("http://127.0.0.1:8000/api/projects", {
    headers: { accept: "application/json" },
  }, false);
  if (!res.ok) throw new Error(`Projects unavailable: ${res.status}`);
  const data = await res.json();
  const items = Array.isArray(data) ? data : data.projects ?? [];
  return items.flatMap((item: Record<string, unknown>) => {
    const id = item.project_id ?? item.id;
    if (id === undefined || id === null) return [];
    return [{ id: String(id), name: String(item.name ?? item.display_name ?? item.slug ?? id) }];
  });
}