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
  id: number;
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

export interface HealthResponse {
  status: string;
}

export interface SystemHealth {
  status: string;
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
  const res = await fetch("http://127.0.0.1:8000/api/chat", {
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

  return res.json();
}

export async function listDocuments(): Promise<Document[]> {
  const res = await fetch("http://127.0.0.1:8000/api/documents", {
    method: "GET",
    headers: { accept: "application/json" },
  });

  if (!res.ok) {
    throw new Error(`Failed to load documents: ${res.status}`);
  }

  const data: DocumentsResponse = await res.json();
  return data.documents.map((doc) => ({
    ...doc,
    status: doc.indexed ? "indexed" : "pending",
    chunk_count: doc.indexed ? undefined : 0,
  }));
}

export async function uploadDocument(file: File): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/api/documents/upload", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
}

export async function deleteDocument(id: number): Promise<void> {
  // Ensure id is a proper integer
  const docId = Math.floor(Number(id));
  if (!Number.isFinite(docId) || docId <= 0) {
    throw new Error(`Invalid document ID: ${id}`);
  }

  const url = `http://127.0.0.1:8000/api/documents/${docId}`;
  console.log(`[API] DELETE ${url}`);

  const res = await fetch(url, {
    method: "DELETE",
    headers: { accept: "application/json" },
  });

  console.log(`[API] DELETE response status: ${res.status}`);

  if (!res.ok) {
    const errText = await res.text().catch(() => "Unknown error");
    console.error(`[API] DELETE error body:`, errText);
    throw new Error(`Delete failed: ${res.status} - ${errText}`);
  }

  // Some APIs return 200 with a body, some return 204 No Content
  const contentType = res.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    const body = await res.json();
    console.log(`[API] DELETE response body:`, body);
  } else {
    console.log(`[API] DELETE success (no JSON body)`);
  }
}

export async function getHealth(): Promise<SystemHealth> {
  const res = await fetch("http://127.0.0.1:8000/api/health", {
    method: "GET",
    headers: { accept: "application/json" },
  });

  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }

  return res.json();
}