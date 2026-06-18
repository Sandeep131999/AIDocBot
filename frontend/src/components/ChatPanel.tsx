"use client";

import { useState, useRef, useEffect } from "react";
import {
  MDBCard,
  MDBCardBody,
  MDBCardHeader,
  MDBInputGroup,
  MDBBtn,
  MDBIcon,
  MDBSpinner,
  MDBBadge,
  MDBAccordion,
  MDBAccordionItem,
  MDBTooltip,
} from "mdb-react-ui-kit";
import { useChat } from "@/hooks/useChat";
import type { Message, Source } from "@/lib/api";

// Simple markdown-like renderer for Claude-style output
function ColorizedContent({ content }: { content: string }) {
  const lines = content.split("\n");
  
  return (
    <div style={{ lineHeight: 1.7 }}>
      {lines.map((line, i) => {
        // Headers
        if (line.startsWith("### ")) {
          return (
            <h5 key={i} className="fw-bold mt-3 mb-2 text-dark">
              {line.replace("### ", "")}
            </h5>
          );
        }
        if (line.startsWith("## ")) {
          return (
            <h4 key={i} className="fw-bold mt-3 mb-2 text-dark">
              {line.replace("## ", "")}
            </h4>
          );
        }
        if (line.startsWith("# ")) {
          return (
            <h3 key={i} className="fw-bold mt-3 mb-2 text-dark">
              {line.replace("# ", "")}
            </h3>
          );
        }
        
        // Bullet points
        if (line.startsWith("- ") || line.startsWith("* ")) {
          return (
            <div key={i} className="d-flex gap-2 mb-1">
              <span className="text-primary">•</span>
              <span>{renderInlineStyles(line.substring(2))}</span>
            </div>
          );
        }
        
        // Numbered lists
        if (/^\d+\.\s/.test(line)) {
          const num = line.match(/^\d+/)?.[0] || "1";
          return (
            <div key={i} className="d-flex gap-2 mb-1">
              <span className="text-primary fw-bold">{num}.</span>
              <span>{renderInlineStyles(line.replace(/^\d+\.\s/, ""))}</span>
            </div>
          );
        }
        
        // Code blocks
        if (line.startsWith("```")) {
          return null; // Handled by block rendering
        }
        
        // Empty lines
        if (!line.trim()) {
          return <div key={i} className="mb-2" />;
        }
        
        // Regular paragraph with inline styling
        return (
          <p key={i} className="mb-2">
            {renderInlineStyles(line)}
          </p>
        );
      })}
    </div>
  );
}

// Render bold, italic, inline code, and colored highlights
function renderInlineStyles(text: string) {
  // Split by patterns but keep delimiters
  const parts = text.split(/(\*\*.*?\*\*|\*.*?\*|`.*?`|\[.*?\]\(.*?\))/g);
  
  return parts.map((part, i) => {
    // Bold **text**
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="text-dark">{part.slice(2, -2)}</strong>;
    }
    // Italic *text*
    if (part.startsWith("*") && part.endsWith("*") && !part.startsWith("**")) {
      return <em key={i} className="text-secondary">{part.slice(1, -1)}</em>;
    }
    // Inline code `text`
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={i} className="px-1 py-0 rounded bg-light border text-danger" style={{ fontSize: "0.9em" }}>
          {part.slice(1, -1)}
        </code>
      );
    }
    // Links [text](url)
    const linkMatch = part.match(/\[(.*?)\]\((.*?)\)/);
    if (linkMatch) {
      return (
        <a key={i} href={linkMatch[2]} target="_blank" rel="noopener noreferrer" className="text-primary">
          {linkMatch[1]}
        </a>
      );
    }
    
    // Auto-highlight entities: dates, IDs, names with titles
    return highlightEntities(part, i);
  });
}

// Highlight specific entity patterns
function highlightEntities(text: string, key: number) {
  // Date patterns
  const datePattern = /(\d{1,2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2})/g;
  // Employee ID pattern
  const idPattern = /\b(\d{6,})\b/g;
  // Email pattern
  const emailPattern = /([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/g;
  // Name with designation pattern
  const namePattern = /(Name:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g;
  // Designation pattern
  const designationPattern = /(Designation:\s*)([A-Za-z\s]+)/g;
  
  let result = text;
  const elements: React.ReactNode[] = [];
  let lastIndex = 0;
  
  // Combined regex for all patterns
  const combinedPattern = /(\d{1,2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2}|\b\d{6,}\b|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|Name:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Designation:\s*[A-Za-z\s]+)/g;
  
  let match;
  const matches: Array<{ index: number; length: number; text: string; type: string }> = [];
  
  // Find all dates
  while ((match = datePattern.exec(text)) !== null) {
    matches.push({ index: match.index, length: match[0].length, text: match[0], type: "date" });
  }
  // Find all IDs
  while ((match = idPattern.exec(text)) !== null) {
    matches.push({ index: match.index, length: match[0].length, text: match[0], type: "id" });
  }
  // Find all emails
  while ((match = emailPattern.exec(text)) !== null) {
    matches.push({ index: match.index, length: match[0].length, text: match[0], type: "email" });
  }
  
  // Sort by index
  matches.sort((a, b) => a.index - b.index);
  
  // Remove overlapping matches
  const filteredMatches = matches.filter((m, i) => {
    if (i === 0) return true;
    const prev = matches[i - 1];
    return m.index >= prev.index + prev.length;
  });
  
  filteredMatches.forEach((m) => {
    if (m.index > lastIndex) {
      elements.push(<span key={`${key}-${lastIndex}`}>{text.slice(lastIndex, m.index)}</span>);
    }
    
    const colorClass = m.type === "date" ? "text-success fw-bold" : 
                       m.type === "id" ? "text-warning fw-bold" : 
                       m.type === "email" ? "text-info" : "";
    
    elements.push(
      <span key={`${key}-${m.index}`} className={colorClass} style={m.type === "id" ? { backgroundColor: "#fff3cd", padding: "0 4px", borderRadius: 4 } : {}}>
        {m.text}
      </span>
    );
    
    lastIndex = m.index + m.length;
  });
  
  if (lastIndex < text.length) {
    elements.push(<span key={`${key}-end`}>{text.slice(lastIndex)}</span>);
  }
  
  return elements.length > 0 ? elements : <span key={key}>{text}</span>;
}

function SourceCard({ src, index }: { src: Source; index: number }) {
  const score =
    typeof src.combined_score === "number"
      ? (src.combined_score * 100).toFixed(1)
      : "N/A";
  const finalScore =
    typeof src.final_score === "number"
      ? (src.final_score * 100).toFixed(1)
      : "N/A";

  const docPreview =
    src.document?.split("\n")[0]?.substring(0, 60) || "Document";

  return (
    <MDBAccordionItem
      collapseId={index}
      headerTitle={
        <span className="d-flex align-items-center gap-2 flex-wrap">
          <MDBBadge color="info" pill className="small">
            <MDBIcon fas icon="star" className="me-1" />
            {score}%
          </MDBBadge>
          <MDBBadge color="secondary" pill className="small">
            final: {finalScore}%
          </MDBBadge>
          <small
            className="text-muted text-truncate"
            style={{ maxWidth: 200 }}
          >
            {docPreview}
          </small>
        </span>
      }
    >
      <div className="small">
        {src.explanation && (
          <p className="mb-2">
            <strong>Why matched:</strong> {src.explanation}
          </p>
        )}

        {src.query_rewrite && (
          <div className="mb-2 p-2 rounded bg-light border">
            <div className="text-muted smaller mb-1">Query Rewrite:</div>
            <div className="fw-bold">{src.query_rewrite.optimized}</div>
            {src.query_rewrite.variations && (
              <div className="mt-1">
                {src.query_rewrite.variations.map((v: string, i: number) => (
                  <MDBBadge
                    key={i}
                    color="light"
                    text="dark"
                    pill
                    className="me-1 mb-1"
                  >
                    {v}
                  </MDBBadge>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="row g-2 mb-2">
          <div className="col-6">
            <div className="p-2 rounded bg-light">
              <div className="text-muted smaller">Vector</div>
              <div className="fw-bold text-primary">
                {typeof src.vector_score === "number"
                  ? (src.vector_score * 100).toFixed(1)
                  : "N/A"}
                %
              </div>
            </div>
          </div>
          <div className="col-6">
            <div className="p-2 rounded bg-light">
              <div className="text-muted smaller">Keyword</div>
              <div className="fw-bold text-success">
                {typeof src.keyword_score === "number"
                  ? (src.keyword_score * 100).toFixed(1)
                  : "N/A"}
                %
              </div>
            </div>
          </div>
          <div className="col-6">
            <div className="p-2 rounded bg-light">
              <div className="text-muted smaller">LLM</div>
              <div className="fw-bold text-warning">
                {typeof src.llm_score === "number"
                  ? (src.llm_score * 100).toFixed(1)
                  : "N/A"}
                %
              </div>
            </div>
          </div>
          <div className="col-6">
            <div className="p-2 rounded bg-light">
              <div className="text-muted smaller">Provider</div>
              <div className="fw-bold text-info">
                {src.llm_provider || "N/A"}
              </div>
            </div>
          </div>
        </div>

        <div className="border-top pt-2">
          <div className="text-muted smaller mb-1">Document Excerpt:</div>
          <pre
            className="small text-muted mb-0"
            style={{ whiteSpace: "pre-wrap", fontSize: 11 }}
          >
            {src.document || "No document text"}
          </pre>
        </div>
      </div>
    </MDBAccordionItem>
  );
}

function SourcesSection({ sources }: { sources: Source[] }) {
  if (!Array.isArray(sources) || sources.length === 0) return null;

  return (
    <div className="mt-3">
      <div className="d-flex align-items-center gap-2 mb-2">
        <MDBIcon fas icon="book-open" className="text-primary small" />
        <span className="small fw-bold text-muted">
          Sources ({sources.length})
        </span>
      </div>

      <MDBAccordion flush initialActive={-1}>
        {sources.map((src, i) => (
          <SourceCard key={`source-${i}`} src={src} index={i} />
        ))}
      </MDBAccordion>
    </div>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  
  return (
    <div
      className={`d-flex ${
        isUser ? "justify-content-end" : "justify-content-start"
      } mb-3`}
    >
      <div
        className={`p-3 rounded-4 ${
          isUser ? "bg-primary text-white" : "bg-white text-dark shadow-1"
        }`}
        style={{ maxWidth: "90%", wordBreak: "break-word" }}
      >
        {isUser ? (
          <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
            {msg.content ?? ""}
          </div>
        ) : (
          <ColorizedContent content={msg.content ?? ""} />
        )}

        {!isUser && (
          <div className="d-flex align-items-center gap-2 mt-3 pt-2 border-top flex-wrap">
            {msg.used_model && (
              <MDBBadge color="dark" pill className="small">
                <MDBIcon fas icon="microchip" className="me-1" />
                {msg.used_model}
              </MDBBadge>
            )}
            {msg.provider && (
              <MDBBadge color="secondary" pill className="small">
                <MDBIcon fas icon="robot" className="me-1" />
                {msg.provider}
              </MDBBadge>
            )}
            {typeof msg.latency === "number" && (
              <MDBBadge color="light" text="dark" pill className="small">
                <MDBIcon fas icon="clock" className="me-1" />
                {msg.latency}ms
              </MDBBadge>
            )}
          </div>
        )}

        {!isUser && <SourcesSection sources={msg.sources || []} />}
      </div>
    </div>
  );
}

function LoadingAnimation() {
  return (
    <div className="d-flex justify-content-start mb-3">
      <div
        className="bg-white p-3 rounded-4 shadow-1"
        style={{ minWidth: 280 }}
      >
        <div className="d-flex align-items-center gap-2 mb-3">
          <MDBSpinner color="primary" size="sm" />
          <span className="text-muted small">AI is thinking...</span>
        </div>

        <div className="d-flex flex-column gap-2">
          <div className="placeholder-glow">
            <span
              className="placeholder col-12"
              style={{ height: 12, borderRadius: 6 }}
            ></span>
          </div>
          <div className="placeholder-glow">
            <span
              className="placeholder col-10"
              style={{ height: 12, borderRadius: 6 }}
            ></span>
          </div>
          <div className="placeholder-glow">
            <span
              className="placeholder col-8"
              style={{ height: 12, borderRadius: 6 }}
            ></span>
          </div>
          <div className="d-flex gap-2 mt-2">
            <span
              className="placeholder col-2"
              style={{ height: 20, borderRadius: 10 }}
            ></span>
            <span
              className="placeholder col-2"
              style={{ height: 20, borderRadius: 10 }}
            ></span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ChatPanel() {
  const { messages, isLoading, error, send, clear } = useChat();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    send(input);
    setInput("");
  };

  return (
    <MDBCard className="h-100 shadow-3">
      <MDBCardHeader className="d-flex justify-content-between align-items-center bg-white">
        <div className="d-flex align-items-center gap-2">
          <MDBIcon fas icon="comments" className="text-primary" />
          <span className="fw-bold">Chat</span>
          {messages.length > 0 && (
            <MDBBadge color="light" text="dark" pill>
              {messages.length}
            </MDBBadge>
          )}
        </div>
        <MDBTooltip tag="span" title="Clear conversation">
          <MDBBtn
            color="danger"
            outline
            size="sm"
            onClick={clear}
            disabled={messages.length === 0}
          >
            <MDBIcon fas icon="trash-alt" />
          </MDBBtn>
        </MDBTooltip>
      </MDBCardHeader>

      <MDBCardBody
        className="overflow-auto"
        style={{ flex: 1, backgroundColor: "#f0f2f5" }}
      >
        {error && (
          <div
            className="alert alert-danger d-flex align-items-center mt-3"
            role="alert"
          >
            <MDBIcon fas icon="exclamation-circle" className="me-2" />
            <div>{error}</div>
          </div>
        )}

        {messages.length === 0 && (
          <div className="text-center text-muted mt-5">
            <MDBIcon
              fas
              icon="comment-dots"
              size="3x"
              className="mb-3 text-secondary"
            />
            <p>Ask a question about your documents.</p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}
        {isLoading && <LoadingAnimation />}
        <div ref={bottomRef} />
      </MDBCardBody>

      <div className="p-3 bg-white border-top">
        <form onSubmit={handleSubmit}>
          <MDBInputGroup>
            <input
              type="text"
              className="form-control"
              placeholder="Type your question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
            />
            <MDBBtn
              color="primary"
              type="submit"
              disabled={isLoading || !input.trim()}
            >
              {isLoading ? (
                <MDBSpinner color="light" size="sm" />
              ) : (
                <MDBIcon fas icon="paper-plane" />
              )}
            </MDBBtn>
          </MDBInputGroup>
        </form>
      </div>
    </MDBCard>
  );
}