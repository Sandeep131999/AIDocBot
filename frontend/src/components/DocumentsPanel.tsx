"use client";

import { useState, useCallback, useRef } from "react";
import {
  MDBCard,
  MDBCardBody,
  MDBCardHeader,
  MDBBtn,
  MDBIcon,
  MDBBadge,
  MDBSpinner,
  MDBListGroup,
  MDBListGroupItem,
  MDBTooltip,
  MDBProgress,
} from "mdb-react-ui-kit";
import { useDocuments } from "@/hooks/useDocuments";
import type { Document } from "@/lib/api";

function statusColor(status: Document["status"]) {
  switch (status) {
    case "indexed": return "success";
    case "processing": return "warning";
    case "pending": return "info";
    case "error": return "danger";
    default: return "secondary";
  }
}

function statusIcon(status: Document["status"]) {
  switch (status) {
    case "indexed": return "check-circle";
    case "processing": return "spinner";
    case "pending": return "clock";
    case "error": return "exclamation-circle";
    default: return "question-circle";
  }
}

function formatBytes(bytes: number) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

export default function DocumentsPanel() {
  const { documents, isLoading, uploading, error, lastAction, upload, remove, refresh } = useDocuments();
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      files.forEach((f) => upload(f));
    },
    [upload]
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    files.forEach((f) => upload(f));
    e.target.value = "";
  };

  const docsArray = Array.isArray(documents) ? documents : [];

  return (
    <MDBCard className="h-100 shadow-3">
      <MDBCardHeader className="d-flex justify-content-between align-items-center bg-white">
        <div className="d-flex align-items-center gap-2">
          <MDBIcon fas icon="folder-open" className="text-primary" />
          <span className="fw-bold">Documents</span>
          <MDBBadge color="light" text="dark" pill>
            {docsArray.length}
          </MDBBadge>
        </div>
        <MDBBtn color="link" className="p-1" onClick={refresh} disabled={isLoading}>
          <MDBIcon fas icon="sync-alt" spin={isLoading} />
        </MDBBtn>
      </MDBCardHeader>

      <MDBCardBody className="d-flex flex-column gap-3">
        {/* Last action status */}
        {lastAction && (
          <div className="alert alert-info py-2 px-3 small mb-0 d-flex align-items-center">
            {lastAction.includes("Deleting") || lastAction.includes("Uploading") ? (
              <MDBSpinner color="info" size="sm" className="me-2" />
            ) : (
              <MDBIcon fas icon="check-circle" className="me-2 text-success" />
            )}
            {lastAction}
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="alert alert-danger py-2 px-3 small mb-0 d-flex align-items-center">
            <MDBIcon fas icon="exclamation-circle" className="me-2" />
            {error}
          </div>
        )}

        <div
          className={`border border-2 rounded-4 p-4 text-center cursor-pointer transition-all ${
            dragOver ? "border-primary bg-primary bg-opacity-10" : "border-dashed"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="d-none"
            onChange={handleFileSelect}
            accept=".pdf,.txt,.md,.doc,.docx"
          />
          <MDBIcon fas icon="cloud-upload-alt" size="2x" className="text-primary mb-2" />
          <p className="mb-0 text-muted small">
            {uploading ? (
              <MDBSpinner color="primary" size="sm" className="me-2" />
            ) : (
              <>Drop files here or <span className="text-primary">browse</span></>
            )}
          </p>
          <small className="text-muted">PDF, TXT, MD, DOC</small>
        </div>

        <div className="overflow-auto flex-grow-1" style={{ maxHeight: "60vh" }}>
          {docsArray.length === 0 && !isLoading && (
            <div className="text-center text-muted py-4">
              <MDBIcon fas icon="inbox" size="2x" className="mb-2" />
              <p className="small">No documents yet.</p>
            </div>
          )}

          <MDBListGroup>
            {docsArray.map((doc) => (
              <MDBListGroupItem
                key={doc.id}
                className="d-flex justify-content-between align-items-center px-3 py-2"
              >
                <div className="d-flex align-items-center gap-2 overflow-hidden">
                  <MDBIcon
                    fas
                    icon={statusIcon(doc.status)}
                    className={`text-${statusColor(doc.status)}`}
                    spin={doc.status === "processing"}
                  />
                  <div className="overflow-hidden">
                    <div className="text-truncate small fw-semibold" style={{ maxWidth: 180 }}>
                      {doc.filename}
                    </div>
                    <div className="d-flex gap-2 align-items-center">
                      <small className="text-muted">{formatBytes(doc.size)}</small>
                      <MDBBadge color={statusColor(doc.status)} pill className="small">
                        {doc.status}
                      </MDBBadge>
                      {doc.chunk_count && doc.chunk_count > 0 && (
                        <small className="text-muted">{doc.chunk_count} chunks</small>
                      )}
                    </div>
                    {doc.status === "processing" && (
                      <MDBProgress className="mt-1" height="4">
                        <div
                          className="progress-bar progress-bar-striped progress-bar-animated"
                          role="progressbar"
                          style={{ width: "70%" }}
                        />
                      </MDBProgress>
                    )}
                    {doc.error && (
                      <small className="text-danger">{doc.error}</small>
                    )}
                  </div>
                </div>

                <MDBTooltip tag="span" title="Delete">
                  <MDBBtn
                    color="danger"
                    outline
                    size="sm"
                    floating
                    onClick={() => remove(doc.id, doc.filename)}
                  >
                    <MDBIcon fas icon="trash-alt" />
                  </MDBBtn>
                </MDBTooltip>
              </MDBListGroupItem>
            ))}
          </MDBListGroup>
        </div>
      </MDBCardBody>
    </MDBCard>
  );
}