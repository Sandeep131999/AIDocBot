"use client";

import { useState, useEffect } from "react";
import {
  MDBFooter,
  MDBContainer,
  MDBBadge,
  MDBIcon,
  MDBTooltip,
  MDBCollapse,
  MDBBtn,
} from "mdb-react-ui-kit";
import { getHealth, type SystemHealth, type ProviderStatus } from "@/lib/api";

function ProviderPill({ p }: { p: ProviderStatus }) {
  const color = p.available ? "success" : p.cooldown_until ? "warning" : "danger";
  const icon = p.available ? "check" : p.cooldown_until ? "hourglass-half" : "times";
  return (
    <MDBTooltip
      tag="span"
      title={
        p.last_error
          ? `Error: ${p.last_error}`
          : p.cooldown_until
          ? `Cooldown until ${new Date(p.cooldown_until).toLocaleTimeString()}`
          : `Latency: ${p.latency_ms}ms`
      }
    >
      <MDBBadge color={color} pill className="d-flex align-items-center gap-1 px-2 py-1">
        <MDBIcon fas icon={icon} size="xs" />
        <span className="small">{p.name}</span>
        <span className="small opacity-75">{p.latency_ms}ms</span>
      </MDBBadge>
    </MDBTooltip>
  );
}

function formatUptime(seconds: number) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${h}h ${m}m ${s}s`;
}

export default function StatusBar() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    const id = setInterval(() => {
      getHealth().then(setHealth).catch(() => {});
    }, 5000);
    return () => clearInterval(id);
  }, []);

  if (!health) return null;

  // Guard: ensure providers is an array
  const providers = Array.isArray(health.providers) ? health.providers : [];

  return (
    <MDBFooter color="light" bgColor="light" className="border-top shadow-1">
      <MDBContainer fluid className="py-2">
        <div className="d-flex justify-content-between align-items-center flex-wrap gap-2">
          <div className="d-flex align-items-center gap-2 flex-wrap">
            <span className="text-muted small fw-bold">
              <MDBIcon fas icon="link" className="me-1" />
              Chain:
            </span>
            {providers.map((p, i) => (
              <span key={p.name} className="d-flex align-items-center gap-1">
                <ProviderPill p={p} />
                {i < providers.length - 1 && (
                  <MDBIcon fas icon="arrow-right" className="text-muted small" />
                )}
              </span>
            ))}
            {providers.length === 0 && (
              <span className="text-muted small">No providers configured</span>
            )}
          </div>

          <div className="d-flex align-items-center gap-3">
            <MDBTooltip tag="span" title="Indexed documents">
              <span className="d-flex align-items-center gap-1 text-muted small">
                <MDBIcon fas icon="file-alt" />
                {health.document_count ?? 0}
              </span>
            </MDBTooltip>
            <MDBTooltip tag="span" title="Total chunks">
              <span className="d-flex align-items-center gap-1 text-muted small">
                <MDBIcon fas icon="puzzle-piece" />
                {(health.indexed_chunks ?? 0).toLocaleString()}
              </span>
            </MDBTooltip>
            <MDBTooltip tag="span" title="Uptime">
              <span className="d-flex align-items-center gap-1 text-muted small">
                <MDBIcon fas icon="clock" />
                {formatUptime(health.uptime_seconds ?? 0)}
              </span>
            </MDBTooltip>
            <MDBBtn
              color="link"
              size="sm"
              className="p-0"
              onClick={() => setShowDetails(!showDetails)}
            >
              <MDBIcon fas icon={showDetails ? "chevron-up" : "chevron-down"} />
            </MDBBtn>
          </div>
        </div>

        <MDBCollapse open={showDetails}>
          <div className="mt-2 pt-2 border-top">
            <div className="row g-2">
              {providers.map((p) => (
                <div key={p.name} className="col-md-4">
                  <div className="p-2 rounded bg-white shadow-1">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className="fw-bold small">{p.name}</span>
                      <MDBBadge color={p.available ? "success" : "danger"} pill>
                        {p.available ? "UP" : "DOWN"}
                      </MDBBadge>
                    </div>
                    <div className="text-muted small">
                      <div>Latency: {p.latency_ms}ms</div>
                      {p.last_error && (
                        <div className="text-danger">{p.last_error}</div>
                      )}
                      {p.cooldown_until && (
                        <div className="text-warning">
                          Cooldown: {new Date(p.cooldown_until).toLocaleTimeString()}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </MDBCollapse>
      </MDBContainer>
    </MDBFooter>
  );
}