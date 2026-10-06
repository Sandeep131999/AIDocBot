"use client";

import { useEffect, useState } from "react";
import { MDBBadge, MDBContainer, MDBFooter, MDBIcon } from "mdb-react-ui-kit";
import { getHealth, type SystemHealth } from "@/lib/api";

export default function StatusBar() {
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    const refresh = () => getHealth().then(setHealth).catch(() => setHealth(null));
    refresh();
    const timer = window.setInterval(refresh, 15000);
    return () => window.clearInterval(timer);
  }, []);

  const healthy = health?.status === "healthy";
  const chunkCount = health?.vector_db?.chunks;

  return (
    <MDBFooter className="dashboard-footer border-top">
      <MDBContainer fluid className="d-flex justify-content-between align-items-center flex-wrap gap-2 py-2">
        <span className="small text-muted"><MDBIcon fas icon="link" className="me-2" />API service</span>
        <div className="d-flex align-items-center gap-3">
          <MDBBadge color={healthy ? "success" : health?.status === "degraded" ? "warning" : "secondary"} pill>
            {health?.status ?? "unavailable"}
          </MDBBadge>
          {typeof chunkCount === "number" && (
            <span className="small text-muted"><MDBIcon fas icon="puzzle-piece" className="me-1" />{chunkCount.toLocaleString()} indexed chunks</span>
          )}
        </div>
      </MDBContainer>
    </MDBFooter>
  );
}
