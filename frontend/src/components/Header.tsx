"use client";

import {
  MDBNavbar,
  MDBContainer,
  MDBNavbarBrand,
  MDBBadge,
  MDBBtn,
  MDBIcon,
} from "mdb-react-ui-kit";
import { useState, useEffect } from "react";
import { getHealth, type SystemHealth } from "@/lib/api";

export default function Header() {
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    const id = setInterval(() => {
      getHealth().then(setHealth).catch(() => {});
    }, 10000);
    return () => clearInterval(id);
  }, []);

  const statusColor =
    health?.status === "healthy"
      ? "success"
      : health?.status === "degraded"
      ? "warning"
      : "danger";

  const statusLabel = health?.status ?? "connecting...";

  return (
    <MDBNavbar expand="lg" light bgColor="light" className="shadow-1">
      <MDBContainer fluid>
        <MDBNavbarBrand href="#" className="d-flex align-items-center gap-2">
          <MDBIcon fas icon="brain" className="text-primary fs-4" />
          <span className="fw-bold">RAG Console</span>
        </MDBNavbarBrand>

        <div className="d-flex align-items-center gap-3">
          <MDBBadge
            color={statusColor}
            className="px-3 py-2 text-uppercase"
            pill
          >
            <MDBIcon fas icon="circle" className="me-1" style={{ fontSize: 8 }} />
            {statusLabel}
          </MDBBadge>

          <MDBBtn
            color="link"
            className="p-1"
            onClick={() => window.location.reload()}
            title="Refresh"
          >
            <MDBIcon fas icon="sync-alt" />
          </MDBBtn>
        </div>
      </MDBContainer>
    </MDBNavbar>
  );
}