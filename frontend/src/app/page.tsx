"use client";

import Header from "@/components/Header";
import ChatPanel from "@/components/ChatPanel";
import DocumentsPanel from "@/components/DocumentsPanel";
import StatusBar from "@/components/StatusBar";
import { MDBContainer, MDBRow, MDBCol } from "mdb-react-ui-kit";

export default function Home() {
  return (
    <div className="d-flex flex-column vh-100">
      <Header />

      <MDBContainer fluid className="flex-grow-1 py-3">
        <MDBRow className="g-3 h-100">
          <MDBCol md="8" className="h-100">
            <ChatPanel />
          </MDBCol>
          <MDBCol md="4" className="h-100">
            <DocumentsPanel />
          </MDBCol>
        </MDBRow>
      </MDBContainer>

      <StatusBar />
    </div>
  );
}