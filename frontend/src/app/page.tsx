"use client";

import Header from "@/components/Header";
import ProjectDashboard from "@/components/ProjectDashboard";
import StatusBar from "@/components/StatusBar";
import { AuthProvider } from "@/components/AuthProvider";

export default function Home() {
  return (
    <AuthProvider>
      <div className="d-flex flex-column min-vh-100">
        <Header />
        <ProjectDashboard />
        <StatusBar />
      </div>
    </AuthProvider>
  );
}