"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { listProjects, setApiCredentials, type Project } from "@/lib/api";

interface AuthContextValue {
  accessToken: string | null;
  setAccessToken: (token: string | null) => void;
  projects: Project[];
  selectedProjectId: string | null;
  setSelectedProjectId: (projectId: string | null) => void;
  projectsLoading: boolean;
  projectError: string;
  refreshProjects: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);
const TOKEN_STORAGE_KEY = "aidocbot.accessToken";
const PROJECT_STORAGE_KEY = "aidocbot.projectId";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [accessToken, setAccessTokenState] = useState<string | null>(null);
  const [selectedProjectId, setSelectedProjectIdState] = useState<string | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [projectError, setProjectError] = useState("");
  const projectsRequestId = useRef(0);

  const setAccessToken = useCallback((token: string | null) => {
    setAccessTokenState(token);
    setApiCredentials(token, selectedProjectId);
    if (token) window.sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
    else window.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
  }, [selectedProjectId]);

  const setSelectedProjectId = useCallback((id: string | null) => {
    setSelectedProjectIdState(id);
    setApiCredentials(accessToken, id);
    if (id) window.localStorage.setItem(PROJECT_STORAGE_KEY, id);
    else window.localStorage.removeItem(PROJECT_STORAGE_KEY);
  }, [accessToken]);

  const refreshProjects = useCallback(async () => {
    const requestId = ++projectsRequestId.current;
    setProjectsLoading(true);
    try {
      const available = await listProjects();
      if (requestId !== projectsRequestId.current) return;
      setProjects(available);
      setProjectError("");
      const savedId = window.localStorage.getItem(PROJECT_STORAGE_KEY);
      const nextId = available.find((project) => project.id === savedId)?.id
        ?? available[0]?.id
        ?? null;
      setSelectedProjectIdState(nextId);
      setApiCredentials(window.sessionStorage.getItem(TOKEN_STORAGE_KEY), nextId);
      if (nextId) window.localStorage.setItem(PROJECT_STORAGE_KEY, nextId);
      else window.localStorage.removeItem(PROJECT_STORAGE_KEY);
    } catch (error) {
      if (requestId !== projectsRequestId.current) return;
      setProjects([]);
      setSelectedProjectIdState(null);
      setApiCredentials(window.sessionStorage.getItem(TOKEN_STORAGE_KEY), null);
      window.localStorage.removeItem(PROJECT_STORAGE_KEY);
      setProjectError(error instanceof Error ? error.message : "Could not load projects");
    } finally {
      if (requestId === projectsRequestId.current) setProjectsLoading(false);
    }
  }, []);

  useEffect(() => {
    const token = window.sessionStorage.getItem(TOKEN_STORAGE_KEY);
    const project = window.localStorage.getItem(PROJECT_STORAGE_KEY);
    setAccessTokenState(token);
    setSelectedProjectIdState(project);
    setApiCredentials(token, project);
  }, []);

  useEffect(() => {
    setApiCredentials(accessToken, selectedProjectId);
  }, [accessToken, selectedProjectId]);

  useEffect(() => {
    void refreshProjects();
  }, [accessToken, refreshProjects]);

  const value = useMemo(
    () => ({
      accessToken,
      setAccessToken,
      projects,
      selectedProjectId,
      setSelectedProjectId,
      projectsLoading,
      projectError,
      refreshProjects,
    }),
    [accessToken, setAccessToken, projects, selectedProjectId, setSelectedProjectId, projectsLoading, projectError, refreshProjects],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
}
