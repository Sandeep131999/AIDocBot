"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { listDocuments, uploadDocument, deleteDocument, type Document } from "@/lib/api";
import { useAuth } from "@/components/AuthProvider";

export function useDocuments() {
  const { selectedProjectId } = useAuth();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastAction, setLastAction] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const latestProjectId = useRef(selectedProjectId);
  latestProjectId.current = selectedProjectId;

  const refresh = useCallback(async () => {
    const requestProjectId = selectedProjectId;
    setIsLoading(true);
    try {
      const docs = await listDocuments();
      if (latestProjectId.current !== requestProjectId) return;
      setDocuments(docs);
      setError(null);
      const hasPending = docs.some((d) => d.status === "pending" || d.status === "processing");
      if (!hasPending && intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    } catch (err) {
      if (latestProjectId.current !== requestProjectId) return;
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      if (latestProjectId.current === requestProjectId) setIsLoading(false);
    }
  }, [selectedProjectId]);

  const startPolling = useCallback(() => {
    if (intervalRef.current) return;
    intervalRef.current = setInterval(refresh, 3000);
  }, [refresh]);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setLastAction(`Uploading ${file.name}...`);
      try {
        const result = await uploadDocument(file);
        await refresh();
        startPolling();
        setLastAction(result.message || `Accepted ${file.name}; indexing runs in the background`);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
        setLastAction(`Failed to upload ${file.name}`);
      } finally {
        setUploading(false);
        setTimeout(() => setLastAction(null), 3000);
      }
    },
    [refresh, startPolling]
  );

  const remove = useCallback(
    async (id: string, filename?: string) => {
      const docToDelete = documents.find((d) => d.id === id);
      const name = filename || docToDelete?.filename || `ID ${id}`;

      // Optimistic delete: remove from UI immediately
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      setLastAction(`Deleting ${name}...`);

      try {
        await deleteDocument(id);
        // Refresh to confirm deletion and get updated list
        await refresh();
        setLastAction(`Deleted ${name}`);
      } catch (err) {
        // Restore document if delete failed
        if (docToDelete) {
          setDocuments((prev) => [...prev, docToDelete].sort((a, b) => a.filename.localeCompare(b.filename)));
        }
        setError(err instanceof Error ? err.message : "Delete failed");
        setLastAction(`Failed to delete ${name}`);
      } finally {
        setTimeout(() => setLastAction(null), 3000);
      }
    },
    [documents, refresh]
  );

  useEffect(() => {
    setDocuments([]);
    refresh();
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [refresh]);

  return { documents, isLoading, uploading, error, lastAction, upload, remove, refresh };
}