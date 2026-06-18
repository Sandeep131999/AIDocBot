"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { listDocuments, uploadDocument, deleteDocument, type Document } from "@/lib/api";

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastAction, setLastAction] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setError(null);
      const hasPending = docs.some((d) => d.status === "pending" || d.status === "processing");
      if (!hasPending && intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startPolling = useCallback(() => {
    if (intervalRef.current) return;
    intervalRef.current = setInterval(refresh, 3000);
  }, [refresh]);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setLastAction(`Uploading ${file.name}...`);
      try {
        await uploadDocument(file);
        await refresh();
        startPolling();
        setLastAction(`Uploaded ${file.name}`);
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
    async (id: number, filename?: string) => {
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
          setDocuments((prev) => [...prev, docToDelete].sort((a, b) => a.id - b.id));
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
    refresh();
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [refresh]);

  return { documents, isLoading, uploading, error, lastAction, upload, remove, refresh };
}