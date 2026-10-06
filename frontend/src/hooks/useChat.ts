"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { sendMessage, type Message, type Source } from "@/lib/api";
import { useAuth } from "@/components/AuthProvider";

export function useChat() {
  const { selectedProjectId } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestVersion = useRef(0);

  useEffect(() => {
    requestVersion.current += 1;
    setMessages([]);
    setError(null);
    setIsLoading(false);
  }, [selectedProjectId]);

  const addMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const send = useCallback(
    async (query: string) => {
      if (!query.trim() || isLoading) return;

      setError(null);
      setIsLoading(true);
      const currentVersion = requestVersion.current;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: query,
        timestamp: new Date().toISOString(),
      };

      addMessage(userMsg);

      const startTime = performance.now();

      try {
        const data = await sendMessage(query);
        if (currentVersion !== requestVersion.current) return;
        const latency = Math.round(performance.now() - startTime);

        const provider =
          data.sources?.find((s: Source) => s.llm_provider)?.llm_provider ??
          "unknown";

        const assistantMsg: Message = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.assistant_message ?? data.content ?? "No response",
          sources: data.sources,
          provider,
          latency,
          used_model: data.used_model ?? "unknown",
          timestamp: new Date().toISOString(),
        };

        addMessage(assistantMsg);
      } catch (err) {
        if (currentVersion !== requestVersion.current) return;
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);

        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: `❌ **Error:** ${msg}`,
          timestamp: new Date().toISOString(),
        });
      } finally {
        if (currentVersion === requestVersion.current) setIsLoading(false);
      }
    },
    [isLoading, addMessage]
  );

  const clear = useCallback(() => {
    requestVersion.current += 1;
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    send,
    clear,
  };
}