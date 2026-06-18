"use client";

import { useState, useCallback } from "react";
import { sendMessage, type Message, type Source } from "@/lib/api";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const send = useCallback(
    async (query: string) => {
      if (!query.trim() || isLoading) return;

      setError(null);
      setIsLoading(true);

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
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);

        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: `❌ **Error:** ${msg}`,
          timestamp: new Date().toISOString(),
        });
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, addMessage]
  );

  const clear = useCallback(() => {
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