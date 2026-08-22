'use client'
import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Sparkles,
  ArrowLeft,
  Copy,
  RotateCcw,
  Check,
  Plus,
  StopCircle,
  Network,
  Shield,
} from "lucide-react";
import { PlaceholdersAndVanishInput } from "@/components/ui/placeholders-and-vanish-input";
import { Markdown } from "@/components/ui/markdown";
import { useStore } from "@/store/useStore";
import { motion, AnimatePresence } from "motion/react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { NAVIGATION_ITEMS } from "@/constants/navigation";
import { ReasoningTrace } from "@/components/prod/chat/reasoning-trace";
import { GraphPanel } from "@/components/prod/graph/graph-panel";
import type { GraphPath, TraceStep } from "@/lib/graph";

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
  error?: boolean;
  trace?: TraceStep[];
  graphPath?: GraphPath | null;
}

// Images are intentionally hidden in this version — strip any markdown image
// syntax from assistant responses before they reach the renderer.
const IMAGE_MARKDOWN_REGEX = /!\[[^\]]*\]\s*\([^)]*\)/g;
const stripImages = (text: string) =>
  text.replace(IMAGE_MARKDOWN_REGEX, "").replace(/\n{3,}/g, "\n\n").trim();

const FOLLOW_UPS = [
  "Tell me more.",
  "Can you give an example?",
  "What's your favorite project?",
  "What tech stack do you use?",
  "How can we collaborate?",
  "What are you working on right now?",
];

const PLACEHOLDERS = [
  "Ask about my research in APT detection...",
  "How do I use AI for Cybersecurity?",
  "Tell me about my machine learning projects...",
  "What is the future of threat detection?",
  "Explain my latest publication...",
];

const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentInput, setCurrentInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [graphOpen, setGraphOpen] = useState(false);
  const [latestGraphPath, setLatestGraphPath] = useState<GraphPath | null>(null);

  const { initialMessage, setInitialMessage } = useStore();
  const hasProcessedInitialMessage = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  const isBusy = isStreaming || isThinking;

  // Auto-scroll while streaming and on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isThinking]);

  const submitQuestion = useCallback(
    async (question: string, opts?: { replaceLast?: boolean }) => {
      const trimmed = question.trim();
      if (!trimmed) return;

      // Abort any in-flight stream
      if (abortRef.current) abortRef.current.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const assistantId = newId();

      setMessages((prev) => {
        let next = prev;
        if (opts?.replaceLast) {
          // remove the trailing assistant message and re-use the existing question
          if (next.length && next[next.length - 1].role === "assistant") {
            next = next.slice(0, -1);
          }
        } else {
          next = [...next, { id: newId(), role: "user", content: trimmed }];
        }
        return [
          ...next,
          { id: assistantId, role: "assistant", content: "", trace: [] },
        ];
      });

      setIsThinking(true);
      setIsStreaming(false);

      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_CHAT_API_URL || "http://127.0.0.1:8000"}/chat/stream`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: trimmed }),
            signal: controller.signal,
          }
        );

        if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let fullResponse = "";
        let firstChunk = true;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n");
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const data = JSON.parse(line.slice(6));
              if (data.chunk) {
                fullResponse += data.chunk;
                if (firstChunk) {
                  firstChunk = false;
                  setIsThinking(false);
                  setIsStreaming(true);
                }
                const visible = stripImages(fullResponse);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId ? { ...m, content: visible } : m
                  )
                );
              }
              if (data.trace) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, trace: [...(m.trace ?? []), data.trace] }
                      : m
                  )
                );
              }
              if (data.graph_path) {
                setLatestGraphPath(data.graph_path);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, graphPath: data.graph_path }
                      : m
                  )
                );
              }
              if (data.error) throw new Error(data.error);
            } catch {
              // skip invalid lines
            }
          }
        }

        // Images are disabled in this version — only keep the text body.
        const cleanText = stripImages(fullResponse);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: cleanText } : m
          )
        );
      } catch (err: unknown) {
        if ((err as { name?: string })?.name === "AbortError") {
          // Mark partial response as stopped
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId && !m.content
                ? { ...m, content: "_(stopped)_", error: true }
                : m
            )
          );
          return;
        }
        console.error("Chat error:", err);
        const errorMessage =
          err instanceof Error ? `Error: ${err.message}` : "System error. Please retry.";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: errorMessage, error: true } : m
          )
        );
      } finally {
        setIsThinking(false);
        setIsStreaming(false);
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    []
  );

  // Initial message from homepage
  useEffect(() => {
    if (initialMessage && !hasProcessedInitialMessage.current) {
      hasProcessedInitialMessage.current = true;
      submitQuestion(initialMessage);
      setInitialMessage(null);
    }
  }, [initialMessage, setInitialMessage, submitQuestion]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentInput(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!currentInput.trim() || isBusy) return;
    const msg = currentInput.trim();
    setCurrentInput("");
    submitQuestion(msg);
  };

  const handleRetry = () => {
    // Find the last user message and resubmit, replacing the assistant response
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    submitQuestion(lastUser.content, { replaceLast: true });
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleNewChat = () => {
    abortRef.current?.abort();
    setMessages([]);
    setCurrentInput("");
  };

  const handleCopy = async (id: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedId(id);
      setTimeout(() => setCopiedId((c) => (c === id ? null : c)), 1500);
    } catch {
      // ignore
    }
  };

  // Rotating follow-ups, deterministic per turn count
  const followUps = useMemo(() => {
    const seed = messages.length;
    const start = seed % FOLLOW_UPS.length;
    return [0, 1, 2].map((i) => FOLLOW_UPS[(start + i) % FOLLOW_UPS.length]);
  }, [messages.length]);

  const lastMessage = messages[messages.length - 1];
  const showFollowUps =
    !isBusy &&
    lastMessage?.role === "assistant" &&
    !lastMessage.error &&
    lastMessage.content.length > 0;

  return (
    <div className="min-h-[100dvh] bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex flex-col">
      {/* Header */}
      <nav className="fixed top-0 left-0 right-0 z-50 px-4 sm:px-6 py-3 border-b border-zinc-100 dark:border-zinc-900 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-white transition-colors"
            aria-label="Back to home"
          >
            <ArrowLeft size={18} />
            <span className="font-medium hidden sm:inline">Back</span>
          </button>

          <div className="flex items-center gap-2">
            <div className="relative w-7 h-7 rounded-full overflow-hidden border border-zinc-200 dark:border-zinc-800">
              <Image src="/me.png" alt="Ahmed" width={28} height={28} />
            </div>
            <div className="flex flex-col leading-tight">
              <span className="text-sm font-medium">AI BARGADY</span>
              <span className="flex items-center gap-1 text-[10px] text-zinc-500 dark:text-zinc-500">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                Online
              </span>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={() => router.push("/security")}
              className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors px-2.5 py-1.5 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-900"
              aria-label="Security showcase"
              title="Security — red-team the twin"
            >
              <Shield size={14} />
              <span className="hidden sm:inline font-medium">Security</span>
            </button>
            <button
              onClick={() => setGraphOpen((o) => !o)}
              className={`flex items-center gap-1.5 text-xs transition-colors px-2.5 py-1.5 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-900 ${
                graphOpen
                  ? "text-blue-600 dark:text-blue-400"
                  : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
              }`}
              aria-label="Toggle knowledge graph"
              title="Knowledge graph"
            >
              <Network size={14} />
              <span className="hidden sm:inline font-medium">Graph</span>
            </button>
            <button
              onClick={handleNewChat}
              disabled={messages.length === 0 && !isBusy}
              className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-900 dark:hover:text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed px-2.5 py-1.5 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-900"
              aria-label="Start new chat"
            >
              <Plus size={14} />
              <span className="hidden sm:inline font-medium">New</span>
            </button>
          </div>
        </div>
      </nav>

      {/* Messages */}
      <main className="flex-1 pt-20 pb-40 px-4 sm:px-6">
        <div className="max-w-3xl mx-auto">
          <AnimatePresence mode="wait" initial={false}>
            {messages.length === 0 && !isBusy ? (
              <motion.div
                key="welcome"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4 }}
                className="text-center py-12 sm:py-20"
              >
                <div className="mb-6 flex justify-center">
                  <div className="relative">
                    <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-full overflow-hidden border-4 border-zinc-100 dark:border-zinc-800">
                      <Image
                        src="/me.png"
                        alt="Ahmed BARGADY"
                        width={96}
                        height={96}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-emerald-500 rounded-full border-2 border-white dark:border-zinc-950 flex items-center justify-center">
                      <Sparkles className="w-3 h-3 text-white" />
                    </div>
                  </div>
                </div>

                <h1 className="text-2xl sm:text-4xl font-semibold tracking-tight mb-3">
                  Chat with{" "}
                  <span className="text-blue-600 dark:text-blue-400">AI</span>{" "}
                  <span className="text-red-600 dark:text-red-400">BARGADY</span>
                </h1>

                <p className="text-zinc-500 dark:text-zinc-400 max-w-lg mx-auto mb-8 text-sm sm:text-base">
                  Ask me anything about my research, projects, or experience in AI
                  and Cybersecurity.
                </p>

                <p className="text-xs uppercase tracking-wider text-zinc-400 dark:text-zinc-600 mb-3">
                  Start with
                </p>
                <div className="flex flex-wrap justify-center gap-2 max-w-xl mx-auto">
                  {NAVIGATION_ITEMS.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => submitQuestion(item.prompt)}
                      className="group flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-100 dark:bg-zinc-900 hover:bg-zinc-200 dark:hover:bg-zinc-800 text-xs sm:text-sm font-medium text-zinc-700 dark:text-zinc-300 transition-colors"
                    >
                      <item.icon size={14} className="text-zinc-500" />
                      {item.label}
                    </button>
                  ))}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="conversation"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="space-y-5 pt-6"
              >
                {messages.map((m, idx) => {
                  const isUser = m.role === "user";
                  const isLastAssistant =
                    !isUser && idx === messages.length - 1;
                  const showCursor = isLastAssistant && isStreaming;
                  const showThinking =
                    isLastAssistant && isThinking && !m.content;

                  if (isUser) {
                    return (
                      <motion.div
                        key={m.id}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.25 }}
                        className="flex justify-end"
                      >
                        <div className="max-w-[85%] sm:max-w-[80%] bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm sm:text-[15px]">
                          {m.content}
                        </div>
                      </motion.div>
                    );
                  }

                  return (
                    <motion.div
                      key={m.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.25 }}
                      className="flex gap-3"
                    >
                      <div className="w-8 h-8 rounded-full overflow-hidden flex-shrink-0 border border-zinc-200 dark:border-zinc-800">
                        <Image
                          src="/me.png"
                          alt="AI"
                          width={32}
                          height={32}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="flex-grow min-w-0">
                        {m.trace && m.trace.length > 0 && (
                          <ReasoningTrace
                            steps={m.trace}
                            active={isLastAssistant && isBusy}
                            onOpenGraph={() => {
                              if (m.graphPath) setLatestGraphPath(m.graphPath);
                              setGraphOpen(true);
                            }}
                          />
                        )}
                        <div
                          className={`bg-white dark:bg-zinc-900 border ${
                            m.error
                              ? "border-red-200 dark:border-red-900/50"
                              : "border-zinc-200 dark:border-zinc-800"
                          } rounded-2xl rounded-tl-sm px-4 sm:px-5 py-3 shadow-sm`}
                        >
                          {showThinking ? (
                            <ThinkingDots />
                          ) : (
                            <div className="prose prose-sm dark:prose-invert prose-zinc max-w-none break-words">
                              <Markdown content={m.content} />
                              {showCursor && (
                                <span className="inline-block w-1.5 h-4 bg-blue-500 ml-0.5 align-middle animate-pulse" />
                              )}
                            </div>
                          )}
                        </div>

                        {/* Message actions */}
                        {!showThinking && m.content && !showCursor && (
                          <div className="flex items-center gap-1 mt-1.5 ml-1">
                            <button
                              onClick={() => handleCopy(m.id, m.content)}
                              className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 px-1.5 py-1 rounded transition-colors"
                              aria-label="Copy message"
                            >
                              {copiedId === m.id ? (
                                <>
                                  <Check size={11} /> Copied
                                </>
                              ) : (
                                <>
                                  <Copy size={11} /> Copy
                                </>
                              )}
                            </button>
                            {isLastAssistant && !isBusy && (
                              <button
                                onClick={handleRetry}
                                className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 px-1.5 py-1 rounded transition-colors"
                                aria-label="Regenerate response"
                              >
                                <RotateCcw size={11} /> Retry
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  );
                })}

                {/* Follow-up suggestions */}
                <AnimatePresence>
                  {showFollowUps && (
                    <motion.div
                      key="followups"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3, delay: 0.1 }}
                      className="flex flex-wrap gap-2 pl-11"
                    >
                      {followUps.map((f) => (
                        <button
                          key={f}
                          onClick={() => submitQuestion(f)}
                          className="text-xs px-3 py-1.5 rounded-full border border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-900 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
                        >
                          {f}
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>

                <div ref={messagesEndRef} className="h-1" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Knowledge-graph panel */}
      <GraphPanel
        open={graphOpen}
        onClose={() => setGraphOpen(false)}
        highlight={latestGraphPath}
      />

      {/* Input */}
      <div
        className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-white/80 dark:from-zinc-950 dark:via-zinc-950 dark:to-zinc-950/80 px-4 sm:px-6 pt-4 pb-4 z-40"
        style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom))" }}
      >
        <div className="max-w-3xl mx-auto">
          <div className="relative bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-full overflow-hidden focus-within:border-zinc-300 dark:focus-within:border-zinc-700 transition-colors">
            <PlaceholdersAndVanishInput
              placeholders={PLACEHOLDERS}
              onChange={handleChange}
              onSubmit={handleSubmit}
            />
            {isBusy && (
              <button
                onClick={handleStop}
                className="absolute right-14 top-1/2 -translate-y-1/2 z-50 text-zinc-500 hover:text-red-500 transition-colors"
                aria-label="Stop generating"
                title="Stop"
              >
                <StopCircle size={18} />
              </button>
            )}
          </div>
          <p className="text-center mt-2 text-[11px] text-zinc-400 dark:text-zinc-600">
            AI-powered responses · I might be wrong sometimes
          </p>
        </div>
      </div>
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 py-1" aria-label="Thinking">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-zinc-400 dark:bg-zinc-600"
          animate={{ opacity: [0.3, 1, 0.3], y: [0, -2, 0] }}
          transition={{
            duration: 1.1,
            repeat: Infinity,
            delay: i * 0.15,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
