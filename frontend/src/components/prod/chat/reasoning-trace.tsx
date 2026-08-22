"use client";

// Inline "thinking" trace — the agent's reasoning steps as animated pills.
// This is the visible signal that the answer came from an agentic GraphRAG
// pipeline (route -> search -> verify -> answer), not a plain prompt.

import { motion, AnimatePresence } from "motion/react";
import {
  Compass,
  Network,
  FileSearch,
  ShieldCheck,
  ShieldAlert,
  RefreshCw,
  PenLine,
  type LucideIcon,
} from "lucide-react";
import { TraceStep, TRACE_LABELS } from "@/lib/graph";

const ICONS: Record<string, LucideIcon> = {
  routing: Compass,
  graph_search: Network,
  vector_search: FileSearch,
  verifying: ShieldCheck,
  rewriting: RefreshCw,
  answering: PenLine,
  shield: ShieldAlert,
};

interface Props {
  steps: TraceStep[];
  active?: boolean; // still streaming → pulse the last step
  onOpenGraph?: () => void;
}

export function ReasoningTrace({ steps, active, onOpenGraph }: Props) {
  if (!steps.length) return null;

  return (
    <div className="mb-2 flex flex-wrap items-center gap-1.5">
      <AnimatePresence initial={false}>
        {steps.map((s, i) => {
          const Icon = ICONS[s.step] ?? Compass;
          const label = TRACE_LABELS[s.step] ?? s.step;
          const isLast = i === steps.length - 1;
          const isGraph = s.step === "graph_search";
          const isShield = s.step === "shield";
          return (
            <motion.button
              key={`${s.step}-${i}`}
              type="button"
              onClick={isGraph ? onOpenGraph : undefined}
              initial={{ opacity: 0, y: 4, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.2 }}
              className={`flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium
                ${
                  isShield
                    ? "border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300"
                    : active && isLast
                    ? "border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300"
                    : "border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-zinc-500"
                }
                ${isGraph ? "cursor-pointer hover:border-blue-400" : "cursor-default"}`}
            >
              <Icon
                size={11}
                className={active && isLast ? "animate-pulse" : ""}
              />
              <span>{label}</span>
              {s.detail && isGraph && (
                <span className="text-zinc-400">· {s.detail.split(",")[0]}</span>
              )}
            </motion.button>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
