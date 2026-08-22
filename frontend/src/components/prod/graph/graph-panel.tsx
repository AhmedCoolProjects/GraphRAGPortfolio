"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { X, Network, Loader2 } from "lucide-react";
import {
  FullGraph,
  GraphPath,
  fetchFullGraph,
  typeColor,
} from "@/lib/graph";
import { KnowledgeGraph } from "./knowledge-graph";

interface Props {
  open: boolean;
  onClose: () => void;
  highlight?: GraphPath | null;
}

export function GraphPanel({ open, onClose, highlight }: Props) {
  const [graph, setGraph] = useState<FullGraph | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Lazy-load the full graph the first time the panel opens.
  useEffect(() => {
    if (!open || graph) return;
    const ctrl = new AbortController();
    fetchFullGraph(ctrl.signal)
      .then(setGraph)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => ctrl.abort();
  }, [open, graph]);

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* backdrop (mobile) */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm lg:hidden"
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 260 }}
            className="fixed top-0 right-0 z-50 h-[100dvh] w-full sm:w-[440px] lg:w-[480px]
                       bg-white dark:bg-zinc-950 border-l border-zinc-200 dark:border-zinc-800
                       flex flex-col shadow-2xl"
          >
            {/* header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-100 dark:border-zinc-900">
              <div className="flex items-center gap-2">
                <Network size={18} className="text-blue-600 dark:text-blue-400" />
                <div className="flex flex-col leading-tight">
                  <span className="text-sm font-semibold">Knowledge graph</span>
                  <span className="text-[10px] text-zinc-500">
                    {highlight
                      ? `${highlight.nodes.length} nodes traversed for this answer`
                      : graph
                      ? `${graph.nodes.length} nodes · ${graph.edges.length} links`
                      : "loading…"}
                  </span>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-1.5 rounded-full text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
                aria-label="Close graph"
              >
                <X size={18} />
              </button>
            </div>

            {/* canvas */}
            <div className="relative flex-1 overflow-hidden">
              {error ? (
                <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-zinc-500">
                  Couldn&apos;t load the graph.
                  <br />
                  {error}
                </div>
              ) : !graph ? (
                <div className="absolute inset-0 flex items-center justify-center text-zinc-400">
                  <Loader2 className="animate-spin" size={22} />
                </div>
              ) : (
                <KnowledgeGraph graph={graph} highlight={highlight} />
              )}
            </div>

            {/* legend */}
            {graph && (
              <div className="px-4 py-3 border-t border-zinc-100 dark:border-zinc-900">
                <div className="flex flex-wrap gap-x-3 gap-y-1.5">
                  {graph.legend.map((l) => (
                    <span
                      key={l.id}
                      className="flex items-center gap-1.5 text-[10px] text-zinc-500"
                    >
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ background: typeColor(l.id) }}
                      />
                      {l.label}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
