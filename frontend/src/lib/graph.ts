// Shared types + helpers for the knowledge-graph visualization.

export const CHAT_API_URL =
  process.env.NEXT_PUBLIC_CHAT_API_URL || "http://127.0.0.1:8000";

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  description?: string;
  image?: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface FullGraph {
  meta?: Record<string, unknown>;
  legend: { id: string; label: string }[];
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// The subgraph the agent traversed for one answer (streamed as `graph_path`).
export interface GraphPath {
  nodes: { id: string; label: string; type: string; seed?: boolean }[];
  edges: { source: string; target: string; type: string }[];
}

// One reasoning step streamed as `trace`.
export interface TraceStep {
  step: string;
  detail?: string;
}

// Colour per node type. Tuned to read on both light and dark backgrounds.
export const TYPE_COLORS: Record<string, string> = {
  person: "#ef4444", // red — the root
  org: "#3b82f6", // blue — institutions
  company: "#0ea5e9", // sky — companies
  role: "#6366f1", // indigo — roles
  project: "#22c55e", // green — projects
  skill: "#f59e0b", // amber — skills/tech
  area: "#a855f7", // purple — research areas
  topic: "#ec4899", // pink — domains
  book: "#14b8a6", // teal — books
  language: "#eab308", // yellow — languages
  hobby: "#94a3b8", // slate — hobbies
};

export const typeColor = (type: string) => TYPE_COLORS[type] ?? "#94a3b8";

export async function fetchFullGraph(signal?: AbortSignal): Promise<FullGraph> {
  const res = await fetch(`${CHAT_API_URL}/graph`, { signal });
  if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`);
  return res.json();
}

// Human labels for the agent's trace steps.
export const TRACE_LABELS: Record<string, string> = {
  routing: "Routing",
  graph_search: "Searching the graph",
  vector_search: "Searching notes",
  verifying: "Verifying",
  rewriting: "Refining query",
  answering: "Answering",
  shield: "Blocked by guardrails",
};
