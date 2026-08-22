// Types + API calls for the security showcase (red-team playground + stats).

import { CHAT_API_URL } from "./graph";

export interface Signature {
  category: string;
  match: string;
}

export interface ScreenVerdict {
  blocked: boolean;
  risk: "none" | "low" | "medium" | "high";
  score: number;
  signatures: Signature[];
  reason: string;
}

export interface RedteamCase {
  label: string;
  payload: string;
  expected_block: boolean;
  blocked: boolean;
  risk: string;
  correct: boolean;
  signatures: Signature[];
}

export interface RedteamReport {
  results: RedteamCase[];
  passed: number;
  total: number;
  accuracy: number;
}

export interface SecurityEvent {
  ts: number;
  kind: string;
  risk: string;
  detail: string;
  signatures: Signature[];
}

export interface SecurityStats {
  stats: Record<string, number>;
  recent: SecurityEvent[];
}

export const RISK_STYLES: Record<string, string> = {
  high: "text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 border-red-200 dark:border-red-900",
  medium:
    "text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-900",
  low: "text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-950/40 border-yellow-200 dark:border-yellow-900",
  none: "text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-900",
};

export const CATEGORY_LABELS: Record<string, string> = {
  instruction_override: "Instruction override",
  system_prompt_leak: "System-prompt extraction",
  role_hijack: "Role / persona hijack",
  exfiltration: "Data exfiltration",
  encoding_evasion: "Encoding evasion",
  delimiter_injection: "Delimiter injection",
  oversized_input: "Oversized input",
};

export async function screenText(text: string): Promise<ScreenVerdict> {
  const res = await fetch(`${CHAT_API_URL}/security/screen`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`Screen failed: ${res.status}`);
  return res.json();
}

export async function runRedteam(): Promise<RedteamReport> {
  const res = await fetch(`${CHAT_API_URL}/security/redteam`, { method: "POST" });
  if (!res.ok) throw new Error(`Red-team failed: ${res.status}`);
  return res.json();
}

export async function fetchSecurityStats(
  signal?: AbortSignal
): Promise<SecurityStats> {
  const res = await fetch(`${CHAT_API_URL}/security/stats`, { signal });
  if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
  return res.json();
}
