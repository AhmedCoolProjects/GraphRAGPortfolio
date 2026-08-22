"""
Guardrails for the digital twin: prompt-injection detection, PII redaction,
and a system-prompt-leak guard.

Everything here is deterministic and dependency-free (stdlib `re` only), so it
runs with zero API keys and is fully unit-testable. The detector is intentionally
transparent — it reports *which* signatures fired and *why* — because the point
is to demonstrate defensive reasoning, not to hide it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

MAX_INPUT_CHARS = 2000

# ---------------------------------------------------------------------------
# Prompt-injection signatures
# ---------------------------------------------------------------------------
# Each signature: (category, weight, compiled regex). Weights accumulate into a
# risk score; the category explains the attacker's intent to the UI.
_SIGNATURES: List[Tuple[str, int, "re.Pattern[str]"]] = [
    # Instruction override / jailbreak framing
    ("instruction_override", 3, re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b"
        r"(previous|prior|above|earlier|all|your|the)\b.{0,20}"
        r"(instruction|prompt|rule|guideline|context|direction)", re.I)),
    ("instruction_override", 3, re.compile(
        r"\bnew\b.{0,15}\b(instruction|rule|system prompt|directive)s?\b", re.I)),
    # System-prompt / secret extraction
    ("system_prompt_leak", 3, re.compile(
        r"\b(reveal|show|print|repeat|output|display|give me|what (is|are))\b"
        r".{0,40}\b(system prompt|initial prompt|instructions|your prompt|"
        r"the prompt above|persona|template|guidelines|rules)\b", re.I)),
    ("system_prompt_leak", 4, re.compile(
        r"\b(repeat|print|echo)\b.{0,20}\b(everything|all)\b.{0,20}\babove\b", re.I)),
    # Role-play / persona hijack
    ("role_hijack", 3, re.compile(
        r"\byou are (now|no longer|actually)\b", re.I)),
    ("role_hijack", 4, re.compile(
        r"\b(DAN|do anything now|developer mode|jailbreak|unfiltered|"
        r"without (any )?(restrictions|filters|rules|limitations))\b", re.I)),
    ("role_hijack", 3, re.compile(
        r"\b(pretend|act as|roleplay|simulate)\b.{0,30}\b"
        r"(no rules|no restrictions|evil|unrestricted|admin|root|developer)\b", re.I)),
    # Data / credential exfiltration
    ("exfiltration", 3, re.compile(
        r"\b(send|exfiltrate|leak|post|upload|email)\b.{0,30}\b"
        r"(api key|api_key|secret|token|password|credential|env|environment "
        r"variable|.env)\b", re.I)),
    ("exfiltration", 2, re.compile(
        r"\b(api[_ ]?key|secret key|access token|password|credentials?)\b", re.I)),
    # Encoding / delimiter evasion
    ("encoding_evasion", 2, re.compile(
        r"\b(base64|rot13|hex decode|decode this|in reverse)\b", re.I)),
    ("delimiter_injection", 2, re.compile(
        r"(```+\s*system|<\s*/?\s*\|?(system|im_start|endoftext)\|?\s*>|"
        r"\[/?INST\])", re.I)),
    ("delimiter_injection", 2, re.compile(
        r"^\s*(system|assistant)\s*:", re.I | re.M)),
]

RISK_THRESHOLDS = {"high": 4, "medium": 2}  # score >= value -> that level


@dataclass
class InputVerdict:
    blocked: bool
    risk: str                      # "none" | "low" | "medium" | "high"
    score: int
    signatures: List[dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "risk": self.risk,
            "score": self.score,
            "signatures": self.signatures,
            "reason": self.reason,
        }


def detect_injection(text: str) -> InputVerdict:
    """Score a user input for prompt-injection / jailbreak intent."""
    if len(text) > MAX_INPUT_CHARS:
        return InputVerdict(
            blocked=True, risk="medium", score=RISK_THRESHOLDS["medium"],
            reason=f"Input exceeds {MAX_INPUT_CHARS} characters.",
            signatures=[{"category": "oversized_input", "match": f"{len(text)} chars"}],
        )

    score = 0
    sigs: List[dict] = []
    seen = set()
    for category, weight, pattern in _SIGNATURES:
        m = pattern.search(text)
        if m:
            score += weight
            snippet = m.group(0).strip()
            key = (category, snippet.lower())
            if key not in seen:
                seen.add(key)
                sigs.append({"category": category, "match": snippet[:80]})

    if score >= RISK_THRESHOLDS["high"]:
        risk = "high"
    elif score >= RISK_THRESHOLDS["medium"]:
        risk = "medium"
    elif score > 0:
        risk = "low"
    else:
        risk = "none"

    blocked = risk in ("high", "medium")
    reason = (
        "Detected prompt-injection / jailbreak patterns."
        if blocked else ("Low-confidence signals; allowed." if score else "")
    )
    return InputVerdict(blocked=blocked, risk=risk, score=score,
                        signatures=sigs, reason=reason)


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------
_PII_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("PHONE", re.compile(
        r"(?<!\d)(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("API_KEY", re.compile(r"\b(sk-[a-zA-Z0-9]{16,}|gsk_[a-zA-Z0-9]{16,}|"
                           r"AKIA[0-9A-Z]{16})\b")),
]


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """Replace PII with typed placeholders. Returns (clean_text, kinds_found)."""
    kinds: List[str] = []
    clean = text
    for label, pattern in _PII_PATTERNS:
        if pattern.search(clean):
            kinds.append(label)
            clean = pattern.sub(f"[REDACTED_{label}]", clean)
    return clean, kinds


# ---------------------------------------------------------------------------
# Output system-prompt-leak guard
# ---------------------------------------------------------------------------
# Distinctive markers from TEMPLATE_ME.md — if several appear in an answer, the
# model is likely regurgitating its own instructions.
_LEAK_MARKERS = [
    "ANSWER STRATEGY", "ANTI-PATTERNS", "WORKED EXAMPLES", "## VOICE",
    "## IDENTITY", "Concrete > superlative", "CONTEXT (your memory",
    "Answer now, in first person",
]


def detect_prompt_leak(output: str) -> bool:
    hits = sum(1 for m in _LEAK_MARKERS if m.lower() in output.lower())
    return hits >= 2


def sanitize_output(output: str) -> Tuple[str, dict]:
    """Final pass over a generated answer: block leaks, redact PII."""
    info = {"leak_blocked": False, "pii_redacted": []}
    if detect_prompt_leak(output):
        info["leak_blocked"] = True
        return (
            "I keep my own configuration private — but I'm happy to talk about "
            "my research, projects, or experience.",
            info,
        )
    clean, kinds = redact_pii(output)
    info["pii_redacted"] = kinds
    return clean, info


# Safe canned reply when an input is blocked.
REFUSAL = (
    "That looks like an attempt to manipulate my instructions, so I won't act "
    "on it. Ask me about my research, projects, or experience and I'm all yours."
)
