"""
Knowledge-graph schema for Ahmed Bargady's digital twin.

This is the contract for the "context graph": the closed set of node types and
relationship types the graph is allowed to contain. Keeping it explicit makes
the graph self-documenting, lets graph_store.py validate on load, and gives the
frontend a stable legend to colour/group nodes by.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------
# Each node in knowledge_graph.json has a "type" drawn from this set. The value
# is a short human label used by the visualization legend.
NODE_TYPES: Dict[str, str] = {
    "person": "Person",
    "org": "Institution",
    "company": "Company",
    "role": "Role",
    "project": "Project",
    "skill": "Skill / Technology",
    "area": "Research area",
    "topic": "Domain",
    "book": "Book",
    "language": "Language",
    "hobby": "Hobby",
}

# ---------------------------------------------------------------------------
# Edge (relationship) types
# ---------------------------------------------------------------------------
# Directed relationships. The tuple documents the (typical source -> target)
# node types so the schema reads like a sentence and stays auditable.
EDGE_TYPES: Dict[str, str] = {
    "STUDIED_AT": "person studied at an institution",
    "STUDIES_AT": "person currently studies at an institution",
    "WORKED_AT": "person worked at a company / org",
    "HELD_ROLE": "person held a role",
    "ROLE_AT": "role was held at a company / org",
    "BUILT": "person built a project",
    "BUILT_FOR": "project was built for a company / org",
    "CONDUCTED_AT": "project was conducted at an institution",
    "USES": "project uses a skill / technology",
    "SKILLED_IN": "person is skilled in a technology",
    "RESEARCHES": "person researches a research area",
    "ADDRESSES": "project / area addresses a domain",
    "PART_OF": "project is part of a larger effort",
    "READS": "person read a book",
    "SPEAKS": "person speaks a language",
    "ENJOYS": "person enjoys a hobby",
    "RELATED_TO": "generic association between two domains / areas",
}

# Node types that are reasonable retrieval "seeds" (things a visitor asks about).
SEEDABLE_TYPES = set(NODE_TYPES) - {"person"}


def is_valid_node_type(node_type: str) -> bool:
    return node_type in NODE_TYPES


def is_valid_edge_type(edge_type: str) -> bool:
    return edge_type in EDGE_TYPES


def node_type_labels() -> List[Dict[str, str]]:
    """Legend payload for the frontend: [{id, label}, ...]."""
    return [{"id": k, "label": v} for k, v in NODE_TYPES.items()]
