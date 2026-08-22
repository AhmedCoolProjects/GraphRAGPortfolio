"""
Embedded knowledge-graph store (NetworkX) for the agentic GraphRAG twin.

Responsibilities:
  * Load + validate knowledge_graph.json against schema.py.
  * Entity-link a free-text question to seed nodes (alias / token matching).
  * Expand seeds into a k-hop subgraph (the "context graph" for a question).
  * Serialize that subgraph into compact natural-language facts for the LLM.
  * Emit the exact nodes + edges touched, so the frontend can highlight the
    traversal path used to answer.

No external services, no API keys. Pure local graph reasoning.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .schema import is_valid_edge_type, is_valid_node_type, NODE_TYPES

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")

# Words too generic to be useful seeds during token matching.
_STOPWORDS = {
    "what", "who", "when", "where", "which", "how", "why", "is", "are", "the",
    "a", "an", "of", "to", "in", "on", "for", "and", "or", "your", "you",
    "do", "does", "did", "have", "has", "tell", "me", "about", "with", "can",
    "was", "were", "his", "her", "their", "that", "this", "it", "i", "im",
    "whats", "work", "use", "used", "using", "like", "any", "some", "more",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9#+&. ]", " ", text.lower()).strip()


# Intent rules map a free-text question with no explicit entity to the node
# TYPES it is really asking about. Each rule is (keywords, node_types). The
# first keyword found wins for that rule. Order matters: more specific first.
_INTENT_RULES: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
    (("stack", "technolog", "framework", "library", "libraries", "tooling",
      "tools", "programming"), ("skill",)),
    (("study", "studied", "education", "educational", "degree", "university",
      "school", "academic", "graduate", "graduated"), ("org",)),
    (("speak", "spoken", "fluent", "multilingual", "languages", "language"),
     ("language",)),
    (("experience", "career", "job", "jobs", "employer", "employers",
      "companies", "worked", "professional"), ("company", "role")),
    (("project", "projects", "built", "build", "portfolio of"), ("project",)),
    (("research", "researching", "thesis", "phd", "ph.d", "doctoral", "paper",
      "papers"), ("area", "project")),
    (("book", "books", "reading", "read"), ("book",)),
    (("hobby", "hobbies", "free time", "fun", "leisure", "outside work"),
     ("hobby",)),
    (("skill", "skills", "good at", "expertise", "proficient"), ("skill",)),
]


class KnowledgeGraph:
    def __init__(self, path: str = GRAPH_PATH):
        self.path = path
        self.G: nx.DiGraph = nx.DiGraph()
        self._alias_index: Dict[str, str] = {}      # alias -> node_id
        self._meta: Dict = {}
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._meta = data.get("meta", {})

        for node in data["nodes"]:
            nid, ntype = node["id"], node["type"]
            if not is_valid_node_type(ntype):
                raise ValueError(f"Node {nid} has unknown type '{ntype}'")
            self.G.add_node(
                nid,
                type=ntype,
                label=node["label"],
                description=node.get("description", ""),
                aliases=node.get("aliases", []),
                image=node.get("image"),
                meta=node.get("meta", {}),
            )
            # Build the alias index (label + explicit aliases).
            for alias in [node["label"], *node.get("aliases", [])]:
                key = _normalize(alias)
                if key:
                    self._alias_index.setdefault(key, nid)

        for edge in data["edges"]:
            src, dst, etype = edge["source"], edge["target"], edge["type"]
            if not is_valid_edge_type(etype):
                raise ValueError(f"Edge {src}->{dst} has unknown type '{etype}'")
            if src not in self.G or dst not in self.G:
                raise ValueError(f"Edge references missing node: {src} -> {dst}")
            self.G.add_edge(src, dst, type=etype, meta=edge.get("meta", {}))

    # --------------------------------------------------------- entity linking
    def _detect_intent_types(self, q: str) -> Set[str]:
        """Map an entity-less question to the node TYPES it asks about."""
        types: Set[str] = set()
        for keywords, node_types in _INTENT_RULES:
            if any(kw in q for kw in keywords):
                types.update(node_types)
        return types

    def _nodes_of_types(self, types: Set[str]) -> List[str]:
        return [
            nid for nid, data in self.G.nodes(data=True)
            if data["type"] in types and data["type"] != "person"
        ]

    def link_entities(self, question: str, limit: int = 12) -> List[str]:
        """Return seed node ids the question most likely refers to.

        Strategy: (1) longest-alias phrase matches, then (2) single-token
        fallback, then (3) intent rules that expand to all nodes of a type
        (e.g. "tech stack" -> every skill). Person root is never a seed.
        """
        q = _normalize(question)
        scored: Dict[str, int] = {}

        # (1) phrase matches — prefer longer aliases (more specific).
        for alias, nid in self._alias_index.items():
            if len(alias) < 2 or alias in _STOPWORDS:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", q):
                scored[nid] = max(scored.get(nid, 0), len(alias))

        # (2) token fallback if no phrase matched.
        if not scored:
            for tok in q.split():
                if tok in _STOPWORDS or len(tok) < 3:
                    continue
                nid = self._alias_index.get(tok)
                if nid:
                    scored[nid] = max(scored.get(nid, 0), len(tok))

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        seeds = [nid for nid, _ in ranked if self.G.nodes[nid]["type"] != "person"]
        seeds = seeds[:limit]

        # (3) intent expansion — add every node of an asked-about type.
        intent_types = self._detect_intent_types(q)
        if intent_types:
            for nid in self._nodes_of_types(intent_types):
                if nid not in seeds:
                    seeds.append(nid)

        return seeds

    # ----------------------------------------------------------- subgraph hop
    def k_hop_subgraph(
        self, seeds: List[str], hops: int = 2, max_nodes: int = 40
    ) -> Tuple[Set[str], List[Tuple[str, str, Dict]]]:
        """BFS outward from seeds over the undirected view, up to `hops`.

        The person root is a super-hub: nearly every node connects to it, so
        expanding *through* it would pull in the whole graph. We therefore add
        the root as an anchor but never traverse its neighbours — keeping each
        question's subgraph tight and relevant.

        Returns (node_ids, edges) where edges lie entirely within the node set.
        """
        if not seeds:
            return set(), []

        root = self._meta.get("root_id", "person:ahmed")
        undirected = self.G.to_undirected(as_view=True)
        visited: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)

        for _ in range(hops):
            nxt: Set[str] = set()
            for node in frontier:
                if node == root:  # never expand the hub
                    continue
                for nbr in undirected.neighbors(node):
                    if nbr not in visited and nbr != root:
                        nxt.add(nbr)
            if not nxt:
                break
            visited |= nxt
            frontier = nxt
            if len(visited) >= max_nodes:
                break

        # Anchor every answer to Ahmed, but only after expansion is done.
        if root in self.G:
            visited.add(root)

        edges: List[Tuple[str, str, Dict]] = [
            (u, v, d)
            for u, v, d in self.G.edges(data=True)
            if u in visited and v in visited
        ]
        return visited, edges

    # ------------------------------------------------------- context for LLM
    def retrieve(self, question: str, hops: int = 2) -> Dict:
        """Full graph-retrieval step for one question.

        Returns a dict with:
          seeds        : seed node ids
          facts        : list of natural-language fact strings (LLM context)
          images       : image paths attached to involved nodes
          graph_path   : {nodes:[...], edges:[...]} for the frontend viz
        """
        seeds = self.link_entities(question)
        nodes, edges = self.k_hop_subgraph(seeds, hops=hops)

        facts = [self._edge_to_sentence(u, v, d) for u, v, d in edges]
        # De-dup while preserving order.
        seen: Set[str] = set()
        facts = [f for f in facts if not (f in seen or seen.add(f))]

        images: List[Dict[str, str]] = []
        for nid in nodes:
            img = self.G.nodes[nid].get("image")
            if img:
                images.append({"image": img, "alt": self.G.nodes[nid]["label"]})

        return {
            "seeds": seeds,
            "facts": facts,
            "images": images,
            "graph_path": self._serialize_path(nodes, edges, seeds),
        }

    # --------------------------------------------------------- serialization
    def _edge_to_sentence(self, u: str, v: str, d: Dict) -> str:
        ul = self.G.nodes[u]["label"]
        vl = self.G.nodes[v]["label"]
        verb = {
            "STUDIES_AT": "currently studies at",
            "STUDIED_AT": "studied at",
            "WORKED_AT": "worked at",
            "HELD_ROLE": "held the role of",
            "ROLE_AT": "was a role at",
            "BUILT": "built",
            "BUILT_FOR": "was built for",
            "CONDUCTED_AT": "was conducted at",
            "USES": "uses",
            "SKILLED_IN": "is skilled in",
            "RESEARCHES": "researches",
            "ADDRESSES": "addresses",
            "PART_OF": "is part of",
            "READS": "has read",
            "SPEAKS": "speaks",
            "ENJOYS": "enjoys",
            "RELATED_TO": "is related to",
        }.get(d["type"], d["type"].lower().replace("_", " "))

        extra = ""
        meta = d.get("meta", {})
        bits = []
        for key in ("role", "degree", "track", "dates", "level"):
            if meta.get(key):
                bits.append(f"{key}: {meta[key]}")
        if bits:
            extra = f" ({'; '.join(bits)})"
        return f"{ul} {verb} {vl}{extra}."

    def _serialize_path(
        self, nodes: Set[str], edges: List[Tuple[str, str, Dict]], seeds: List[str]
    ) -> Dict:
        seed_set = set(seeds)
        return {
            "nodes": [
                {
                    "id": nid,
                    "label": self.G.nodes[nid]["label"],
                    "type": self.G.nodes[nid]["type"],
                    "seed": nid in seed_set,
                }
                for nid in nodes
            ],
            "edges": [
                {"source": u, "target": v, "type": d["type"]} for u, v, d in edges
            ],
        }

    # ----------------------------------------------------- full export (viz)
    def export_full(self) -> Dict:
        """The entire graph, for the frontend's ambient visualization."""
        return {
            "meta": self._meta,
            "legend": [{"id": k, "label": v} for k, v in NODE_TYPES.items()],
            "nodes": [
                {
                    "id": nid,
                    "label": data["label"],
                    "type": data["type"],
                    "description": data["description"],
                    "image": data.get("image"),
                }
                for nid, data in self.G.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "type": d["type"]}
                for u, v, d in self.G.edges(data=True)
            ],
        }

    # -------------------------------------------------------------- helpers
    @property
    def stats(self) -> Dict[str, int]:
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges()}


# Module-level singleton (cheap to hold; the graph is tiny).
_KG: Optional[KnowledgeGraph] = None


def get_graph() -> KnowledgeGraph:
    global _KG
    if _KG is None:
        _KG = KnowledgeGraph()
    return _KG
