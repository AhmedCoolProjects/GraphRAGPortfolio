"""
Local smoke test for the knowledge graph + retrieval. No API keys needed.

    python scripts/test_graph.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kg.graph_store import get_graph  # noqa: E402


QUESTIONS = [
    "Tell me about your time at PayRue",
    "What did you build that combines security and machine learning?",
    "What's your tech stack?",
    "What books have you read?",
    "Where did you study?",
    "what languages do you speak?",
    "the weather today",  # should link nothing
]


def main() -> None:
    kg = get_graph()
    print(f"Graph loaded: {kg.stats['nodes']} nodes, {kg.stats['edges']} edges\n")

    for q in QUESTIONS:
        res = kg.retrieve(q)
        print(f"Q: {q}")
        print(f"   seeds: {res['seeds']}")
        print(f"   path : {len(res['graph_path']['nodes'])} nodes / "
              f"{len(res['graph_path']['edges'])} edges")
        for fact in res["facts"][:6]:
            print(f"     - {fact}")
        if len(res["facts"]) > 6:
            print(f"     ... (+{len(res['facts']) - 6} more facts)")
        if res["images"]:
            print(f"   images: {[i['image'] for i in res['images']]}")
        print()


if __name__ == "__main__":
    main()
