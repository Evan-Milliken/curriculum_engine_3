"""
Knowledge Graph module — Methodology Section 3 & 4.

Loads a domain concept DAG (nodes = concepts, edges = prerequisite-of)
and exposes:
  - prerequisites_satisfied(concept, mastery_state, threshold)
  - topo_order()  -> a valid learning order respecting all prerequisite edges
  - knowledge_gap(mastery_state, target_concept, threshold) -> ordered list of
    concepts on the dependency path to `target_concept` whose mastery is below
    threshold (Methodology Section 3: Knowledge Gap Identification)
"""
from __future__ import annotations
import json
from pathlib import Path
import networkx as nx

DATA_PATH = Path(__file__).parent / "data" / "knowledge_graph.json"


class KnowledgeGraph:
    def __init__(self, path: Path = DATA_PATH):
        with open(path) as f:
            raw = json.load(f)
        self.domain = raw["domain"]
        self.quiz = raw["quiz"]
        self.graph = nx.DiGraph()
        for c in raw["concepts"]:
            self.graph.add_node(c["id"], name=c["name"])
            for prereq in c["prerequisites"]:
                # edge direction: prereq -> concept ("prerequisite-of")
                self.graph.add_edge(prereq, c["id"])
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Knowledge graph must be a DAG (no circular prerequisites)")
        # topological "depth" per node (generation index) — used only for
        # frontend layout (drawing left-to-right by prerequisite depth),
        # not for any scoring/optimization logic.
        for depth, generation in enumerate(nx.topological_generations(self.graph)):
            for n in generation:
                self.graph.nodes[n]["depth"] = depth

    def concept_ids(self) -> list[str]:
        return list(self.graph.nodes)

    def name_of(self, concept_id: str) -> str:
        return self.graph.nodes[concept_id].get("name", concept_id)

    def direct_prerequisites(self, concept_id: str) -> list[str]:
        return list(self.graph.predecessors(concept_id))

    def ancestors(self, concept_id: str) -> set[str]:
        """All transitive prerequisites of a concept."""
        return nx.ancestors(self.graph, concept_id)

    def topo_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def prerequisites_satisfied(
        self, concept_id: str, mastery: dict[str, float], threshold: float = 0.6
    ) -> bool:
        for prereq in self.direct_prerequisites(concept_id):
            if mastery.get(prereq, 0.0) < threshold:
                return False
        return True

    def knowledge_gap(
        self, mastery: dict[str, float], target_concept: str, threshold: float = 0.6
    ) -> list[str]:
        """
        Returns concepts on the path to target_concept (including target)
        that are below mastery threshold, ordered so prerequisites come first.
        This implements Methodology Section 3 directly: gaps are identified
        relative to the target, not by popularity/similarity.
        """
        if target_concept not in self.graph:
            raise ValueError(f"Unknown target concept: {target_concept}")
        relevant = self.ancestors(target_concept) | {target_concept}
        gap_nodes = {c for c in relevant if mastery.get(c, 0.0) < threshold}
        # order by topological position so prerequisites precede dependents
        order = self.topo_order()
        return [c for c in order if c in gap_nodes]


    def to_viz_json(self, mastery: dict[str, float] | None = None) -> dict:
        """
        Node/edge structure for the frontend's D3 knowledge-map view.
        If `mastery` is given, each node carries the learner's current
        mastery so the frontend can color nodes (locked/available/mastered)
        without recomputing anything client-side.
        """
        mastery = mastery or {}
        nodes = [
            {
                "id": n,
                "name": self.name_of(n),
                "mastery": mastery.get(n, 0.0),
                "depth": self.graph.nodes[n].get("depth"),
            }
            for n in self.graph.nodes
        ]
        edges = [{"source": u, "target": v} for u, v in self.graph.edges]
        return {"domain": self.domain, "nodes": nodes, "edges": edges}


_kg_registry: dict[str, KnowledgeGraph] = {}
DOMAIN_FILES = {
    "demo_ml": DATA_PATH,
    "cs_lecturebank": Path(__file__).parent / "data" / "knowledge_graph_lecturebank.json",
}


def get_knowledge_graph(domain: str = "demo_ml") -> KnowledgeGraph:
    if domain not in _kg_registry:
        if domain not in DOMAIN_FILES:
            raise ValueError(f"Unknown domain '{domain}'. Valid: {list(DOMAIN_FILES.keys())}")
        _kg_registry[domain] = KnowledgeGraph(DOMAIN_FILES[domain])
    return _kg_registry[domain]


def list_domains() -> list[dict]:
    out = []
    for domain_id, path in DOMAIN_FILES.items():
        kg = get_knowledge_graph(domain_id)
        out.append({
            "id": domain_id,
            "label": kg.domain,
            "num_concepts": kg.graph.number_of_nodes(),
            "num_edges": kg.graph.number_of_edges(),
            "has_quiz": len(kg.quiz) > 0,
        })
    return out
