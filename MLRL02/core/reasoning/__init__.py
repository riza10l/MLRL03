"""
Reasoning subsystem — loading, querying, reasoning, reflection, and knowledge graphs.

Usage:
    from core.reasoning import ReasoningEngine, KnowledgeGraph, load_markdown
"""

__all__ = [
    "RetrievalEngine",
    "QueryClassifier",
    "ReasoningEngine",
    "ReasoningResult",
    "ReflectionEngine",
    "ReflectionResult",
    "PromptBuilder",
    "ConceptLinker",
    "ConceptGraph",
    "KnowledgeGraph",
]

def __getattr__(name):
    if name == "PromptBuilder":
        from core.reasoning.prompt_builder import PromptBuilder as _PB
        return _PB
    if name == "RetrievalEngine":
        from core.reasoning.query import RetrievalEngine as _RE
        return _RE
    if name == "QueryClassifier":
        from core.reasoning.query_classifier import QueryClassifier as _QC
        return _QC
    if name in ("ReasoningEngine", "ReasoningResult"):
        from core.reasoning.reasoning_engine import ReasoningEngine, ReasoningResult
        return {"ReasoningEngine": ReasoningEngine,
                "ReasoningResult": ReasoningResult}[name]
    if name in ("ReflectionEngine", "ReflectionResult"):
        from core.reasoning.reflection_engine import ReflectionEngine, ReflectionResult
        return {"ReflectionEngine": ReflectionEngine,
                "ReflectionResult": ReflectionResult}[name]
    if name in ("ConceptLinker", "ConceptGraph"):
        from core.reasoning.concept_linker import ConceptLinker, ConceptGraph
        return {"ConceptLinker": ConceptLinker, "ConceptGraph": ConceptGraph}[name]
    if name == "KnowledgeGraph":
        from core.reasoning.knowledge_graph import KnowledgeGraph as _KG
        return _KG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
