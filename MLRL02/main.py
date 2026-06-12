"""
MLRL02 — Adaptive AI Memory & Reasoning System
===============================================

Unified CLI for the entire MLRL02 system.

Usage:
    python main.py                          → Interactive chat mode
    python main.py chat "question"          → One-shot chat with AI
    python main.py chat "question" --agent  → Deep reasoning mode (AgentLoop)
    python main.py search "query"           → Semantic memory search
    python main.py reason "question"        → Deep reasoning analysis
    python main.py ingest                   → Ingest markdown into memory
    python main.py analyze                  → Analyze project structure
    python main.py graph                    → Build & export knowledge graph
    python main.py status                   → System-wide status
    python main.py docs                     → Generate documentation
"""

import sys
import os

# Configure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeErrors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mlrl02 import MLRL02


# ──────────────────────────────────────────────
#  CLI COMMANDS
# ──────────────────────────────────────────────

def cmd_chat(question: str, use_agent: bool = False):
    """Chat with the AI system."""
    system = MLRL02()
    system.boot()

    if use_agent:
        print(f"\n🧠 Deep reasoning mode: \"{question}\"\n")
    else:
        print(f"\n💬 Chat mode: \"{question}\"\n")

    answer = system.chat_with_ai(question, use_agent=use_agent)
    print(f"🤖 {answer}\n")


def cmd_interactive():
    """Interactive chat loop."""
    system = MLRL02()
    system.boot()

    print("\n" + "=" * 50)
    print("  🧠 MLRL02 — Interactive Chat")
    print("  Type 'quit' to exit, 'stats' for system status")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if user_input.lower() == "stats":
            stats = system.status()
            for section, data in stats.items():
                print(f"  {section}: {data}")
            print()
            continue

        answer = system.route_query(user_input)
        print(f"\n🤖 {answer}\n")


def cmd_search(query: str, top_k: int = 5):
    """Search semantic memory."""
    system = MLRL02()
    system.boot()

    print(f"\n🔍 Searching memory for: \"{query}\"\n")

    results = system.search_memory(query, top_k=top_k)

    if not results:
        print("  No results found.\n")
        return

    for i, ctx in enumerate(results, 1):
        print(f"  [{i}] {ctx.source} (score: {ctx.score:.0%}, quality: {ctx.quality:.0%})")
        print(f"      {ctx.text[:120]}...")
        print()

    print(f"  Found {len(results)} result(s).\n")


def cmd_reason(question: str):
    """Deep reasoning about a question."""
    system = MLRL02()
    system.boot()

    print(f"\n🧠 Reasoning about: \"{question}\"\n")

    result = system.reason_about(question)

    print(f"  Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
    print(f"\n  Intent:     {result['intent']}")
    print(f"  Key terms:  {', '.join(result['key_terms'][:5])}")
    print(f"  Steps:      {result['steps']}")
    print(f"  Contexts:   {result['contexts_used']}")
    print(f"  Links:      {result['concept_links']}")
    print(f"  Confidence: {result['confidence']:.0%}\n")


def cmd_ingest():
    """Ingest markdown files into memory."""
    system = MLRL02()
    system.boot()

    print("\n📥 Ingesting markdown files...\n")

    stats = system.ingest()

    print(f"  Vector store:  {stats['vector_count']} documents")
    print(f"  Concepts:      {stats['concept_count']}")
    print(f"  Concept links: {stats['link_count']}")
    print(f"  Graph nodes:   {stats['graph_nodes']}\n")


def cmd_analyze():
    """Analyze project structure."""
    system = MLRL02()
    system.boot()

    print("\n📊 Analyzing project...\n")

    result = system.analyze_project()

    print(f"  Files:         {result['files']}")
    print(f"  Directories:   {result['dirs']}")
    print(f"  Knowledge:     {result['knowledge_docs']} docs")
    print(f"  Concepts:      {result['concepts']}")
    print(f"  Suggestions:   {result['suggestions']}\n")

    if result.get("improvements"):
        print("  💡 Improvements:")
        for imp in result["improvements"]:
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(imp["priority"], "•")
            print(f"    {icon} [{imp['priority']}] {imp['title']}")
            print(f"        {imp['action']}")
        print()


def cmd_graph():
    """Build and export knowledge graph."""
    system = MLRL02()
    system.boot()

    print("\n🕸️  Building knowledge graph...\n")

    # First ingest if graph is empty
    if system.knowledge_graph.node_count == 0:
        system.ingest()
        print()

    filepath = system.export_knowledge_graph()
    s = system.knowledge_graph.summary()

    print(f"  Nodes:       {s['nodes']}")
    print(f"  Edges:       {s['edges']}")
    print(f"  Density:     {s['density']:.4f}")
    print(f"  Communities: {s['communities']}")
    print(f"\n  Exported to: {filepath}\n")


def cmd_status():
    """Show system-wide status."""
    system = MLRL02()
    system.boot()

    print("\n" + "=" * 50)
    print("   MLRL02 — System Status")
    print("=" * 50 + "\n")

    stats = system.status()

    for section, data in stats.items():
        print(f"  {section}:")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"    {k}: {v}")
        else:
            print(f"    {data}")
        print()


def cmd_docs():
    """Generate project documentation."""
    system = MLRL02()
    system.boot()

    print("\n📝 Generating documentation...\n")

    report = system.workspace_agent.generate_docs()
    log = system.workspace_agent.log_evolution()

    print(f"  Report: {len(report)} characters")
    print(f"  Evolution log: {log[:80]}...\n")


# ──────────────────────────────────────────────
#  CLI ROUTER
# ──────────────────────────────────────────────

def print_usage():
    """Print usage information."""
    print("""
MLRL02 — Adaptive AI Memory & Reasoning System

Usage:
    python main.py                          → Interactive chat mode
    python main.py chat "question"          → One-shot chat with AI
    python main.py chat "q" --agent         → Deep reasoning mode
    python main.py search "query"           → Semantic memory search
    python main.py reason "question"        → Deep reasoning analysis
    python main.py ingest                   → Ingest markdown into memory
    python main.py analyze                  → Analyze project structure
    python main.py graph                    → Build & export knowledge graph
    python main.py status                   → System-wide status
    python main.py docs                     → Generate documentation
""")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        cmd_interactive()
        return

    command = sys.argv[1].lower()

    if command == "chat":
        # Extract question from remaining args, skip --flags
        question_parts = []
        use_agent = False
        for arg in sys.argv[2:]:
            if arg == "--agent":
                use_agent = True
            else:
                question_parts.append(arg)

        question = " ".join(question_parts)
        if not question:
            print("Error: Please provide a question.")
            print("Usage: python main.py chat \"your question\"")
            return
        cmd_chat(question, use_agent=use_agent)

    elif command == "search":
        query = " ".join(sys.argv[2:])
        if not query:
            print("Error: Please provide a search query.")
            return
        cmd_search(query)

    elif command == "reason":
        question = " ".join(sys.argv[2:])
        if not question:
            print("Error: Please provide a question.")
            return
        cmd_reason(question)

    elif command == "ingest":
        cmd_ingest()

    elif command == "analyze":
        cmd_analyze()

    elif command == "graph":
        cmd_graph()

    elif command == "status":
        cmd_status()

    elif command == "docs":
        cmd_docs()

    elif command in ("help", "--help", "-h"):
        print_usage()

    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
