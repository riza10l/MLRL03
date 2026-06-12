"""
REFLECTION ENGINE — Self-Evaluation Layer

Responsibilities:
    - Evaluate AI-generated responses for quality
    - Score reasoning quality across multiple dimensions
    - Estimate confidence levels
    - Detect weak or incomplete answers
    - Suggest specific improvements

Architecture:
    This module acts as the self-reflection layer. After an answer
    is generated, the reflection engine evaluates it against the
    original question and available context:

        Response → Quality Scoring → Gap Analysis → Improvement Suggestions

    Dimensions evaluated:
    1. Relevance — Does the answer address the question?
    2. Completeness — Is the answer thorough?
    3. Accuracy — Does it align with retrieved context?
    4. Clarity — Is the answer well-structured and readable?
    5. Confidence — How certain should the AI be?

Usage:
    engine = ReflectionEngine()
    result = engine.evaluate(
        question="Apa itu embeddings?",
        answer="Embeddings are vectors...",
        contexts=[context_item_1, context_item_2]
    )
    print(result.score)
    print(result.improvements)
"""

import os
import sys
import re
from dataclasses import dataclass, field
from typing import Optional

# Project root sys.path hack removed from module level (M-1)

from core.memory.memory_context import ContextItem


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

# Scoring weights for overall quality
DIMENSION_WEIGHTS = {
    "relevance": 0.30,
    "completeness": 0.25,
    "accuracy": 0.25,
    "clarity": 0.20,
}

# Thresholds
STRONG_THRESHOLD = 0.75    # Above this = strong answer
WEAK_THRESHOLD = 0.40      # Below this = needs improvement
MIN_ANSWER_LENGTH = 20     # Minimum characters for a valid answer


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class DimensionScore:
    """
    Score for one evaluation dimension.

    Attributes:
        name:       Dimension name
        score:      Score (0.0 – 1.0)
        rationale:  Why this score was given
    """
    name: str
    score: float
    rationale: str


@dataclass
class Improvement:
    """
    A specific suggestion for improvement.

    Attributes:
        type:        Category of improvement
        suggestion:  What to do differently
        priority:    How important this is (high/medium/low)
    """
    type: str
    suggestion: str
    priority: str = "medium"


@dataclass
class ReflectionResult:
    """
    Complete evaluation of a response.

    Attributes:
        question:       Original question
        answer:         The generated response
        overall_score:  Weighted average of all dimensions (0.0 – 1.0)
        dimensions:     Individual dimension scores
        improvements:   Suggestions for better answers
        is_strong:      Whether the answer meets quality threshold
        is_weak:        Whether the answer needs improvement
        confidence:     Estimated confidence in this evaluation
        missing_concepts: Concepts from context not mentioned in answer
    """
    question: str
    answer: str
    overall_score: float = 0.0
    dimensions: list[DimensionScore] = field(default_factory=list)
    improvements: list[Improvement] = field(default_factory=list)
    is_strong: bool = False
    is_weak: bool = False
    confidence: float = 0.0
    missing_concepts: list[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """Letter grade equivalent."""
        if self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.75:
            return "B"
        elif self.overall_score >= 0.60:
            return "C"
        elif self.overall_score >= 0.40:
            return "D"
        return "F"

    def print_report(self):
        """Pretty-print the reflection report."""
        print("\n" + "=" * 60)
        print(f"  🔍 REFLECTION REPORT")
        print("=" * 60)

        print(f"\n📝 Question: {self.question}")
        print(f"📄 Answer: {self.answer[:150]}{'...' if len(self.answer) > 150 else ''}")

        print(f"\n📊 Overall Score: {self.overall_score:.0%} (Grade: {self.grade})")
        print(f"   Strong: {'✅' if self.is_strong else '❌'}  "
              f"Weak: {'⚠️' if self.is_weak else '❌'}  "
              f"Confidence: {self.confidence:.0%}")

        print(f"\n📋 Dimension Scores:")
        for dim in self.dimensions:
            bar = "█" * int(dim.score * 10) + "░" * (10 - int(dim.score * 10))
            print(f"   {dim.name:15s} [{bar}] {dim.score:.0%}")
            print(f"                 {dim.rationale}")

        if self.improvements:
            print(f"\n💡 Improvements ({len(self.improvements)}):")
            for imp in self.improvements:
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(imp.priority, "•")
                print(f"   {icon} [{imp.type}] {imp.suggestion}")

        if self.missing_concepts:
            print(f"\n❌ Missing concepts from context:")
            for concept in self.missing_concepts:
                print(f"   • {concept}")

        print()


# ──────────────────────────────────────────────
#  EVALUATION DIMENSIONS
# ──────────────────────────────────────────────

class RelevanceScorer:
    """
    Dimension 1: Does the answer address the question?

    Strategy: Check if key terms from the question appear in
    the answer, and if the answer's semantic focus matches.
    """

    def score(self, question: str, answer: str) -> DimensionScore:
        """
        Evaluate relevance.

        Args:
            question: The original question
            answer:   The generated response

        Returns:
            DimensionScore with relevance rating
        """
        if not answer.strip():
            return DimensionScore(
                name="relevance",
                score=0.0,
                rationale="Answer is empty.",
            )

        # Extract key terms from question
        q_terms = self._extract_terms(question)
        if not q_terms:
            return DimensionScore(
                name="relevance",
                score=0.5,
                rationale="Could not identify key terms in question.",
            )

        # Check term overlap
        answer_lower = answer.lower()
        matched = sum(1 for t in q_terms if t in answer_lower)
        term_score = matched / len(q_terms)

        # Check if answer directly engages with question intent
        intent_score = self._check_intent_match(question, answer)

        # Combined relevance
        score = min(1.0, term_score * 0.6 + intent_score * 0.4)

        # Rationale
        if score >= 0.8:
            rationale = f"Answer addresses {matched}/{len(q_terms)} key concepts from question."
        elif score >= 0.5:
            rationale = f"Answer partially addresses question ({matched}/{len(q_terms)} key terms found)."
        else:
            rationale = f"Answer barely addresses the question ({matched}/{len(q_terms)} key terms)."

        return DimensionScore(
            name="relevance",
            score=round(score, 3),
            rationale=rationale,
        )

    def _extract_terms(self, text: str) -> list[str]:
        """Extract meaningful terms from text."""
        from core.reasoning.constants import STOP_WORDS
        tokens = re.findall(r'\w+', text.lower())
        terms = [
            t for t in tokens
            if len(t) >= 3 and t not in STOP_WORDS and not t.isdigit()
        ]
        # Keep unique, max 5
        seen = set()
        unique = []
        for t in terms:
            if t not in seen and len(unique) < 5:
                seen.add(t)
                unique.append(t)
        return unique

    def _check_intent_match(self, question: str, answer: str) -> float:
        """Check if the answer matches the question's intent."""
        q_lower = question.lower()

        # "What is" questions → answer should define/explain
        if re.search(r'\b(what|apa)\b.*\b(is|itu|are)\b', q_lower):
            definition_patterns = [
                r'\b(is|are|refers to|merupakan|adalah)',
                r'\b(defined as|didefinisikan)',
                r'\b(type of|jenis|form of|bentuk)',
            ]
            for pattern in definition_patterns:
                if re.search(pattern, answer.lower()):
                    return 0.9
            return 0.4

        # "How" questions → answer should explain process
        if re.search(r'\b(how|bagaimana|cara)\b', q_lower):
            process_patterns = [
                r'\b(work|works|process|step|method|cara|proses)',
                r'\b(first|then|next|finally|pertama|kemudian)',
                r'\d+\.\s',  # Numbered steps
            ]
            for pattern in process_patterns:
                if re.search(pattern, answer.lower()):
                    return 0.85
            return 0.35

        # "Why" questions → answer should give reasons
        if re.search(r'\b(why|kenapa|mengapa)\b', q_lower):
            reason_patterns = [
                r'\b(because|since|due to|karena|oleh karena)',
                r'\b(reason|alasan|cause|penyebab)',
            ]
            for pattern in reason_patterns:
                if re.search(pattern, answer.lower()):
                    return 0.85
            return 0.35

        # Default: partial match by length and engagement
        if len(answer) > 100:
            return 0.6
        return 0.4


class CompletenessScorer:
    """
    Dimension 2: Is the answer thorough enough?

    Strategy: Check answer length, coverage of key concepts,
    and whether multiple aspects of the question are addressed.
    """

    def score(
        self,
        question: str,
        answer: str,
        contexts: Optional[list[ContextItem]] = None,
    ) -> DimensionScore:
        """
        Evaluate completeness.

        Args:
            question:  The original question
            answer:    The generated response
            contexts:  Retrieved context that could inform the answer

        Returns:
            DimensionScore with completeness rating
        """
        if not answer.strip():
            return DimensionScore(
                name="completeness",
                score=0.0,
                rationale="Answer is empty.",
            )

        scores = []
        rationales = []

        # Length score
        word_count = len(answer.split())
        if word_count >= 100:
            length_score = 1.0
            length_rationale = f"Comprehensive answer ({word_count} words)."
        elif word_count >= 50:
            length_score = 0.7
            length_rationale = f"Moderate length ({word_count} words)."
        elif word_count >= 20:
            length_score = 0.4
            length_rationale = f"Brief answer ({word_count} words)."
        else:
            length_score = 0.1
            length_rationale = f"Very short answer ({word_count} words)."

        scores.append(length_score)
        rationales.append(length_rationale)

        # Context coverage (if contexts available)
        if contexts:
            coverage = self._check_context_coverage(answer, contexts)
            scores.append(coverage)
            if coverage >= 0.7:
                rationales.append(f"Covered {coverage:.0%} of available context.")
            else:
                rationales.append(f"Only covered {coverage:.0%} of available context.")

        # Multi-aspect check
        aspect_score = self._check_aspects(question, answer)
        scores.append(aspect_score)
        rationales.append(
            "Addresses multiple aspects." if aspect_score >= 0.7
            else "Single-aspect answer."
        )

        # Average
        overall = sum(scores) / len(scores) if scores else 0.0

        return DimensionScore(
            name="completeness",
            score=round(overall, 3),
            rationale="; ".join(rationales),
        )

    def _check_context_coverage(
        self,
        answer: str,
        contexts: list[ContextItem],
    ) -> float:
        """Check how much of the context is reflected in the answer."""
        if not contexts:
            return 0.5

        answer_lower = answer.lower()
        covered = 0

        for ctx in contexts[:5]:  # Check top 5 contexts
            # Extract key terms from context
            ctx_terms = set(
                w.lower() for w in re.findall(r'\w{4,}', ctx.text)
            )
            if not ctx_terms:
                continue

            # Check how many appear in answer
            matched = sum(1 for t in ctx_terms if t in answer_lower)
            coverage = matched / len(ctx_terms)
            if coverage > 0.15:  # At least 15% overlap
                covered += 1

        return covered / len(contexts) if contexts else 0.0

    def _check_aspects(self, question: str, answer: str) -> float:
        """Check if answer addresses multiple aspects."""
        # Count distinct informational units in answer
        sentences = re.split(r'[.!?]+', answer)
        meaningful = [
            s.strip() for s in sentences
            if len(s.strip().split()) >= 4
        ]

        if len(meaningful) >= 4:
            return 1.0
        elif len(meaningful) >= 2:
            return 0.6
        return 0.3


class AccuracyScorer:
    """
    Dimension 3: Does the answer align with retrieved context?

    Strategy: Compare answer claims against context facts.
    Detect contradictions and unsupported statements.
    """

    def score(
        self,
        answer: str,
        contexts: Optional[list[ContextItem]] = None,
    ) -> DimensionScore:
        """
        Evaluate accuracy against available context.

        Args:
            answer:   The generated response
            contexts: Retrieved context to verify against

        Returns:
            DimensionScore with accuracy rating
        """
        if not contexts:
            return DimensionScore(
                name="accuracy",
                score=0.5,
                rationale="No context available to verify accuracy.",
            )

        # Check for contradictions
        contradictions = self._find_contradictions(answer, contexts)
        if contradictions:
            return DimensionScore(
                name="accuracy",
                score=0.2,
                rationale=(
                    f"Found potential contradictions: "
                    f"{'; '.join(contradictions[:3])}"
                ),
            )

        # Check for context alignment
        alignment = self._check_alignment(answer, contexts)

        if alignment >= 0.7:
            rationale = "Answer aligns well with retrieved context."
        elif alignment >= 0.4:
            rationale = "Answer partially aligns with context."
        else:
            rationale = "Answer diverges from available context."

        return DimensionScore(
            name="accuracy",
            score=round(alignment, 3),
            rationale=rationale,
        )

    def _find_contradictions(
        self,
        answer: str,
        contexts: list[ContextItem],
    ) -> list[str]:
        """Detect potential contradictions between answer and context."""
        contradictions = []

        # Simple negation check
        negation_patterns = [
            (r'\b(not|no|never|tidak|bukan)\b\s+\w+', "negation"),
            (r"\b(don't|doesn't|cannot|can't|won't)\b", "negation"),
        ]

        answer_negations = set()
        for pattern, _ in negation_patterns:
            for match in re.finditer(pattern, answer.lower()):
                answer_negations.add(match.group())

        for ctx in contexts[:3]:
            ctx_lower = ctx.text.lower()
            for negation in answer_negations:
                # Check if context says the opposite
                clean_negation = re.sub(r'[^\w\s]', '', negation).strip()
                words = clean_negation.split()
                if len(words) >= 2:
                    key_word = words[-1]  # The word being negated
                    # Answer negates key_word, but context affirms it
                    # (key_word appears in context without negation nearby)
                    if key_word in ctx_lower:
                        # Check context does NOT also negate this word
                        ctx_has_negation = any(
                            re.search(
                                rf'\b(not|no|never|tidak|bukan)\b\s+\w*{re.escape(key_word)}',
                                ctx_lower,
                            )
                            for _ in [1]  # single check
                        )
                        if not ctx_has_negation:
                            contradictions.append(
                                f"Answer negates '{key_word}' but context affirms it"
                            )

        return contradictions

    def _check_alignment(
        self,
        answer: str,
        contexts: list[ContextItem],
    ) -> float:
        """Check how well the answer aligns with context."""
        answer_lower = answer.lower()
        total_alignment = 0.0
        count = 0

        for ctx in contexts[:5]:
            ctx_terms = set(
                w.lower() for w in re.findall(r'\w{4,}', ctx.text)
                if len(w) >= 4
            )
            if not ctx_terms:
                continue

            matched = sum(1 for t in ctx_terms if t in answer_lower)
            alignment = matched / len(ctx_terms)
            total_alignment += alignment
            count += 1

        return total_alignment / count if count > 0 else 0.5


class ClarityScorer:
    """
    Dimension 4: Is the answer well-structured and readable?

    Strategy: Check formatting, sentence structure, and
    logical flow.
    """

    def score(self, answer: str) -> DimensionScore:
        """
        Evaluate clarity.

        Args:
            answer: The generated response

        Returns:
            DimensionScore with clarity rating
        """
        if not answer.strip():
            return DimensionScore(
                name="clarity",
                score=0.0,
                rationale="Answer is empty.",
            )

        scores = []
        rationales = []

        # Structure score (headings, bullets, paragraphs)
        structure = self._check_structure(answer)
        scores.append(structure)
        if structure >= 0.7:
            rationales.append("Well-structured with clear formatting.")
        else:
            rationales.append("Lacks structural organization.")

        # Readability score (sentence length variation)
        readability = self._check_readability(answer)
        scores.append(readability)
        if readability >= 0.7:
            rationales.append("Sentences are varied and readable.")
        else:
            rationales.append("Sentences are too uniform or too long.")

        # Language quality (no excessive repetition)
        quality = self._check_language_quality(answer)
        scores.append(quality)
        if quality >= 0.7:
            rationales.append("Language is clear and varied.")
        else:
            rationales.append("Language is repetitive or unclear.")

        overall = sum(scores) / len(scores) if scores else 0.0

        return DimensionScore(
            name="clarity",
            score=round(overall, 3),
            rationale="; ".join(rationales),
        )

    def _check_structure(self, text: str) -> float:
        """Check for structural elements."""
        score = 0.0

        # Has paragraphs (double newlines)
        if "\n\n" in text:
            score += 0.3

        # Has bullet points or numbered lists
        if re.search(r'[\-\*\•]\s', text) or re.search(r'\d+\.\s', text):
            score += 0.3

        # Has section markers
        if re.search(r'\*\*.*\*\*', text) or re.search(r'#{1,3}\s', text):
            score += 0.2

        # Has multiple paragraphs
        paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 20]
        if len(paragraphs) >= 2:
            score += 0.2

        return min(1.0, score)

    def _check_readability(self, text: str) -> float:
        """Check sentence-level readability."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.5

        # Average sentence length
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

        # Ideal: 10-25 words per sentence
        if 10 <= avg_len <= 25:
            return 1.0
        elif 5 <= avg_len <= 35:
            return 0.7
        return 0.4

    def _check_language_quality(self, text: str) -> float:
        """Check for repetition and language issues."""
        words = text.lower().split()
        if len(words) < 10:
            return 0.5

        # Check for repeated phrases (3+ words)
        ngrams = {}
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            ngrams[trigram] = ngrams.get(trigram, 0) + 1

        repeats = sum(1 for count in ngrams.values() if count > 2)
        if repeats > 3:
            return 0.3
        elif repeats > 1:
            return 0.6
        return 0.9


# ──────────────────────────────────────────────
#  IMPROVEMENT GENERATOR
# ──────────────────────────────────────────────

class ImprovementGenerator:
    """
    Generate specific improvement suggestions based on evaluation.
    """

    def generate(
        self,
        question: str,
        answer: str,
        dimensions: list[DimensionScore],
        contexts: Optional[list[ContextItem]] = None,
    ) -> list[Improvement]:
        """
        Generate improvement suggestions.

        Args:
            question:   Original question
            answer:     Generated response
            dimensions: Evaluation scores
            contexts:   Available context

        Returns:
            List of Improvement suggestions
        """
        improvements = []

        for dim in dimensions:
            if dim.score >= 0.7:
                continue  # No improvement needed

            if dim.name == "relevance" and dim.score < 0.5:
                improvements.append(Improvement(
                    type="relevance",
                    suggestion="Address the key terms from the question more directly. "
                              "Start with a clear definition or direct answer.",
                    priority="high",
                ))

            elif dim.name == "completeness" and dim.score < 0.5:
                improvements.append(Improvement(
                    type="completeness",
                    suggestion="Expand the answer with more details. "
                              "Cover multiple aspects of the topic.",
                    priority="high" if dim.score < 0.3 else "medium",
                ))

            elif dim.name == "accuracy" and dim.score < 0.5:
                improvements.append(Improvement(
                    type="accuracy",
                    suggestion="Verify claims against available context. "
                              "Remove or qualify statements not supported by memory.",
                    priority="high",
                ))

            elif dim.name == "clarity" and dim.score < 0.5:
                improvements.append(Improvement(
                    type="clarity",
                    suggestion="Improve structure with paragraphs, bullet points, "
                              "or numbered steps. Vary sentence length.",
                    priority="medium",
                ))

        # Missing context concepts
        if contexts:
            missing = self._find_missing_concepts(answer, contexts)
            if missing:
                improvements.append(Improvement(
                    type="coverage",
                    suggestion=f"Missing key concepts: {', '.join(missing[:3])}. "
                              f"Incorporate these from retrieved memory.",
                    priority="medium",
                ))

        # Answer too short
        if len(answer.split()) < 20:
            improvements.append(Improvement(
                type="length",
                suggestion="Answer is too brief. Provide more explanation and context.",
                priority="high",
            ))

        return improvements

    def _find_missing_concepts(
        self,
        answer: str,
        contexts: list[ContextItem],
    ) -> list[str]:
        """Find important concepts from context not mentioned in answer."""
        answer_lower = answer.lower()
        missing = []

        for ctx in contexts[:3]:
            # Extract key terms from context
            ctx_terms = [
                w.lower() for w in re.findall(r'\w{5,}', ctx.text)
                if len(w) >= 5 and w.lower() not in {
                    "there", "their", "would", "could", "should", "about",
                    "other", "these", "those", "which", "through",
                }
            ]

            # Find terms not in answer
            for term in ctx_terms:
                if term not in answer_lower and term not in missing:
                    missing.append(term)
                    if len(missing) >= 5:
                        return missing

        return missing


# ──────────────────────────────────────────────
#  REFLECTION ENGINE (Main Class)
# ──────────────────────────────────────────────

class ReflectionEngine:
    """
    Main self-evaluation engine.

    Orchestrates all scoring dimensions and generates
    improvement suggestions.

    Usage:
        engine = ReflectionEngine()
        result = engine.evaluate(
            question="What are embeddings?",
            answer="Embeddings are...",
            contexts=[ctx1, ctx2]
        )
    """

    def __init__(self):
        self._relevance = RelevanceScorer()
        self._completeness = CompletenessScorer()
        self._accuracy = AccuracyScorer()
        self._clarity = ClarityScorer()
        self._improver = ImprovementGenerator()

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: Optional[list[ContextItem]] = None,
    ) -> ReflectionResult:
        """
        Full evaluation pipeline.

        Args:
            question:  The original question
            answer:    The generated response
            contexts:  Retrieved context (optional)

        Returns:
            ReflectionResult with scores and improvements
        """
        dimensions: list[DimensionScore] = []

        # Score each dimension
        dimensions.append(self._relevance.score(question, answer))
        dimensions.append(self._completeness.score(question, answer, contexts))
        dimensions.append(self._accuracy.score(answer, contexts))
        dimensions.append(self._clarity.score(answer))

        # Compute weighted overall score
        overall = 0.0
        for dim in dimensions:
            weight = DIMENSION_WEIGHTS.get(dim.name, 0.25)
            overall += dim.score * weight

        overall = round(min(1.0, overall), 3)

        # Generate improvements
        improvements = self._improver.generate(
            question, answer, dimensions, contexts
        )

        # Find missing concepts
        missing = []
        if contexts:
            missing = self._improver._find_missing_concepts(answer, contexts)

        return ReflectionResult(
            question=question,
            answer=answer,
            overall_score=overall,
            dimensions=dimensions,
            improvements=improvements,
            is_strong=overall >= STRONG_THRESHOLD,
            is_weak=overall < WEAK_THRESHOLD,
            confidence=self._evaluate_confidence(dimensions),
            missing_concepts=missing,
        )

    def _evaluate_confidence(
        self, dimensions: list[DimensionScore]
    ) -> float:
        """
        Estimate confidence in the evaluation.

        Higher confidence when scores are consistent.
        """
        if not dimensions:
            return 0.0

        scores = [d.score for d in dimensions]
        avg = sum(scores) / len(scores)

        # Variance penalizes confidence
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        confidence = max(0.0, 1.0 - variance * 2)

        return round(min(1.0, confidence), 3)


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure project root is on path for standalone execution
    import os
    import sys
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    print("=" * 60)
    print("  REFLECTION ENGINE — Quick Test")
    print("=" * 60 + "\n")

    engine = ReflectionEngine()

    # ── Test 1: Strong answer ──
    print("--- Test 1: Strong Answer ---")
    result1 = engine.evaluate(
        question="What are embeddings?",
        answer=(
            "Embeddings are numerical representations of text as high-dimensional vectors. "
            "They capture the semantic meaning of text, not just keywords. "
            "Similar texts produce similar embeddings, which allows machines "
            "to understand relationships between concepts. "
            "\n\nVector databases store these embeddings for fast semantic retrieval. "
            "When you search, the system converts your query to an embedding "
            "and finds the closest matching vectors using cosine similarity. "
            "\n\nThis is fundamentally different from keyword search because "
            "it understands meaning rather than exact word matches."
        ),
        contexts=[
            ContextItem(
                text="Embeddings are numerical representations of text. "
                     "Vector databases store them for semantic search. "
                     "Similar texts have similar embeddings.",
                source="ai_notes.md",
                score=0.85,
            ),
        ],
    )
    result1.print_report()

    # ── Test 2: Weak answer ──
    print("\n" + "-" * 60)
    print("--- Test 2: Weak Answer ---")
    result2 = engine.evaluate(
        question="How does semantic search work?",
        answer="It searches for things.",
        contexts=[
            ContextItem(
                text="Semantic search finds results based on meaning "
                     "rather than exact keyword matches. By comparing "
                     "embedding vectors using cosine similarity.",
                source="notes.md",
                score=0.80,
            ),
        ],
    )
    result2.print_report()
