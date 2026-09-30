from dataclasses import dataclass, field
from typing import Literal, Any, Optional, List, Dict


@dataclass
class Evidence:
    """
    Represents a single piece of analytical evidence supporting a parameter estimation
    or hypothesis validation result.
    """
    description: str
    source: str
    score: float  # Confidence/strength score [0.0 - 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class Parameter:
    """
    Stores an extracted or known parameter along with provenance and supporting evidence.
    """
    name: str
    value: Any
    unit: str
    source: Literal["KNOWN", "ESTIMATED", "ASSUMED", "UNKNOWN"]
    method: Optional[str] = None
    confidence: Optional[float] = None
    evidence: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "method": self.method,
            "confidence": self.confidence,
            "evidence": [ev.to_dict() for ev in self.evidence]
        }


@dataclass
class EvidenceCollection:
    """
    Container for managing and aggregating multiple evidence items.
    """
    items: List[Evidence] = field(default_factory=list)

    def add(self, evidence: Evidence) -> None:
        self.items.append(evidence)

    def mean_score(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.score for item in self.items) / len(self.items)

    def weighted_score(self, weights: Optional[List[float]] = None) -> float:
        if not self.items:
            return 0.0
        if weights is None:
            return self.mean_score()
        if len(weights) != len(self.items):
            raise ValueError("Weights length must match evidence items length.")
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(w * ev.score for w, ev in zip(weights, self.items)) / total_weight

    def filter_by_source(self, source: str) -> List[Evidence]:
        return [item for item in self.items if item.source == source]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.items]
