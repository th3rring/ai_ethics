from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationTarget:
    title: str
    body: str
    tags: List[str]

@dataclass
class ClassificationResult:
    target: ClassificationTarget
    sentiment_score: float
    positive_words: List[str]
    negative_words: List[str]
