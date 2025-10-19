
from dataclasses import dataclass

@dataclass
class ScoreRubric:
    rubric_version: str = "v1"
    weights: dict = None

DEFAULT_WEIGHTS = {
    "empathy": 0.25,
    "accuracy": 0.35,
    "tone": 0.15,
    "resolution": 0.25
}
