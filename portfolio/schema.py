from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import json, numpy as np

@dataclass
class Portfolio:
    name: str
    weights: Dict[str, float]

    def as_vector(self, symbols: List[str]) -> np.ndarray:
        w = np.array([self.weights.get(s, 0.0) for s in symbols], dtype=float)
        s = w.sum()
        return w / s if s > 0 else w

def load_portfolio(json_str: str) -> Portfolio:
    obj = json.loads(json_str)
    return Portfolio(name=obj.get("name","portfolio"), weights=obj["weights"])
