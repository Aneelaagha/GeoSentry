"""Sanity check for the Change-1 fusion rewrite: recompute the previous Kabaleqa 2024-07-15
result (LLM confidence 0.32, viirs_z 2.02, rainfall 33rd pct, area 473 ha) with the new
magnitude-scaled likelihood ratios and confirm it no longer clears the 0.72 alert gate.
Run with: python -m scripts.test_fusion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.bayes import fuse_confidence

PRIOR = 0.05
EVIDENCE = {
    "llm_conf": 0.32,
    "viirs_z": 2.02,
    "rain_percentile": 33,
    "area_ha": 473,
}
GATE = 0.72


def main() -> None:
    posterior = fuse_confidence(prior=PRIOR, evidence=EVIDENCE)
    print(f"prior={PRIOR}, evidence={EVIDENCE}")
    print(f"posterior = {posterior:.4f}")
    print(f"clears {GATE} gate: {posterior >= GATE}")


if __name__ == "__main__":
    main()
