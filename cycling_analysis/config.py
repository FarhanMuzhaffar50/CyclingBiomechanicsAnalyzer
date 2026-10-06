from dataclasses import dataclass

@dataclass(frozen=True)
class AnalysisConfig:
    fps: float = 60.0
    cutoff_hz: float = 6.0
    filter_order: int = 2
    peak_prominence_fraction: float = 0.2
    min_cycle_seconds: float = 0.45
    normalized_points: int = 100

# MotionBERT infer_wild.py converts AlphaPose Halpe-26 into H36M-17.
JOINTS = {"pelvis": 0, "right_hip": 1, "right_knee": 2, "right_ankle": 3,
          "left_hip": 4, "left_knee": 5, "left_ankle": 6}
