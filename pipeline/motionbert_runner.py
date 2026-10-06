import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_motionbert(video_path, pose2d_json, output_dir, python_executable=None):
    root = Path(os.getenv("MOTIONBERT_ROOT", ROOT / "MotionBERT-main")).resolve()
    weights = Path(os.getenv("MOTIONBERT_CHECKPOINT", root / "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"))
    for needed in (root / "infer_wild.py", weights, Path(pose2d_json)):
        if not needed.is_file(): raise FileNotFoundError(f"MotionBERT dependency missing: {needed}")
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    command = [python_executable or os.getenv("MOTIONBERT_PYTHON", sys.executable), "infer_wild.py",
               "--vid_path", str(Path(video_path).resolve()), "--json_path", str(Path(pose2d_json).resolve()),
               "--out_path", str(output_dir), "--evaluate", str(weights), "--no-render"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (output_dir / "motionbert.log").write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr)
    if completed.returncode: raise RuntimeError(f"MotionBERT failed (exit {completed.returncode}); see {output_dir / 'motionbert.log'}: {completed.stderr[-800:]}")
    result = output_dir / "X3D.npy"
    if not result.is_file(): raise RuntimeError(f"MotionBERT produced no X3D.npy; see {output_dir / 'motionbert.log'}")
    return result
