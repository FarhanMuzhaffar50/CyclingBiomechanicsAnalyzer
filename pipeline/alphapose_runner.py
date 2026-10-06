import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_alphapose(video_path, output_dir, python_executable=None):
    root = Path(os.getenv("ALPHAPOSE_ROOT", ROOT / "AlphaPose-master")).resolve()
    config = root / "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
    weights = Path(os.getenv("ALPHAPOSE_CHECKPOINT", root / "pretrained_models/halpe26_fast_res50_256x192.pth"))
    for needed in (root / "scripts/demo_inference.py", config, weights):
        if not needed.is_file(): raise FileNotFoundError(f"AlphaPose dependency missing: {needed}")
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    command = [python_executable or os.getenv("ALPHAPOSE_PYTHON", sys.executable),
               "scripts/demo_inference.py", "--cfg", str(config), "--checkpoint", str(weights),
               "--video", str(Path(video_path).resolve()), "--outdir", str(output_dir), "--sp", "--gpus", "-1"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), environment.get("PYTHONPATH")]))
    environment["ALPHAPOSE_SKIP_IMAGENET_INIT"] = "1"
    completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    (output_dir / "alphapose.log").write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr)
    if completed.returncode: raise RuntimeError(f"AlphaPose failed (exit {completed.returncode}); see {output_dir / 'alphapose.log'}: {completed.stderr[-800:]}")
    result = output_dir / "alphapose-results.json"
    if not result.is_file(): raise RuntimeError(f"AlphaPose produced no keypoint JSON; see {output_dir / 'alphapose.log'}")
    return result
