import json
import subprocess
from pathlib import Path
from cycling_analysis.pipeline import analyze_pose
from .alphapose_runner import run_alphapose
from .motionbert_runner import run_motionbert

def video_fps(video_path):
    path = Path(video_path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Video file is missing or empty")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "json", str(path)], capture_output=True, text=True)
    if probe.returncode: raise ValueError(f"Invalid video: {probe.stderr[-400:]}")
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams: raise ValueError("Video has no video stream")
    from fractions import Fraction
    fps = float(Fraction(streams[0]["r_frame_rate"]))
    if fps <= 0: raise ValueError("Video FPS is invalid")
    return fps

def analyze_video(video_path, output_dir, status_callback=None):
    """Run bundled AlphaPose and MotionBERT, then the custom analysis package."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    fps = video_fps(video_path)
    def stage(name):
        if status_callback: status_callback(name)
    stage("extracting_2d_pose")
    json_path = run_alphapose(video_path, output / "alphapose")
    stage("estimating_3d_pose")
    pose_path = run_motionbert(video_path, json_path, output / "motionbert")
    stage("analyzing")
    result = analyze_pose(pose_path, fps=fps, output_dir=output)
    stage("completed")
    return result
