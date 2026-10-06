from pathlib import Path
import numpy as np
from .config import AnalysisConfig, JOINTS
from .filtering import filter_trajectory
from .cycles import detect_cycle_peaks, normalize_cycles
from .kinematics import knee_flexion, angular_velocity
from .symmetry import normalized_symmetry_index, cross_correlation
from .stability import derivatives, rms, cycle_consistency
from .reporting import write_report

def analyze_pose(pose_path, fps=60.0, output_dir=None, config=None):
    """Analyze a MotionBERT X3D.npy file; output paths are relative to output_dir."""
    path = Path(pose_path)
    c = config or AnalysisConfig(fps=fps)
    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 3 or raw.shape[1:] != (17, 3):
        raise ValueError(f"Expected MotionBERT H36M pose (frames,17,3), got {raw.shape}")
    out = Path(output_dir) if output_dir else path.parent / f"{path.stem}_analysis"
    out.mkdir(parents=True, exist_ok=True)
    pose = filter_trajectory(raw, c.fps, c.cutoff_hz, c.filter_order)
    np.save(out / "X3Dfiltered.npy", pose)
    left = pose[:, JOINTS["left_knee"], 1]
    right = pose[:, JOINTS["right_knee"], 1]
    peaks = detect_cycle_peaks(right, c.fps, c.peak_prominence_fraction, c.min_cycle_seconds)
    lc = normalize_cycles(left, peaks, c.normalized_points)
    rc = normalize_cycles(right, peaks, c.normalized_points)
    nsi = normalized_symmetry_index(lc, rc)
    per_cycle = [cross_correlation(l, r) for l,r in zip(lc,rc)]
    overall = cross_correlation(left,right)
    left_flex = knee_flexion(pose, 4,5,6)
    right_flex = knee_flexion(pose, 1,2,3)
    _,_,jerk = derivatives(pose[:,0,1], c.fps)
    duration = float((len(raw)-1)/c.fps)
    result = {
      "metadata": {"source": str(path), "frames": len(raw), "fps": c.fps, "duration_seconds": duration,
                   "pose_format": "MotionBERT H36M-17", "vertical_axis": "Y", "coordinate_units": "model output units (unvalidated physical scale)",
                   "filter": {"method":"zero-phase Butterworth", "cutoff_hz":c.cutoff_hz,"order":c.filter_order}},
      "cycles": {"count":len(peaks)-1,"boundary_frames":peaks.tolist(),"mean_duration_seconds":float(np.diff(peaks).mean()/c.fps),
                 "right_knee_y_consistency_model_units":cycle_consistency(rc)},
      "symmetry": {"mean_signed_nsi_percent":float(nsi.mean()),"mean_absolute_nsi_percent":float(np.abs(nsi).mean()),
                    "cross_correlation":overall["coefficient"],"phase_lag_frames":overall["lag_frames"],
                    "phase_lag_seconds":None if overall["lag_frames"] is None else overall["lag_frames"]/c.fps,
                    "per_cycle_cross_correlation":per_cycle},
      "kinematics": {"left_knee_flexion_degrees":{"min":float(np.nanmin(left_flex)),"max":float(np.nanmax(left_flex)),"mean":float(np.nanmean(left_flex))},
                      "right_knee_flexion_degrees":{"min":float(np.nanmin(right_flex)),"max":float(np.nanmax(right_flex)),"mean":float(np.nanmean(right_flex))},
                      "left_knee_angular_velocity_rms_deg_per_sec":rms(angular_velocity(left_flex,c.fps)),
                      "right_knee_angular_velocity_rms_deg_per_sec":rms(angular_velocity(right_flex,c.fps))},
      "stability": {"pelvis_vertical_rms_jerk_model_units_per_sec3":rms(jerk),
                    "pelvis_vertical_cycle_consistency_model_units":cycle_consistency(normalize_cycles(pose[:,0,1], peaks,c.normalized_points))}}
    charts = [("knee_trajectory.svg",[("Left",left),("Right",right)],"Knee vertical trajectory","Model units"),
              ("knee_flexion.svg",[("Left",left_flex),("Right",right_flex)],"Knee flexion","Degrees"),
              ("symmetry.svg",[("Mean NSI",nsi.mean(axis=0))],"Mean cycle symmetry index","Percent"),
              ("pelvis_jerk.svg",[("Jerk",jerk)],"Pelvis vertical jerk","Model units/s³")]
    return write_report(result,out,charts)
