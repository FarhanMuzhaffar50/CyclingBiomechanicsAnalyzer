import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Configuration parameters
# -------------------------
RAW_POSES_FILE  = 'examples/ActualTest/Height5/X3D.npy'         # Unfiltered 3D poses from MotionBERT
FILT_POSES_FILE = 'examples/ActualTest/Height5/X3Dfiltered.npy'     # Filtered 3D poses
PELVIS_IDX      = 0                     # Joint index for pelvis
FRAME_COUNT     = 120                   # Analyze first 120 frames (~2 revolutions)
DT              = 1 / 60.0              # Time step in seconds (60 fps)

# -------------------------
# 1. Load raw and filtered pelvis trajectories (first 120 frames)
# -------------------------
raw_poses  = np.load(RAW_POSES_FILE)[:FRAME_COUNT]
filt_poses = np.load(FILT_POSES_FILE)[:FRAME_COUNT]

raw_y  = raw_poses[:, PELVIS_IDX, 1]    # raw pelvis vertical (meters)
filt_y = filt_poses[:, PELVIS_IDX, 1]   # filtered pelvis vertical

# -------------------------
# 2. Compute instantaneous jerk (3rd derivative) in m/s³
# -------------------------
raw_vel   = np.gradient(raw_y, DT)
raw_acc   = np.gradient(raw_vel, DT)
raw_jerk  = np.abs(np.gradient(raw_acc, DT))

filt_vel  = np.gradient(filt_y, DT)
filt_acc  = np.gradient(filt_vel, DT)
filt_jerk = np.abs(np.gradient(filt_acc, DT))

# -------------------------
# 3. Compute RMS jerk for raw vs. filtered
# -------------------------
rms_raw  = np.sqrt(np.mean(raw_jerk**2))
rms_filt = np.sqrt(np.mean(filt_jerk**2))

# -------------------------
# 4. Compute residual jitter (raw position - filtered position) in mm
# -------------------------
residual_mm = (raw_y - filt_y) * 1000  # convert meters to millimeters

# -------------------------
# 5. Plot residual jitter over time
# -------------------------
plt.figure(figsize=(6, 3))
plt.plot(residual_mm, marker='o', markersize=4, linestyle='-')
plt.axhline(0, color='gray', linewidth=1)
plt.xlabel('Frame')
plt.ylabel('Residual (Raw − Filtered) (mm)')
plt.title('Pelvis Position Residual Jitter')
plt.tight_layout()
plt.savefig('pelvis_residual_jitter.png', dpi=300)
plt.show()

# -------------------------
# 6. Plot RMS jerk comparison
# -------------------------
plt.figure(figsize=(5, 3))
plt.bar(['Raw', 'Filtered'], [rms_raw, rms_filt], color=['gray', 'C1'])
plt.ylabel('RMS Jerk (m/s³)')
plt.title('Pelvis RMS Jerk: Raw vs. Filtered')
plt.tight_layout()
plt.savefig('pelvis_rms_jerk_comparison.png', dpi=300)
plt.show()

# -------------------------
# 7. Print summary metrics
# -------------------------
print(f'Pelvis residual jitter (std): {np.std(residual_mm):.2f} mm')
print(f'Raw RMS jerk:      {rms_raw:.3f} m/s³')
print(f'Filtered RMS jerk: {rms_filt:.3f} m/s³')
