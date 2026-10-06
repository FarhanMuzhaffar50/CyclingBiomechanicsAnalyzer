import numpy as np
import matplotlib.pyplot as plt

# Configuration
POSES_FILE = 'examples/ActualTest/Height5/X3Dfiltered.npy'  # Filtered 3D pose file (in meters)
PELVIS_IDX = 0                  # Pelvis joint index
FPS = 60.0                      # Frames per second of the video

# Load filtered 3D poses
poses = np.load(POSES_FILE)  # shape: (n_frames, n_joints, 3)

# Extract pelvis vertical coordinate (Y-axis) in meters
pelvis_y = poses[:, PELVIS_IDX, 1]

# Time step
dt = 1.0 / FPS

# Compute instantaneous jerk (third derivative) in m/s³
vel  = np.gradient(pelvis_y, dt)
acc  = np.gradient(vel, dt)
jerk = np.gradient(acc, dt)

# Compute RMS jerk
rms_jerk = np.sqrt(np.mean(jerk**2))

# Plot histogram of instantaneous jerk
plt.figure(figsize=(6, 4))
plt.hist(jerk, bins=50, edgecolor='black')
plt.axvline(rms_jerk, color='red', linestyle='--',
            label=f'RMS jerk = {rms_jerk:.2f} m/s³')
plt.xlabel('Instantaneous Jerk (m/s³)')
plt.ylabel('Frequency')
plt.title('Distribution of Pelvis Instantaneous Jerk')
plt.legend()
plt.tight_layout()
plt.savefig('pelvis_jerk_hist_mps3.png', dpi=300)
plt.show()

# Print the RMS jerk value
print(f'Computed RMS jerk: {rms_jerk:.2f} m/s³')