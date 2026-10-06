import numpy as np

# Load the file once
poses = np.load('examples/ActualTest/Height5/X3Dfiltered.npy')  # shape (n_frames, n_joints, 3)

# Define your knee index
KNEE_IDX = 2

# Extract knee Y (meters) and convert to mm
knee_y    = poses[:, KNEE_IDX, 1]
knee_mm   = knee_y * 1000

# Print summary stats
print("POSES file          :", 'X3Dfiltered.npy')
print("Joint count         :", poses.shape[1])
print("Knee index          :", KNEE_IDX)
print("knee_y (m) range    : {:.3f} to {:.3f}".format(knee_y.min(), knee_y.max()))
print("knee_mm (mm) range  : {:.1f} to {:.1f}".format(knee_mm.min(), knee_mm.max()))
print("Sample knee_mm      :", knee_mm[:10])