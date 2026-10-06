import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the saved 3D poses
poses = np.load("examples/biketest/X3D.npy")  # Shape: [N_frames, N_joints, 3]

# Plot the first frame
frame_idx = 0  # Change this to visualize other frames
joints_3d = poses[frame_idx]  # Shape: [N_joints, 3]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of joints
ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2])

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()