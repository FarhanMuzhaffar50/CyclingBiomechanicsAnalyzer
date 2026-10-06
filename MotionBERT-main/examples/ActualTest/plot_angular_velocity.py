import numpy as np
import math
import matplotlib.pyplot as plt

def compute_knee_angles(filtered_file='examples/ActualTest/Height5/X3Dfiltered.npy',
                        hip_idx=1, knee_idx=2, ankle_idx=3):
    """
    Compute knee flexion-extension angles for each frame from filtered 3D poses.
    Uses Halpe-26 indexing: 1=RHip, 2=RKnee, 3=RAnkle by default.
    """
    # Load filtered poses: shape (n_frames, n_joints, 3)
    poses = np.load(filtered_file)
    n_frames = poses.shape[0]
    angles = np.zeros(n_frames)
    
    for i in range(n_frames):
        hip = poses[i, hip_idx]
        knee = poses[i, knee_idx]
        ankle = poses[i, ankle_idx]
        # Vectors thigh (hip->knee) and shin (ankle->knee)
        v_thigh = hip - knee
        v_shin = ankle - knee
        # Compute angle at the knee
        dot = np.dot(v_thigh, v_shin)
        norm_product = np.linalg.norm(v_thigh) * np.linalg.norm(v_shin)
        # Avoid numerical errors
        cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
        angles[i] = math.degrees(math.acos(cos_angle))
        
    return angles

def plot_angular_velocity(angles, fs=60.0, bins=50, threshold=10.0):
    """
    Plot and save histogram of frame-by-frame angular velocities.
    """
    # Compute angular velocity (degrees per frame)
    ang_vel = np.abs(np.diff(angles))
    
    plt.figure(figsize=(6, 4))
    plt.hist(ang_vel, bins=bins, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'{threshold}°/frame threshold')
    plt.xlabel('Angular Velocity (° per frame)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Knee Angular Velocities\n'
              '(Absolute frame-to-frame changes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure4_angular_velocity_hist.png', dpi=300)
    plt.show()
    print("Saved histogram as 'figure4_angular_velocity_hist.png'")

def main():
    # Step 1: Compute knee angles
    angles = compute_knee_angles(filtered_file='examples/ActualTest/Height5/X3Dfiltered.npy',
                                 hip_idx=1, knee_idx=2, ankle_idx=3)
    
    # Optional: save angles to file
    np.save('knee_angles.npy', angles)
    print("Saved knee_angles.npy")
    
    # Step 2: Plot angular velocity histogram
    plot_angular_velocity(angles, fs=60.0, bins=50, threshold=10.0)

if __name__ == '__main__':
    main()
