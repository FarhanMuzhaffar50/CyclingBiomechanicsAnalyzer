import numpy as np

def joint_angle(hip, knee, ankle):
    """Included angle at knee in degrees; straight leg is 180 degrees."""
    a = np.asarray(hip) - np.asarray(knee)
    b = np.asarray(ankle) - np.asarray(knee)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.divide(np.sum(a*b, axis=-1), denom,
                       out=np.full_like(denom, np.nan, dtype=float), where=denom > 0)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

def knee_flexion(pose, hip_idx, knee_idx, ankle_idx):
    return 180.0 - joint_angle(pose[:, hip_idx], pose[:, knee_idx], pose[:, ankle_idx])

def angular_velocity(angles_degrees, fps):
    return np.gradient(np.asarray(angles_degrees, dtype=float), 1.0/fps)
