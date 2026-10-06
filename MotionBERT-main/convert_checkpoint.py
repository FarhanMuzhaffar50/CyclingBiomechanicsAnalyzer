import torch

# Load the CUDA checkpoint and map it to CPU
cuda_checkpoint_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
cpu_checkpoint_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch_cpu.bin'

print(f"Loading CUDA checkpoint from {cuda_checkpoint_path}...")
checkpoint = torch.load(cuda_checkpoint_path, map_location='cpu')

print("Saving the checkpoint in CPU format...")
torch.save(checkpoint, cpu_checkpoint_path)

print(f"CPU checkpoint saved at {cpu_checkpoint_path}")
