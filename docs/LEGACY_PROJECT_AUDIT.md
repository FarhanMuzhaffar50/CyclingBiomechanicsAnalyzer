# Legacy project audit

## Provenance and layout

`AlphaPose-master/` and the neural-network code in `MotionBERT-main/` are third-party sources. The custom university research scripts live chiefly in `MotionBERT-main/examples/ActualTest/`. Both supplied directories are extracted snapshots without Git history, so historical authorship of individual upstream edits cannot be proved from commits. The five `Height1`–`Height5` directories are preserved unchanged.

## Data flow

1. `AlphaPose-master/scripts/demo_inference.py --video ...` writes `alphapose-results.json`. The saved records contain `image_id`, `category_id`, `keypoints`, `score`, `box`, `idx`; `keypoints` has 78 values (26 Halpe joints × x/y/confidence). Each saved trial has 882–908 frame records.
2. `MotionBERT-main/lib/data/dataset_wild.py` converts Halpe-26 to Human3.6M 17 joints (`halpe2h36m`). `MotionBERT-main/infer_wild.py --vid_path --json_path --out_path` writes `(frames,17,3)` `X3D.npy` and `X3D.mp4`.
3. ActualTest scripts filter the pose and analyze knee/pelvis trajectories. Original videos are approximately 59.94 FPS; scripts typically use 60 FPS. The new CLI accepts measured/explicit FPS and records it.

H36M indices in the MotionBERT output: pelvis 0; right hip/knee/ankle 1/2/3; left hip/knee/ankle 4/5/6. Some legacy comments incorrectly refer to this 17-joint output as Halpe-26. `knee_consistency_test.py` uses index 10 as a knee, inconsistent with the conversion and thus suspect.

## Custom script inventory

| Script | Input | Main output or issue |
| --- | --- | --- |
| `filter3D.py` | Height5 `X3D.npy` | 2nd-order 6 Hz Butterworth `X3Dfiltered.npy` |
| `compute_nsi.py` | Height5 filtered pose | Knee cycle NSI; **writes CSV into Height1** by mistake |
| `compute_ccc.py` | Height5 filtered pose | Per-cycle and overall normalized cross-correlation, CSV/PNG |
| `cycle_consistency.py` | Height5 filtered pose | Mid-cycle knee Y consistency/plot |
| `knee_consistency_test.py` | Height5 filtered pose | Similar consistency calculation with incorrect knee index 10 |
| `plot_angular_velocity.py` | Height5 filtered pose | Knee included angle, angular velocity histogram |
| `jerkjerk.py` | Height5 filtered pose | Pelvis Y derivatives, RMS jerk and histogram |
| `noise_analysis.py` | Height5 raw/filtered poses | Residual jitter and jerk comparison plots |
| `plot_knee_filter.py` | Height5 raw pose | Raw/filtered knee plot; duplicates filter settings |
| `sanitycheck.py` | Height5 filtered pose | Basic coordinate range printout |

Most scripts hard-code `examples/ActualTest/Height5/...` and write output to the process working directory. `compute_nsi.py`/`compute_ccc.py` duplicate cycle segmentation; the two consistency scripts duplicate peak and midpoint logic. These original files remain for research provenance; the root `cycling_analysis/` package is the reusable implementation.

## Existing compatibility modifications preserved

- `MotionBERT-main/infer_wild.py` conditionally uses CUDA for model/input, loads checkpoints with CPU-safe `map_location`, strips `module.` keys, and configures `DataLoader` with `num_workers=0`, `prefetch_factor=None`, `persistent_workers=False`.
- `MotionBERT-main/lib/utils/learning.py` includes checkpoint key-prefix handling.
- `AlphaPose-master/scripts/demo_inference.py` selects CPU when CUDA is unavailable and supports `--gpus -1`; it has device-aware checkpoint loading and a NumPy `np.float` shim. Its detection/writer paths can use the `--sp` single-process mode.
- These are observed code differences supporting the user's reported prior macOS/CPU adaptation. Without upstream commit history, this audit does not attribute each line to a specific author. Some optional AlphaPose tracking code still assumes CUDA; the new wrapper does not enable it.

No broad device refactor was made. The new wrapper passes `--sp --gpus -1` for AlphaPose and preserves MotionBERT's existing conditional device logic. During this upgrade, one targeted line in `AlphaPose-master/alphapose/models/fastpose.py` gained an environment switch to skip an unnecessary ImageNet initialization download when the complete AlphaPose checkpoint will immediately be loaded. This local AlphaPose checkout is excluded from Git under its redistribution-restricted license. `MotionBERT-main/infer_wild.py` gained `--no-render` so CPU inference can save `X3D.npy` without invoking its renderer, which aborted in the test environment. Its renderer import is now deferred until rendering is requested. Neither change alters a neural-network architecture or trained weights.

## Docker and reproducibility

`MotionBERT-main/Dockerfile` points to nonexistent `scripts/demo_inference.py` and is not a working application image. The root Dockerfile packages the API and custom analysis only. It does not contain pretrained weights or legacy ML dependencies, so full upload inference inside that image is not verified or claimed.

The root `.gitignore` excludes weights, environments, caches, runtime outputs, the local AlphaPose source, and private/generated Height1–5 research media. Saved ActualTest experimental videos, JSON, arrays, and figures remain on disk, unchanged. A 176 KB Height1 pose sample and research scripts copied to `legacy/ActualTest/` make the public Git project demonstrable without publishing the five videos.
