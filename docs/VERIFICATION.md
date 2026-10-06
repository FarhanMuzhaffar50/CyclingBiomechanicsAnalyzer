# Verification record — 2026-10-06

This file separates completed checks from interpretation. Commands below were run in the supplied local workspace. The local test Python was `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` with temporary dependencies in `/private/tmp/cycling-deps`; that path is **not** a project prerequisite.

## Core mathematics and saved research trial

- `python -m pytest -q tests`: **9 passed** after final test additions (run again before release if source changes).
- `python -m cycling_analysis.cli analyze-pose MotionBERT-main/examples/ActualTest/Height1/X3D.npy --fps 59.94 --output sample/height1_analysis`: completed; 882 frames, 13 detected cycles, structured `results.json`, filtered NPY, four SVG charts.
- `python -m cycling_analysis.cli analyze <Height1 video> --skip-pose <Height1 X3D.npy> --output ...`: completed with measured `60000/1001` FPS and 13 cycles.
- JSON decoded and all four charts parsed as valid SVG XML.

## Fresh video inference

- An 180-frame, three-second clip from the preserved Height1 video was used for a smoke test. AlphaPose produced 180 Halpe-26 JSON records; MotionBERT produced a fresh 17-joint `X3D.npy`; the custom analyzer wrote JSON and four SVG charts.
- `python -m cycling_analysis.cli analyze runtime/inference_smoke/short.mp4 --output runtime/inference_smoke/one_command`: completed with **one detected cycle**. This confirms orchestration, not metric reliability on short clips.
- FastAPI `POST /api/analyses` with the same short clip returned a job that reached `completed`; `/results` returned one cycle and four chart names.
- MotionBERT rendering aborted in the local environment before it could save the NPY. The targeted `--no-render` option was added to `infer_wild.py`; the pipeline uses it and the fresh run completed. The neural model and weights were not changed.
- AlphaPose attempted to download ImageNet weights during model construction even though the full AlphaPose checkpoint was supplied. The local `ALPHAPOSE_SKIP_IMAGENET_INIT=1` switch avoids this unnecessary download in the wrapper. This local AlphaPose code is excluded from public Git under its supplied license.

## API, Docker, frontend

- FastAPI TestClient `POST /api/poses` with the public Height1 NPY returned a completed job, `results.json` with 13 cycles, and an SVG chart response.
- `docker compose build`: succeeded. Port 8000 was occupied on this machine, so a container was run on host port 8001. Its `/health` and `/api/poses` routes succeeded; the pose job completed with 13 cycles and four charts.
- The React/Vite production build succeeded in a Node 22 container.
- The dashboard was opened in a browser, switched to “Existing 3D pose,” and uploaded the public sample to the local Docker API. The visible result showed 13 cycles, metric cards, and four inline charts. This was a local browser check, not a hosted deployment.

## Boundaries

The Docker image provides the API and custom analysis layer. It omits AlphaPose/MotionBERT runtime packages and model weights. Full video inference was verified natively with the supplied local third-party checkouts; it was not run inside Docker. Physical pose scale and clinical validity were not measured. The five original research folders were preserved locally, while videos and generated artifacts are excluded from public Git.
