# Backend

Run from the repository root with `uvicorn backend.main:app --reload`.

`POST /api/analyses` accepts a multipart `video` upload and returns a job id.
Poll `GET /api/analyses/{analysis_id}` until `completed`, then fetch
`GET /api/analyses/{analysis_id}/results`. State is intentionally in memory for
the MVP; completed artifacts remain in `runtime/outputs`.
