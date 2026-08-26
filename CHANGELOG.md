# Changelog

Notable changes to Urban Tree Coverage. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with the pre-1.0
convention that the minor number carries both features and breaking changes.

Only the version in `pyproject.toml` is authoritative. The web API's `/docs`
page and the reproducibility manifest in every run both read it from the
installed package metadata, so an editable checkout needs `pip install -e .`
re-run after a bump for them to agree.

## [0.2.0] — 2026-08-26

Web API only. The CLI, pipeline, evaluation and taxonomy are unchanged from
0.1.0, and no response key was removed or renamed, so a 0.1.0 client keeps
working against a 0.2.0 instance.

### Added

- **Bearer authentication.** `UC_API_TOKENS` (comma-separated) guards `/ready`
  and both `/analyse` endpoints; `GET /ping` stays open so liveness probes and
  tunnel health checks keep working. Off by default, and startup logs which
  mode is active. Tokens are compared in constant time, and accepting a list
  means one can be revoked without disturbing the others.
- **Per-request backend selection.** Both analysis endpoints accept `backend`
  (`oneformer`, `mask2former`, `detectron2`, `deeplab`); omitting it uses
  `UC_SEG_BACKEND`. Backends load on first use and stay resident, so nothing is
  paid for a backend nobody asks for.
- **`UC_API_BACKENDS`** narrows which backends an instance will serve, for a
  machine short on VRAM. The default backend must be one of them.
- **Backend listing on `/ready`.** New `default_backend` and `backends[]`
  fields, each entry marked `loaded`, `available` or `unavailable` with the
  reason — enough for a client to disable what it cannot use. The existing
  `status` and `backend` fields are unchanged.
- **Overlays on `/analyse/multi`.** `return_overlays` adds an `overlays` object
  to every view, so headings can be compared side by side. Plans larger than
  `UC_API_MAX_OVERLAY_VIEWS` (default 8) are refused with 422 rather than
  served: a 640x640 frame is roughly a megabyte of PNG and each view carries
  three.

Backend selection is what makes `allow_vegetation_proxy` reachable in practice.
It changes nothing on a backend that already has a tree class, and matters only
on `deeplab`, whose Cityscapes class space merges trees into `vegetation`.
Selecting that backend with the proxy off reports no tree coverage at all, which
is the honest answer; turning it on reports the vegetation number with
`tree_source="vegetation_proxy"` and a quality flag saying so.

### Fixed

- **A loaded backend is no longer vetoed by the availability probe.** The probe
  is a cheap pre-flight check (an import spec, a file on disk) meant for a
  backend that would have to be built now. Consulting it for one already built
  let an import check overrule a working model, which made every analysis answer
  503 in an environment without the ML extras installed — including CI, and any
  instance whose segmenter was injected rather than imported.
- **`UC_API_*` settings are read from `.env`.** They were previously read only
  from the process environment, even though `.env.example` has shipped these
  keys since 0.1.0, so a value edited there was silently ignored. Real
  environment variables still take precedence. The file is parsed without being
  loaded into `os.environ`, so importing the module has no effect on anything
  else sharing the interpreter.
- **Test isolation from the working directory.** The suite no longer inherits
  whatever `.env` sits in the checkout: a developer's own `UC_TRUST_CHECKPOINT`
  used to fail the backend defaults test on their machine and nowhere else.

### Changed

- The API's advertised version now comes from the installed package metadata
  instead of a literal, removing the second place a release had to be
  remembered.

### Upgrading

- **Check your `.env` before restarting.** `UC_API_CORS_ORIGINS`,
  `UC_API_MAX_CONCURRENCY` and `UC_API_QUEUE_TIMEOUT_S` now take effect from
  that file. A stale value that was ignored under 0.1.0 will start applying —
  a restrictive `UC_API_CORS_ORIGINS` is the one most likely to surprise.
- **`/ready` requires a token once `UC_API_TOKENS` is set.** A client that
  polled it unauthenticated needs to send the token. Instances that leave
  authentication off are unaffected.
- **Internal:** `PipelineRegistry` now takes a `SegmenterRegistry` rather than a
  segmenter, and `get()` returns `(pipeline, provenance)`. `webapi` is not
  exported from `urban_canopy/__init__.py`, so library and CLI users are
  unaffected.

## [0.1.0] — 2026-08-21

First tagged release.

- Visible tree coverage from Google Street View imagery or local images, as a
  continuous per-image ratio, with no qualitative bands and no tree counting.
- Four segmentation backends behind one contract: OneFormer, Mask2Former,
  Detectron2 panoptic FPN and DeepLab V3+. A backend whose class space cannot
  express a tree reports no tree ratio rather than relabelling its vegetation
  number.
- Deterministic multi-view planning — fixed, offsets or equiangular — chosen by
  configuration and never by the segmentation output.
- Conservative mask refinement with a growth guard capping how much any setting
  can inflate the mask.
- Robust aggregation across the views of a location: mean, median, IQR, p25,
  p75, and the counts needed to judge them.
- Two independent evaluation levels against manual COCO ground truth: pixels
  (IoU, Dice/F1, precision, recall) and the coverage indicator itself (MAE,
  RMSE, bias in percentage points).
- Audit artifacts per view, plus a run manifest embedding package versions,
  device, taxonomy, refinement config and RNG state.
- `tree-ai` CLI and a FastAPI web API (`/ping`, `/ready`, `/analyse/single`,
  `/analyse/multi`).

[Unreleased]: https://github.com/juanocv/urban_canopy/compare/0.2.0...HEAD
[0.2.0]: https://github.com/juanocv/urban_canopy/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/juanocv/urban_canopy/releases/tag/0.1.0
