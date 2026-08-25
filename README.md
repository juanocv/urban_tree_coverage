> 🇧🇷 **Português:** [Leia esta página em português](docs/pt-br/README.md)

# Urban Tree Coverage

Urban Tree Coverage estimates the **visible tree coverage** of urban streets from
Google Street View imagery, using semantic and panoptic segmentation. The
production package lives in `urban_canopy/`; third-party model checkouts
(OneFormer via HuggingFace, Detectron2, DeepLab) stay outside the package
boundary.

The primary indicator is continuous and per image:

```
tree_coverage_ratio = tree pixels / all image pixels      (in [0, 1])
tree_coverage_pct   = 100 * tree_coverage_ratio
```

A wider `vegetation_coverage_ratio` is reported separately when the model can
distinguish it. Tree, grass and shrub classes are **never merged silently** —
the mapping from model classes to these groups is explicit, inspectable and
overridable (`urban_canopy/models/taxonomy.py`). No qualitative bands ("low /
medium / high greenery") are produced, and the project measures area, never
counts. The [FAQ](docs/faq.md#the-indicator) has the reasoning for both.

## What it does

1. **Acquisition** — Street View (cached, with panorama id and capture date
   recorded) or local images.
2. **View strategy** — single view, or a deterministic multi-view plan, chosen
   by configuration and never by the segmentation output.
3. **Segmentation** — four backends behind one common contract.
4. **Refinement** — conservative, optional mask cleanup with a growth guard that
   caps how much any setting can inflate the mask.
5. **Indicators** — coverage ratios per image, with quality flags and capture
   provenance.
6. **Aggregation** — mean / median / IQR / p25 / p75 across the views of a
   location.
7. **Evaluation** — two independent levels against manual COCO ground truth:
   pixels (IoU, Dice/F1, precision, recall) and the coverage indicator itself
   (MAE, RMSE, bias in percentage points).
8. **Audit artifacts** — per view: RGB, raw and refined masks, overlays, metrics
   JSON; plus CSV/JSON exports per run.

### What the backends can claim

| Backend | Pretraining | Tree class |
|---|---|---|
| OneFormer | ADE20K-150 | `tree` (stuff) + `palm` |
| Mask2Former | ADE20K / COCO / Cityscapes | depends on the checkpoint |
| Detectron2 panoptic FPN | COCO-panoptic 133 | `tree-merged` (stuff) |
| DeepLab V3+ | Cityscapes-19 | none (`vegetation` merges trees+bushes) |

A backend whose class space cannot express a tree reports **no tree ratio**,
rather than relabelling its vegetation number. See
[which backend to use](docs/faq.md#choosing-a-backend) for the measured
comparison.

## Repository layout

```text
urban_canopy/              Python package used by the CLI and API
urban_canopy/core/         Pipeline orchestration, config, results, view plans
urban_canopy/io/           Street View, image and geospatial I/O, artifacts
urban_canopy/models/       Backend adapters, taxonomy, factory
urban_canopy/processing/   Coverage, refinement, multi-view aggregation
urban_canopy/evaluation/   COCO ground truth, metrics, prediction interchange
urban_canopy/tests/        Offline, CPU-only unit tests
docs/                      Architecture, annotation protocol, evaluation, FAQ
notebooks/                 Two worked examples, runnable without an API key
samples/images/            Small curated image set for trying the pipeline
samples/annotations/       Manual COCO ground truth for those images
```

## Trying it without an API key

`samples/images/` holds seven curated frames — including a no-trees negative
case and a four-heading sweep of one location — all manually annotated, spanning
0% to 29% labelled tree coverage.

```bash
python -m pip install -e ".[ml,notebooks]"
jupyter lab notebooks/
```

- [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) — one image
  end to end: the indicator, the tree/vegetation split, raw vs refined masks,
  and the refinement growth guard.
- [`02_multiview_and_evaluation.ipynb`](notebooks/02_multiview_and_evaluation.ipynb)
  — multi-view aggregation and the two evaluation levels.

## Setup

Needs Python 3.10 or newer on Windows or Linux (CI tests 3.10 and 3.13).

**Linux** — Debian/Ubuntu do not ship `venv` with the interpreter, and OpenCV
links against libGL:

```bash
sudo apt install python3-venv libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Or use the helper:

```bash
./scripts/setup-dev.sh --api --ml                                          # Linux
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1 -WithApi -WithMl  # Windows
```

The base install is enough for the unit tests and package imports: adapter
modules keep Torch, Transformers, Pillow, Torchvision and Detectron2 imports at
construction time. Running real segmentation needs the ML layer:

```bash
python -m pip install -e ".[ml]"
```

PyTorch itself is left to you: install the CPU or CUDA build matching your
machine from [pytorch.org](https://pytorch.org/get-started/locally/).

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` before Street View calls.
Importing modules and running unit tests never need the key.

Backend-specific setup — DeepLab's checkpoint and `network` package, Detectron2's
compile step and its `pkg_resources` failure, download sizes, CUDA
troubleshooting — is in [`docs/faq.md`](docs/faq.md#installation) and
[`docs/reproducibility.md`](docs/reproducibility.md#backend-specific-setup).

## Running

The editable install exposes a `tree-ai` console script
(`python -m urban_canopy.cli.main` is the same entry point).

```bash
# Local image, single view
tree-ai --image street.jpg --single-view --seg oneformer --device cpu

# Coordinates, multi-view (0/90/180/270 around a reference heading by default)
tree-ai --lat -23.678479 --lon -46.559621 --multi-view --seg oneformer

# Address, multi-view with a known street bearing
tree-ai "Av. Paulista 1578, Sao Paulo" --multi-view --reference-heading 45 --offsets 90,270

# Everything an evaluation or audit needs, in this run's directory
tree-ai --image street.jpg --save-artifacts
```

Evaluate against Roboflow COCO ground truth, and check an export before
labelling more:

```bash
tree-ai evaluate --predictions artifacts_out/<run>/predictions.json \
                 --annotations annotations.json --report-json report.json

tree-ai validate-dataset --annotations annotations.json
```

Flags worth knowing up front: `--no-refine` (raw mask baseline),
`--allow-vegetation-proxy` (let Cityscapes `vegetation` stand in for trees, with
`tree_source="vegetation_proxy"` recorded), `--view-mode` (deterministic
multi-view plans) and `--min-successful-views` (abort a run that produced too
little imagery). `tree-ai --help` lists the rest; the
[FAQ](docs/faq.md#running-and-outputs) explains the ones with consequences.

### Where results go

Each invocation gets its own directory under `--outdir` (default
`artifacts_out/`), named after the timestamp and the backend:

```text
artifacts_out/
  20260818-104512_oneformer/
    run.json            manifest, aggregate, every view
    views.csv           one row per view
    predictions.json    for `tree-ai evaluate`
    views/
      000_street/       rgb.png  mask_raw.png  mask_refined.png
                        overlay_tree.png  metrics.json
      001_...           further views, in acquisition order
```

Runs accumulate instead of overwriting, and nothing is written unless an output
flag asks for it.

## Web API

```bash
python -m pip install -e ".[api,ml]"
uvicorn urban_canopy.webapi:app --host 127.0.0.1 --port 8000
```

The API reads the same backend settings as the CLI from `.env`. `POST
/analyse/single` and `POST /analyse/multi` return the coverage metrics plus
backend/checkpoint/taxonomy provenance. `GET /ping` is a liveness probe; `GET
/ready` confirms model startup and returns the same provenance, including a
SHA-256 when weights are local. Interactive docs are at `/docs`. Dataset
evaluation stays in the CLI.

Both analysis endpoints accept a `backend` field, so a caller picks the
segmentation backend per request rather than living with whatever
`UC_SEG_BACKEND` was set to at startup. Omitting it uses that default. Backends
load on first use and stay resident, so nothing is paid for a backend nobody
asks for; `UC_API_BACKENDS` narrows the offer on a machine short on VRAM. `GET
/ready` lists what the instance offers, each entry marked `loaded`, `available`
or `unavailable` with the reason, which is enough for a client to grey out what
it cannot use.

This is what makes `allow_vegetation_proxy` reachable in practice: it changes
nothing on a backend that already has a tree class, and only matters on
`deeplab`, whose Cityscapes class space merges trees into `vegetation`. Selecting
that backend and leaving the proxy off reports no tree coverage at all, which is
the honest answer; turning it on reports the vegetation number with
`tree_source="vegetation_proxy"` and a quality flag saying so.

Both analysis endpoints accept `return_overlays` (default off), which adds
base64 PNGs of the RGB frame, the tree overlay and the refined mask — on
`/single` under a top-level `overlays` key, on `/multi` under `overlays` on each
view, so headings can be compared side by side. They dominate the response size:
a 640x640 frame is roughly a megabyte of PNG and each view carries three, so
`/multi` refuses a plan larger than `UC_API_MAX_OVERLAY_VIEWS` (default 8)
rather than serving a response nobody asked to receive.

`UC_API_TOKENS` (comma-separated) turns on bearer authentication: `/ready` and
both `/analyse` endpoints then require `Authorization: Bearer <token>`, while
`GET /ping` stays open for liveness probes. Left empty the API is
unauthenticated, which is fine on localhost and reckless anywhere else — it calls
a paid Google API on every request, so an open instance spends your quota for
whoever finds it. Startup logs which mode is active.

Nothing here rate-limits: the concurrency semaphore bounds how many inferences
run at once, not how many a token holder may run. Set a budget cap on the Google
side too.

A static web console for this API lives in
[urban_canopy-web](https://github.com/juanocv/urban_canopy-web). For additional info
on how to expose the API to the internet, check the _tunneling_ documentation in that 
repository's README.

## Ground truth and evaluation

Labelling happens in Roboflow, exported as **COCO Instance Segmentation**, one
polygon/mask per tree. The pixel-level ground truth is their union.

Every prediction file embeds a manifest (package versions, model name, device,
taxonomy, refinement config, RNG seed and deterministic-runtime flags), so any
reported number can be traced to the run that produced it.

- [`docs/faq.md`](docs/faq.md) — installation trouble, backend choice, and why
  the design decisions are what they are
- [`docs/annotation_protocol.md`](docs/annotation_protocol.md) — what counts as
  a tree, crowns vs trunks, occlusions, partial trees, minimum visibility
- [`docs/evaluation.md`](docs/evaluation.md) — metrics, matching rules,
  empty-case conventions, validation/test split policy
- [`docs/architecture.md`](docs/architecture.md) — module contracts and the
  mapping from `sidewalk_analysis` components
- [`docs/reproducibility.md`](docs/reproducibility.md) — environment capture and
  backend-specific setup
- [`docs/detectron2-windows.md`](docs/detectron2-windows.md) — Detectron2 on
  Windows, and the WSL question

## Quality checks

```bash
./scripts/check.sh                                             # Linux
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1    # Windows
```

Or individually:

```bash
python -m pytest --cov=urban_canopy --cov-report=term-missing \
  --cov-report=json:coverage.json --cov-fail-under=80
python -m ruff check urban_canopy
python -m black --check urban_canopy
python -m pyright
python scripts/check_coverage.py coverage.json --fail-under 60
```

The default suite is offline and CPU-only; `pytest -m gpu` and `pytest -m network`
run the excluded checks. See the [FAQ](docs/faq.md#development) for what each
gate enforces and why.

## Generative AI Usage Transparency

Generative AI tools were used to support the conception and development of this project, 
including activities such as discussing implementation alternatives, reviewing and 
organizing code, developing tests, and reviewing documentation.

Suggestions and content produced with the assistance of these tools were reviewed, adapted, 
and validated by the author. Project decisions, the final implementation, experiments, 
interpretation of results, and responsibility for the contents of this repository remain 
entirely with the author.

This use of generative AI as a development support tool is distinct from the segmentation 
models employed by Urban Tree Coverage as part of its analysis pipeline.

## Citation

```bibtex
@misc{urban_tree_coverage_2026,
  author = {Juan Oliveira de Carvalho},
  title = {Urban Tree Coverage: Visible Street-Level Tree Coverage from Street View Imagery Using Semantic Segmentation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/juanocv/urban_tree_coverage}
}
```
