"""
Urban Canopy Web API.

Run it with::

    python -m pip install -e ".[api,ml]"
    uvicorn urban_canopy.webapi:app --host 127.0.0.1 --port 8000

Endpoints
---------
* ``POST /analyse/single`` -- one Street View frame -> coverage metrics
* ``POST /analyse/multi``  -- several headings -> per-view metrics + aggregate
* ``GET /ping``            -- liveness probe

Dataset evaluation is CLI-only (``tree-ai evaluate``): it reads local files and
produces large reports, neither of which belongs in a request/response cycle.

Overlays
--------
Both analysis endpoints can return base64 PNGs (``return_overlays``), off by
default. They are the dominant cost of a response: a 640x640 street frame is
roughly a megabyte of PNG, and a view carries three of them. On ``/analyse/multi``
that multiplies by the number of headings, so a plan larger than
``UC_API_MAX_OVERLAY_VIEWS`` (default 8) is refused rather than served as a
response no client asked to receive.

Concurrency
-----------
The endpoints are synchronous, so Starlette runs them in a worker thread pool
and several requests can overlap. The segmentation model behind them is neither
thread-safe nor cheap in VRAM, so model work is serialised through a semaphore
sized by ``UC_API_MAX_CONCURRENCY`` (default 1).

Authentication
--------------
Off by default, which keeps a localhost instance as convenient as the CLI.
Setting ``UC_API_TOKENS`` to a comma-separated list of secrets turns on bearer
authentication for everything except ``GET /ping``: requests must carry
``Authorization: Bearer <token>``. Startup logs which mode is active, loudly,
because the difference decides whether strangers can spend the Google quota this
service is billed for.

Tokens are compared in constant time, but they are still bearer secrets: an
instance reachable from the internet wants TLS in front of it, so the token is
not readable on the wire. Nothing here does rate limiting -- the inference
semaphore bounds concurrency, not spend.
"""

from __future__ import annotations

import importlib.util
import os
import secrets
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any, Literal

from dotenv import dotenv_values
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from urban_canopy.log import configure_logging, get_logger
from urban_canopy.models.backend_settings import (
    BackendName,
    BackendSettings,
    backend_provenance,
    build_segmenter_from_settings,
)
from urban_canopy.models.factory import BACKEND_CLASS_SPACE, BACKENDS
from urban_canopy.validation import (
    validate_image_size,
    validate_latitude,
    validate_longitude,
)

configure_logging(force=False)
logger = get_logger(__name__)

# BackendSettings reads .env for its own fields via pydantic-settings; every
# UC_API_* setting in this module is read separately, with a plain os.getenv
# fallback to values parsed from the same file. dotenv_values() only returns a
# dict -- unlike load_dotenv(), it never writes into os.environ, so importing
# this module (for testing, type-checking, or anything short of actually
# running the server) has no effect on any other code sharing the interpreter.
_DOTENV_VALUES = dotenv_values(".env")


def _env(key: str, default: str) -> str:
    """A real environment variable always wins over the .env fallback."""
    return os.environ.get(key) or _DOTENV_VALUES.get(key) or default


# Serialises model work; see the module docstring.
MAX_CONCURRENCY = max(1, int(_env("UC_API_MAX_CONCURRENCY", "1")))
_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)

# Ceiling on how many views may carry overlays in one multi-view response.
MAX_OVERLAY_VIEWS = max(1, int(_env("UC_API_MAX_OVERLAY_VIEWS", "8")))


def _enabled_backends() -> tuple[str, ...]:
    """
    Backends this instance will serve, in the order they are offered.

    Every backend by default. Each one a request actually uses stays resident
    for the life of the process, so an instance short on VRAM can narrow the
    list rather than discover the limit under load.
    """
    raw = _env("UC_API_BACKENDS", "").strip()
    if not raw:
        return tuple(BACKENDS)
    chosen = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in BACKENDS]
    if unknown:
        raise ValueError(
            f"UC_API_BACKENDS lists unknown backend(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(BACKENDS)}."
        )
    return tuple(chosen)


ENABLED_BACKENDS = _enabled_backends()


def _configured_tokens() -> frozenset[str]:
    """Accepted bearer tokens; empty means the instance is unauthenticated."""
    raw = _env("UC_API_TOKENS", "") or _env("UC_API_TOKEN", "")
    return frozenset(token.strip() for token in raw.split(",") if token.strip())


API_TOKENS = _configured_tokens()

_bearer_scheme = HTTPBearer(auto_error=False, description="UC_API_TOKENS entry")


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> None:
    """
    Gate an endpoint behind ``UC_API_TOKENS``.

    With no tokens configured this is a no-op, so a localhost instance behaves
    exactly as it always has; startup says so in the log rather than leaving the
    operator to infer it.
    """
    if not API_TOKENS:
        return

    unauthorised = HTTPException(
        401,
        "Authentication required: send Authorization: Bearer <token>",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorised

    supplied = credentials.credentials.encode("utf-8")
    # Every candidate is compared in full: a plain `==` would leak the token
    # prefix through response timing.
    if not any(secrets.compare_digest(supplied, known.encode("utf-8")) for known in API_TOKENS):
        raise unauthorised


CORS_ORIGINS = [o.strip() for o in _env("UC_API_CORS_ORIGINS", "*").split(",") if o.strip()]


@contextmanager
def _inference_slot():
    """Hold one of the model-inference slots for the duration of the block."""
    acquired = _inference_slots.acquire(timeout=float(_env("UC_API_QUEUE_TIMEOUT_S", "300")))
    if not acquired:
        raise HTTPException(503, "Server busy: inference queue timed out")
    try:
        yield
    finally:
        _inference_slots.release()


def settings_for_backend(base: BackendSettings, backend: str) -> BackendSettings:
    """
    Copy *base* with a different backend selected.

    ``model_name`` is dropped when the backend changes: a checkpoint configured
    through ``UC_SEG_MODEL`` names weights for one specific backend, and
    carrying it across would either fail to load or -- worse -- load something
    whose class space does not match what the taxonomy expects.
    """
    if backend == base.backend:
        return base
    return base.model_copy(update={"backend": backend, "model_name": None})


def backend_availability(settings: BackendSettings, backend: str) -> tuple[bool, str]:
    """
    Whether *backend* could be built, without paying to build it.

    Cheap checks only -- an import spec and a file on disk. A backend that
    passes here can still fail to load (a corrupt checkpoint, no VRAM); it just
    will not fail for the two reasons that are worth reporting up front.
    """
    if backend in ("oneformer", "mask2former"):
        if importlib.util.find_spec("transformers") is None:
            return False, 'transformers is not installed; pip install -e ".[ml]"'
        return True, ""

    if backend == "detectron2":
        if importlib.util.find_spec("detectron2") is None:
            return False, "Detectron2 is not installed; see docs/detectron2-windows.md"
        return True, ""

    if backend == "deeplab":
        if settings.deeplab_checkpoint is None:
            return False, "UC_DEEPLAB_CKPT is not set"
        if not settings.deeplab_checkpoint.is_file():
            return False, f"checkpoint not found: {settings.deeplab_checkpoint}"
        if settings.deeplab_repo is not None and not settings.deeplab_repo.is_dir():
            return False, f"UC_DEEPLAB_REPO not found: {settings.deeplab_repo}"
        return True, ""

    return False, f"unknown backend {backend!r}"


class SegmenterRegistry:
    """
    One segmenter per backend, built on first use and kept afterwards.

    Eager loading of every backend would multiply startup time and VRAM by four
    for a service that usually answers with one of them, so a backend costs
    nothing until a request names it. It is never unloaded: releasing a Torch
    model's device memory reliably is not something this can promise, and
    pretending otherwise would trade a predictable ceiling for an unpredictable
    one. ``UC_API_BACKENDS`` is the honest control -- it caps which backends an
    instance is willing to hold at all.
    """

    def __init__(self, settings: BackendSettings) -> None:
        self._settings = settings
        self._segmenters: dict[str, Any] = {}
        self._provenance: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, backend: str) -> tuple[Any, dict[str, Any]]:
        """The segmenter for *backend* and its provenance, building on demand."""
        with self._lock:
            if backend not in self._segmenters:
                settings = settings_for_backend(self._settings, backend)
                logger.info("Loading segmentation backend %r", backend)
                segmenter = build_segmenter_from_settings(settings)
                self._segmenters[backend] = segmenter
                self._provenance[backend] = backend_provenance(segmenter, settings)
                logger.info("Backend %r ready", backend)
            return self._segmenters[backend], self._provenance[backend]

    def loaded(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._segmenters))


class PipelineRegistry:
    """
    Lazily builds and caches one pipeline per combination that changes
    behaviour: the backend, and the configuration knobs (refinement on/off,
    vegetation proxy on/off, RGB retained or not). Segmenters come from the
    SegmenterRegistry and are shared; the Street View client is shared by all of
    them, being configuration-independent.
    """

    def __init__(self, segmenters: SegmenterRegistry, streetview: Any) -> None:
        self._segmenters = segmenters
        self._streetview = streetview
        self._pipes: dict[tuple[str, bool, bool, bool], Any] = {}
        self._lock = threading.Lock()

    def get(
        self,
        *,
        backend: str,
        refine: bool,
        allow_vegetation_proxy: bool,
        keep_rgb: bool = False,
    ):
        from urban_canopy.core.config import CanopyConfig
        from urban_canopy.core.pipeline import CanopyPipeline
        from urban_canopy.processing.refinement import RefinementConfig

        # Built outside the pipeline lock: loading a backend is slow, and it has
        # a lock of its own.
        segmenter, provenance = self._segmenters.get(backend)

        key = (backend, refine, allow_vegetation_proxy, keep_rgb)
        with self._lock:
            pipe = self._pipes.get(key)
            if pipe is None:
                pipe = CanopyPipeline(
                    segmenter=segmenter,
                    streetview=self._streetview,
                    config=CanopyConfig(
                        refinement=RefinementConfig(enabled=refine),
                        allow_vegetation_proxy=allow_vegetation_proxy,
                        keep_rgb=keep_rgb,
                    ),
                )
                self._pipes[key] = pipe
            return pipe, provenance


@asynccontextmanager
async def lifespan(app: FastAPI):
    import urban_canopy as uc

    backend_settings = BackendSettings()  # pyright: ignore[reportCallIssue]
    logger.info(
        "Starting Urban Canopy API (backend=%s, max_concurrency=%s)",
        backend_settings.backend,
        MAX_CONCURRENCY,
    )
    if API_TOKENS:
        logger.info(
            "Bearer authentication is ON (%s token(s) accepted); /ping stays open.",
            len(API_TOKENS),
        )
    else:
        logger.warning(
            "Bearer authentication is OFF: every caller can reach /analyse and spend "
            "the configured Google API quota. Set UC_API_TOKENS before exposing this "
            "instance beyond localhost."
        )
    if API_TOKENS and CORS_ORIGINS == ["*"]:
        logger.warning(
            "UC_API_CORS_ORIGINS is '*': any page may call this API with a token it "
            "holds. Pin it to the origins you serve the console from."
        )
    if backend_settings.backend not in ENABLED_BACKENDS:
        raise RuntimeError(
            f"UC_SEG_BACKEND={backend_settings.backend!r} is not in UC_API_BACKENDS "
            f"({', '.join(ENABLED_BACKENDS)}); the default must be one this instance serves."
        )

    streetview = uc.StreetViewClient()
    segmenters = SegmenterRegistry(backend_settings)

    app.state.backend_settings = backend_settings
    app.state.segmenters = segmenters
    app.state.registry = PipelineRegistry(segmenters, streetview)

    # Only the default is loaded now: it makes /ready meaningful and the first
    # request fast, without paying for backends nobody may ask for.
    _, provenance = segmenters.get(backend_settings.backend)
    app.state.backend_provenance = provenance

    loaded = segmenters.loaded()
    offered = [
        (
            name
            if name in loaded
            else f"{name}{'' if backend_availability(backend_settings, name)[0] else ' (unavailable)'}"
        )
        for name in ENABLED_BACKENDS
    ]
    logger.info("Backends offered: %s; default is %s", ", ".join(offered), backend_settings.backend)
    logger.info("Urban Canopy API ready")
    yield


app = FastAPI(
    title="Urban-Canopy",
    version="0.1.0",
    description="Visible tree-canopy coverage from Google Street View imagery.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Request / response schemas                                         #
# ------------------------------------------------------------------ #
class SingleViewRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    address: str | None = Field(
        default=None,
        json_schema_extra={"example": "Av. Paulista 1578, Sao Paulo"},
        description="Ignored if lat+lon are given",
    )
    lat: float | None = Field(None, ge=-90, le=90, description="Latitude (decimal deg)")
    lon: float | None = Field(None, ge=-180, le=180, description="Longitude (decimal deg)")
    heading: int = Field(0, ge=0, le=359)
    pitch: int = Field(0, ge=-90, le=90)
    fov: int = Field(90, ge=10, le=120)
    size: str = "640x640"
    backend: BackendName | None = Field(
        None,
        description=(
            "Segmentation backend; defaults to the instance's UC_SEG_BACKEND. "
            "Which backends are offered is listed by GET /ready."
        ),
    )
    refine: bool = True
    allow_vegetation_proxy: bool = False
    return_overlays: bool = Field(
        False, description="Include base64 PNG overlays (RGB, tree overlay, mask)"
    )

    @field_validator("lat")
    @classmethod
    def _latitude(cls, value):
        return None if value is None else validate_latitude(value)

    @field_validator("lon")
    @classmethod
    def _longitude(cls, value):
        return None if value is None else validate_longitude(value)

    @field_validator("size")
    @classmethod
    def _size(cls, value):
        return validate_image_size(value)

    @model_validator(mode="after")
    def _complete_coordinates(self):
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be provided together")
        return self


class MultiViewRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    address: str | None = None
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    reference_heading: int = Field(0, ge=0, le=359)
    mode: Literal["offsets", "equiangular"] = "offsets"
    offsets: list[int] = Field(default_factory=lambda: [0, 90, 180, 270], min_length=1)
    n_views: int = Field(4, ge=1, le=16)
    min_successful_views: int = Field(1, ge=1, le=16)
    pitch: int = Field(0, ge=-90, le=90)
    fov: int = Field(90, ge=10, le=120)
    size: str = "640x640"
    backend: BackendName | None = Field(
        None,
        description=(
            "Segmentation backend; defaults to the instance's UC_SEG_BACKEND. "
            "Which backends are offered is listed by GET /ready."
        ),
    )
    refine: bool = True
    allow_vegetation_proxy: bool = False
    return_overlays: bool = Field(
        False,
        description=(
            "Include base64 PNG overlays (RGB, tree overlay, mask) on every view. "
            f"Limited to {MAX_OVERLAY_VIEWS} planned headings; see UC_API_MAX_OVERLAY_VIEWS."
        ),
    )

    @field_validator("lat")
    @classmethod
    def _latitude(cls, value):
        return None if value is None else validate_latitude(value)

    @field_validator("lon")
    @classmethod
    def _longitude(cls, value):
        return None if value is None else validate_longitude(value)

    @field_validator("size")
    @classmethod
    def _size(cls, value):
        return validate_image_size(value)

    @model_validator(mode="after")
    def _complete_coordinates(self):
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be provided together")
        return self


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _resolve_location(req) -> tuple[float, float]:
    if req.lat is not None and req.lon is not None:
        return req.lat, req.lon
    if req.address:
        try:
            lat, lon = app.state.registry._streetview.geocode(req.address)
            return validate_latitude(lat), validate_longitude(lon)
        except Exception as exc:
            raise HTTPException(422, f"Geocoding failed: {exc}") from exc
    raise HTTPException(422, "Either address or lat+lon is required")


def _resolve_backend(requested: str | None) -> str:
    """The backend to serve this request with, refusing one we cannot honour."""
    settings = app.state.backend_settings
    backend = requested or settings.backend

    if backend not in ENABLED_BACKENDS:
        raise HTTPException(
            422,
            f"backend {backend!r} is not offered by this instance; "
            f"available: {', '.join(ENABLED_BACKENDS)}",
        )

    # A backend already loaded is usable by definition: whatever it took to
    # build it has happened. Asking the pre-flight probe about it would let an
    # import check overrule a working model -- which is what it did to an
    # instance whose segmenter was injected rather than imported.
    if backend in app.state.segmenters.loaded():
        return backend

    available, reason = backend_availability(settings, backend)
    if not available:
        # 503, not 422: the request is well formed, the server just cannot
        # satisfy it in its current configuration.
        raise HTTPException(503, f"backend {backend!r} is not usable here: {reason}")
    return backend


def _overlays(result) -> dict[str, str]:
    import cv2
    import numpy as np

    from urban_canopy.io.image_io import mask_overlay_bgr, png_b64

    if result.rgb_image is None:
        return {}
    bgr = cv2.cvtColor(np.asarray(result.rgb_image), cv2.COLOR_RGB2BGR)
    return {
        "rgb_png_b64": png_b64(bgr),
        "overlay_tree_png_b64": png_b64(mask_overlay_bgr(result.rgb_image, result.refined_mask)),
        "mask_refined_png_b64": png_b64(result.refined_mask.astype("uint8") * 255),
    }


# ------------------------------------------------------------------ #
# Endpoints                                                          #
# ------------------------------------------------------------------ #
@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", dependencies=[Depends(require_token)])
def ready() -> dict[str, Any]:
    provenance = getattr(app.state, "backend_provenance", None)
    if provenance is None:
        raise HTTPException(503, "Backend is not ready")

    settings = app.state.backend_settings
    loaded = app.state.segmenters.loaded()
    backends = []
    for name in ENABLED_BACKENDS:
        available, reason = backend_availability(settings, name)
        backends.append(
            {
                "name": name,
                # "loaded" is the strongest claim available without building it:
                # this one has already answered a request in this process.
                "status": (
                    "loaded" if name in loaded else ("available" if available else "unavailable")
                ),
                "reason": reason or None,
                "default": name == settings.backend,
                "class_space": BACKEND_CLASS_SPACE.get(name),
            }
        )
    return {
        "status": "ready",
        "backend": provenance,
        "default_backend": settings.backend,
        "backends": backends,
    }


@app.post("/analyse/single", dependencies=[Depends(require_token)])
def analyse_single(req: SingleViewRequest) -> dict[str, Any]:
    lat, lon = _resolve_location(req)
    backend = _resolve_backend(req.backend)
    try:
        pipe, provenance = app.state.registry.get(
            backend=backend,
            refine=req.refine,
            allow_vegetation_proxy=req.allow_vegetation_proxy,
            keep_rgb=req.return_overlays,
        )
    except Exception as exc:
        logger.exception("Loading backend %r failed", backend)
        raise HTTPException(503, f"Could not load backend {backend!r}: {exc}") from exc

    with _inference_slot():
        try:
            result = pipe.analyse_coords(
                lat, lon, heading=req.heading, pitch=req.pitch, fov=req.fov, size=req.size
            )
        except Exception as exc:
            logger.exception("Single-view analysis failed")
            raise HTTPException(500, f"Analysis failed: {exc}") from exc

    payload = result.to_dict()
    payload["backend_provenance"] = provenance
    if req.address:
        payload["capture"]["address"] = req.address
    if req.return_overlays:
        payload["overlays"] = _overlays(result)
    return payload


@app.post("/analyse/multi", dependencies=[Depends(require_token)])
def analyse_multi(req: MultiViewRequest) -> dict[str, Any]:
    from urban_canopy.core.viewplan import ViewPlanConfig, plan_headings

    lat, lon = _resolve_location(req)
    plan = ViewPlanConfig(
        mode=req.mode,
        reference_heading=req.reference_heading,
        offsets=tuple(req.offsets),
        n_views=req.n_views,
        pitch=req.pitch,
        fov=req.fov,
        size=req.size,
        min_successful_views=req.min_successful_views,
    )
    planned_count = len(plan_headings(plan))
    if plan.min_successful_views > planned_count:
        raise HTTPException(
            422,
            "min_successful_views cannot exceed the number of distinct "
            f"planned headings ({planned_count})",
        )
    if req.return_overlays and planned_count > MAX_OVERLAY_VIEWS:
        raise HTTPException(
            422,
            f"return_overlays is limited to {MAX_OVERLAY_VIEWS} planned headings; "
            f"this plan plans {planned_count}. Reduce the plan or omit the overlays.",
        )
    backend = _resolve_backend(req.backend)
    try:
        pipe, provenance = app.state.registry.get(
            backend=backend,
            refine=req.refine,
            allow_vegetation_proxy=req.allow_vegetation_proxy,
            keep_rgb=req.return_overlays,
        )
    except Exception as exc:
        logger.exception("Loading backend %r failed", backend)
        raise HTTPException(503, f"Could not load backend {backend!r}: {exc}") from exc

    with _inference_slot():
        try:
            result = pipe.analyse_multiview(lat, lon, plan=plan, address=req.address)
        except Exception as exc:
            from urban_canopy.core.pipeline import MultiViewAnalysisError

            if isinstance(exc, MultiViewAnalysisError):
                raise HTTPException(502, detail=exc.to_dict()) from exc
            logger.exception("Multi-view analysis failed")
            raise HTTPException(500, f"Analysis failed: {exc}") from exc

    payload = result.to_dict()
    payload["backend_provenance"] = provenance
    if req.return_overlays:
        # `to_dict` builds one entry per view, in order, so the pairing holds.
        # strict=True keeps a future change to that contract from silently
        # attaching one view's imagery to another view's metrics.
        for view, view_payload in zip(result.views, payload["views"], strict=True):
            overlays = _overlays(view)
            if overlays:
                view_payload["overlays"] = overlays
    return payload
