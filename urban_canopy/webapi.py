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
    BackendSettings,
    backend_provenance,
    build_segmenter_from_settings,
)
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


class PipelineRegistry:
    """
    Lazily builds and caches one pipeline per configuration knob that changes
    behaviour (refinement on/off, vegetation proxy on/off). The segmenter and
    Street View client are shared across all of them: they are the expensive
    parts and they are configuration-independent.
    """

    def __init__(self, segmenter: Any, streetview: Any) -> None:
        self._segmenter = segmenter
        self._streetview = streetview
        self._pipes: dict[tuple[bool, bool, bool], Any] = {}
        self._lock = threading.Lock()

    def get(self, *, refine: bool, allow_vegetation_proxy: bool, keep_rgb: bool = False):
        from urban_canopy.core.config import CanopyConfig
        from urban_canopy.core.pipeline import CanopyPipeline
        from urban_canopy.processing.refinement import RefinementConfig

        key = (refine, allow_vegetation_proxy, keep_rgb)
        with self._lock:
            pipe = self._pipes.get(key)
            if pipe is None:
                pipe = CanopyPipeline(
                    segmenter=self._segmenter,
                    streetview=self._streetview,
                    config=CanopyConfig(
                        refinement=RefinementConfig(enabled=refine),
                        allow_vegetation_proxy=allow_vegetation_proxy,
                        keep_rgb=keep_rgb,
                    ),
                )
                self._pipes[key] = pipe
            return pipe


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
    segmenter = build_segmenter_from_settings(backend_settings)
    streetview = uc.StreetViewClient()

    app.state.backend_settings = backend_settings
    app.state.backend_provenance = backend_provenance(segmenter, backend_settings)
    app.state.registry = PipelineRegistry(segmenter, streetview)
    app.state.registry.get(refine=True, allow_vegetation_proxy=False)
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
    return {"status": "ready", "backend": provenance}


@app.post("/analyse/single", dependencies=[Depends(require_token)])
def analyse_single(req: SingleViewRequest) -> dict[str, Any]:
    lat, lon = _resolve_location(req)
    pipe = app.state.registry.get(
        refine=req.refine,
        allow_vegetation_proxy=req.allow_vegetation_proxy,
        keep_rgb=req.return_overlays,
    )
    with _inference_slot():
        try:
            result = pipe.analyse_coords(
                lat, lon, heading=req.heading, pitch=req.pitch, fov=req.fov, size=req.size
            )
        except Exception as exc:
            logger.exception("Single-view analysis failed")
            raise HTTPException(500, f"Analysis failed: {exc}") from exc

    payload = result.to_dict()
    payload["backend_provenance"] = app.state.backend_provenance
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
    pipe = app.state.registry.get(
        refine=req.refine,
        allow_vegetation_proxy=req.allow_vegetation_proxy,
        keep_rgb=req.return_overlays,
    )
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
    payload["backend_provenance"] = app.state.backend_provenance
    if req.return_overlays:
        # `to_dict` builds one entry per view, in order, so the pairing holds.
        # strict=True keeps a future change to that contract from silently
        # attaching one view's imagery to another view's metrics.
        for view, view_payload in zip(result.views, payload["views"], strict=True):
            overlays = _overlays(view)
            if overlays:
                view_payload["overlays"] = overlays
    return payload
