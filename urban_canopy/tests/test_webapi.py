"""Web API tests with a stub segmenter and stubbed Street View I/O."""

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import urban_canopy as uc  # noqa: E402
from urban_canopy import webapi  # noqa: E402
from urban_canopy.io.streetview import Settings, StreetViewClient  # noqa: E402
from urban_canopy.models.base import SegmentationOutput  # noqa: E402
from urban_canopy.models.taxonomy import ADE20K  # noqa: E402


class StubSegmenter:
    backend_name = "stub"
    class_space = "ade20k"
    taxonomy = ADE20K

    def segment(self, img_rgb):
        height, width = img_rgb.shape[:2]
        tree = np.zeros((height, width), bool)
        tree[: height // 2, :] = True
        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks={
                "tree": tree,
                "grass": np.zeros_like(tree),
                "plant_shrub": np.zeros_like(tree),
            },
        )


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    """
    Hermetic defaults: tests must never inherit settings from whatever
    ``.env`` happens to sit in the developer's or CI's working directory.

    ``webapi`` reads ``UC_API_*`` values once at import time -- both into
    ``API_TOKENS`` and into the ``_DOTENV_VALUES`` fallback dict -- so an env
    var set after import would not apply anyway; patching the attributes
    directly is the only thing that reaches every request. Tests that want
    auth on use the ``guarded`` fixture below, which overrides ``API_TOKENS``.
    """
    monkeypatch.setattr(webapi, "API_TOKENS", frozenset())
    monkeypatch.setattr(webapi, "_DOTENV_VALUES", {})


def pipeline_key(*flags: bool) -> tuple:
    """The registry key for the instance's default backend plus *flags*."""
    return (webapi.app.state.backend_settings.backend, *flags)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    frame = tmp_path / "sv.jpg"
    cv2.imwrite(str(frame), np.zeros((40, 60, 3), np.uint8))

    monkeypatch.setattr(webapi, "build_segmenter_from_settings", lambda settings: StubSegmenter())
    monkeypatch.setattr(
        uc,
        "StreetViewClient",
        lambda *a, **k: StreetViewClient(
            cache_dir=tmp_path / "cache", settings=Settings(google_api_key="test-key")
        ),
    )
    monkeypatch.setattr(StreetViewClient, "fetch", lambda self, req: frame)
    monkeypatch.setattr(StreetViewClient, "geocode", lambda self, addr: (-23.0, -46.0))
    monkeypatch.setattr(StreetViewClient, "metadata", lambda self, lat, lon: {})

    with TestClient(webapi.app) as test_client:
        yield test_client


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_exposes_backend_provenance(client):
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["backend"]["backend"] == "stub"
    assert payload["backend"]["class_space"] == "ade20k"
    assert payload["backend"]["taxonomy"]["class_space"] == "ade20k"


def test_single_view(client):
    response = client.post("/analyse/single", json={"lat": -23.0, "lon": -46.0, "heading": 90})
    assert response.status_code == 200
    payload = response.json()
    assert payload["coverage"]["tree_coverage_pct"] == pytest.approx(50.0)
    assert payload["coverage"]["tree_source"] == "tree_class"
    assert payload["capture"]["heading"] == 90
    assert payload["backend_provenance"]["backend"] == "stub"


def test_single_view_by_address(client):
    response = client.post("/analyse/single", json={"address": "Av. Paulista 1578"})
    assert response.status_code == 200
    assert response.json()["capture"]["address"] == "Av. Paulista 1578"


def test_single_view_needs_a_location(client):
    response = client.post("/analyse/single", json={})
    assert response.status_code == 422


def test_single_view_overlays(client):
    response = client.post(
        "/analyse/single",
        json={"lat": -23.0, "lon": -46.0, "return_overlays": True},
    )
    assert response.status_code == 200
    overlays = response.json()["overlays"]
    assert set(overlays) == {"rgb_png_b64", "overlay_tree_png_b64", "mask_refined_png_b64"}
    assert pipeline_key(True, False, True) in webapi.app.state.registry._pipes


def test_single_view_without_overlays_uses_non_rgb_pipeline(client):
    response = client.post("/analyse/single", json={"lat": -23.0, "lon": -46.0})
    assert response.status_code == 200
    assert pipeline_key(True, False, False) in webapi.app.state.registry._pipes


@pytest.mark.parametrize(
    "payload",
    [
        {"lat": 91, "lon": 0},
        {"lat": 0, "lon": 181},
        {"lat": 0},
        {"lat": 0, "lon": 0, "size": "640*640"},
        {"lat": 0, "lon": 0, "size": "5000x640"},
    ],
)
def test_single_view_rejects_invalid_capture_configuration(client, payload):
    assert client.post("/analyse/single", json=payload).status_code == 422


def test_multi_view(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90, 180, 270]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["aggregate"]["tree_coverage"]["n_valid_views"] == 4
    assert payload["aggregate"]["tree_coverage"]["median"] == pytest.approx(0.5)
    assert len(payload["views"]) == 4
    assert payload["backend_provenance"]["backend"] == "stub"


def test_multi_view_equiangular(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "mode": "equiangular", "n_views": 3},
    )
    assert response.status_code == 200
    assert response.json()["plan"]["planned_headings"] == [0, 120, 240]


def test_multi_view_total_failure_is_a_bad_gateway(client, monkeypatch):
    def fail(self, req):
        raise RuntimeError("imagery unavailable")

    monkeypatch.setattr(StreetViewClient, "fetch", fail)
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90]},
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["successful_headings"] == []
    assert [failure["heading"] for failure in detail["failures"]] == [0, 90]


def test_multi_view_rejects_impossible_success_minimum(client):
    response = client.post(
        "/analyse/multi",
        json={
            "lat": -23.0,
            "lon": -46.0,
            "offsets": [0, 90],
            "min_successful_views": 3,
        },
    )
    assert response.status_code == 422
    assert "distinct planned headings (2)" in response.json()["detail"]


def test_multi_view_overlays(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90], "return_overlays": True},
    )
    assert response.status_code == 200
    views = response.json()["views"]
    assert len(views) == 2
    for view in views:
        assert set(view["overlays"]) == {
            "rgb_png_b64",
            "overlay_tree_png_b64",
            "mask_refined_png_b64",
        }
    # Overlays need the RGB-retaining pipeline, same as the single-view endpoint.
    assert pipeline_key(True, False, True) in webapi.app.state.registry._pipes


def test_multi_view_without_overlays_omits_them(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90]},
    )
    assert response.status_code == 200
    assert all("overlays" not in view for view in response.json()["views"])
    assert pipeline_key(True, False, False) in webapi.app.state.registry._pipes


def test_multi_view_overlays_are_capped_by_plan_size(client, monkeypatch):
    monkeypatch.setattr(webapi, "MAX_OVERLAY_VIEWS", 2)
    response = client.post(
        "/analyse/multi",
        json={
            "lat": -23.0,
            "lon": -46.0,
            "offsets": [0, 90, 180],
            "return_overlays": True,
        },
    )
    assert response.status_code == 422
    assert "limited to 2 planned headings" in response.json()["detail"]


def test_multi_view_overlay_cap_ignored_without_overlays(client, monkeypatch):
    monkeypatch.setattr(webapi, "MAX_OVERLAY_VIEWS", 2)
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90, 180]},
    )
    assert response.status_code == 200
    assert len(response.json()["views"]) == 3


def test_multi_view_overlays_skip_views_without_imagery(client, monkeypatch):
    """A view whose RGB was dropped reports no overlays rather than empty ones."""
    real_overlays = webapi._overlays
    calls = {"n": 0}

    def sometimes_missing(result):
        calls["n"] += 1
        return {} if calls["n"] == 1 else real_overlays(result)

    monkeypatch.setattr(webapi, "_overlays", sometimes_missing)
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90], "return_overlays": True},
    )
    assert response.status_code == 200
    views = response.json()["views"]
    assert "overlays" not in views[0]
    assert "overlays" in views[1]


# ------------------------------------------------------------------ #
# Bearer authentication                                              #
# ------------------------------------------------------------------ #
@pytest.fixture()
def guarded(client, monkeypatch):
    """The same client, with two tokens configured."""
    monkeypatch.setattr(webapi, "API_TOKENS", frozenset({"correct-horse", "second-key"}))
    return client


def test_tokens_are_read_from_either_env_name(monkeypatch):
    # _DOTENV_VALUES is parsed once, from whatever .env sits on this machine;
    # patch it too, so this test's outcome does not depend on that file.
    monkeypatch.setattr(webapi, "_DOTENV_VALUES", {})
    monkeypatch.delenv("UC_API_TOKEN", raising=False)
    monkeypatch.setenv("UC_API_TOKENS", " alpha , beta ,, ")
    assert webapi._configured_tokens() == frozenset({"alpha", "beta"})

    monkeypatch.delenv("UC_API_TOKENS", raising=False)
    monkeypatch.setenv("UC_API_TOKEN", "solo")
    assert webapi._configured_tokens() == frozenset({"solo"})

    monkeypatch.delenv("UC_API_TOKEN", raising=False)
    assert webapi._configured_tokens() == frozenset()


def test_tokens_fall_back_to_the_env_file_when_unset_in_the_process(monkeypatch):
    """A token only present in .env still takes effect -- the original bug."""
    monkeypatch.delenv("UC_API_TOKENS", raising=False)
    monkeypatch.delenv("UC_API_TOKEN", raising=False)
    monkeypatch.setattr(webapi, "_DOTENV_VALUES", {"UC_API_TOKENS": "from-the-file"})
    assert webapi._configured_tokens() == frozenset({"from-the-file"})

    # A real process env var still overrides the file, matching every other
    # UC_API_* setting.
    monkeypatch.setenv("UC_API_TOKENS", "from-the-environment")
    assert webapi._configured_tokens() == frozenset({"from-the-environment"})


def test_unauthenticated_instance_stays_open(client):
    """No tokens configured means the localhost workflow is untouched."""
    assert client.get("/ready").status_code == 200
    assert client.post("/analyse/single", json={"lat": -23.0, "lon": -46.0}).status_code == 200


def test_ping_is_reachable_without_a_token(guarded):
    """Liveness has to answer probes that hold no secret."""
    response = guarded.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/ready", None),
        ("post", "/analyse/single", {"lat": -23.0, "lon": -46.0}),
        ("post", "/analyse/multi", {"lat": -23.0, "lon": -46.0, "offsets": [0]}),
    ],
)
def test_guarded_endpoints_reject_a_missing_token(guarded, method, path, body):
    response = getattr(guarded, method)(path, **({"json": body} if body else {}))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    [
        "Bearer wrong-token",
        "Bearer correct-horse-extra",
        "Bearer correct-hors",
        "Basic correct-horse",
        "correct-horse",
        "Bearer ",
    ],
)
def test_guarded_endpoints_reject_a_bad_token(guarded, header):
    response = guarded.get("/ready", headers={"Authorization": header})
    assert response.status_code == 401


@pytest.mark.parametrize("token", ["correct-horse", "second-key"])
def test_any_configured_token_is_accepted(guarded, token):
    response = guarded.get("/ready", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_analysis_succeeds_with_a_token(guarded):
    response = guarded.post(
        "/analyse/single",
        json={"lat": -23.0, "lon": -46.0},
        headers={"Authorization": "Bearer second-key"},
    )
    assert response.status_code == 200
    assert response.json()["coverage"]["tree_coverage_pct"] == pytest.approx(50.0)


def test_bearer_scheme_is_case_insensitive(guarded):
    response = guarded.get("/ready", headers={"Authorization": "bearer correct-horse"})
    assert response.status_code == 200


# ------------------------------------------------------------------ #
# Backend selection                                                  #
# ------------------------------------------------------------------ #
class SecondStubSegmenter(StubSegmenter):
    """A second identity, so a per-request backend switch is observable."""

    backend_name = "stub-two"


@pytest.fixture()
def multi_backend(client, monkeypatch):
    """Serve two backends, each building a distinguishable segmenter."""
    monkeypatch.setattr(webapi, "ENABLED_BACKENDS", ("oneformer", "mask2former"))
    monkeypatch.setattr(webapi, "backend_availability", lambda settings, backend: (True, ""))
    monkeypatch.setattr(
        webapi,
        "build_segmenter_from_settings",
        lambda settings: (
            StubSegmenter() if settings.backend == "oneformer" else SecondStubSegmenter()
        ),
    )
    # The registry built at startup captured the old builder; rebuild it so the
    # per-backend one above is the one under test.
    webapi.app.state.segmenters = webapi.SegmenterRegistry(webapi.app.state.backend_settings)
    webapi.app.state.registry = webapi.PipelineRegistry(
        webapi.app.state.segmenters, webapi.app.state.registry._streetview
    )
    return client


def test_settings_for_backend_drops_a_backend_specific_checkpoint():
    from urban_canopy.models.backend_settings import BackendSettings

    base = BackendSettings(backend="oneformer", model_name="shi-labs/oneformer_ade20k_swin_large")
    assert webapi.settings_for_backend(base, "oneformer") is base

    switched = webapi.settings_for_backend(base, "mask2former")
    assert switched.backend == "mask2former"
    assert switched.model_name is None, "a OneFormer checkpoint must not follow to Mask2Former"


def test_enabled_backends_defaults_to_every_backend(monkeypatch):
    from urban_canopy.models.factory import BACKENDS

    monkeypatch.setattr(webapi, "_DOTENV_VALUES", {})
    monkeypatch.delenv("UC_API_BACKENDS", raising=False)
    assert webapi._enabled_backends() == tuple(BACKENDS)


def test_enabled_backends_can_be_narrowed(monkeypatch):
    monkeypatch.setattr(webapi, "_DOTENV_VALUES", {})
    monkeypatch.setenv("UC_API_BACKENDS", " deeplab , oneformer ")
    assert webapi._enabled_backends() == ("deeplab", "oneformer")


def test_enabled_backends_rejects_an_unknown_name(monkeypatch):
    monkeypatch.setattr(webapi, "_DOTENV_VALUES", {})
    monkeypatch.setenv("UC_API_BACKENDS", "oneformer,not-a-backend")
    with pytest.raises(ValueError, match="not-a-backend"):
        webapi._enabled_backends()


def test_readiness_lists_every_offered_backend(client):
    payload = client.get("/ready").json()
    names = [entry["name"] for entry in payload["backends"]]
    assert names == list(webapi.ENABLED_BACKENDS)
    assert payload["default_backend"] == webapi.app.state.backend_settings.backend

    default_entry = next(e for e in payload["backends"] if e["default"])
    assert default_entry["status"] == "loaded", "the default is built during startup"
    assert all(e["status"] in {"loaded", "available", "unavailable"} for e in payload["backends"])


def test_readiness_explains_an_unavailable_backend(client, monkeypatch):
    monkeypatch.setattr(
        webapi,
        "backend_availability",
        lambda settings, backend: (
            (False, "checkpoint not found: /nope.pth") if backend == "deeplab" else (True, "")
        ),
    )
    entries = {e["name"]: e for e in client.get("/ready").json()["backends"]}
    if "deeplab" in entries:
        assert entries["deeplab"]["status"] == "unavailable"
        assert entries["deeplab"]["reason"] == "checkpoint not found: /nope.pth"


def test_request_selects_the_backend(multi_backend):
    first = multi_backend.post(
        "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "oneformer"}
    )
    second = multi_backend.post(
        "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "mask2former"}
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["backend_provenance"]["backend"] == "stub"
    assert second.json()["backend_provenance"]["backend"] == "stub-two"


def test_multi_view_selects_the_backend(multi_backend):
    response = multi_backend.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0], "backend": "mask2former"},
    )
    assert response.status_code == 200
    assert response.json()["backend_provenance"]["backend"] == "stub-two"


def test_omitting_the_backend_uses_the_instance_default(multi_backend):
    response = multi_backend.post("/analyse/single", json={"lat": -23.0, "lon": -46.0})
    assert response.status_code == 200
    default = webapi.app.state.backend_settings.backend
    expected = "stub" if default == "oneformer" else "stub-two"
    assert response.json()["backend_provenance"]["backend"] == expected


def test_each_backend_gets_its_own_pipeline(multi_backend):
    multi_backend.post("/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "oneformer"})
    multi_backend.post(
        "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "mask2former"}
    )
    keys = webapi.app.state.registry._pipes
    assert ("oneformer", True, False, False) in keys
    assert ("mask2former", True, False, False) in keys


def test_a_backend_outside_the_offer_is_refused(multi_backend):
    response = multi_backend.post(
        "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "deeplab"}
    )
    assert response.status_code == 422
    assert "not offered" in response.json()["detail"]


def test_an_unusable_backend_is_a_service_error_not_a_bad_request(multi_backend, monkeypatch):
    """A well-formed request the server cannot satisfy is 503, not 422."""
    monkeypatch.setattr(
        webapi,
        "backend_availability",
        lambda settings, backend: (
            (False, "UC_DEEPLAB_CKPT is not set") if backend == "mask2former" else (True, "")
        ),
    )
    response = multi_backend.post(
        "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "mask2former"}
    )
    assert response.status_code == 503
    assert "UC_DEEPLAB_CKPT is not set" in response.json()["detail"]


def test_a_backend_that_fails_to_load_reports_which_one(multi_backend, monkeypatch):
    def explode(settings):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(webapi, "build_segmenter_from_settings", explode)
    webapi.app.state.segmenters = webapi.SegmenterRegistry(webapi.app.state.backend_settings)
    webapi.app.state.registry = webapi.PipelineRegistry(
        webapi.app.state.segmenters, webapi.app.state.registry._streetview
    )
    response = multi_backend.post(
        "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "mask2former"}
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "mask2former" in detail and "CUDA out of memory" in detail


def test_an_invalid_backend_name_is_rejected_by_the_schema(client):
    response = client.post(
        "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "not-a-backend"}
    )
    assert response.status_code == 422


def test_segmenters_are_built_once_and_reused(multi_backend):
    for _ in range(3):
        multi_backend.post(
            "/analyse/single", json={"lat": -23.0, "lon": -46.0, "backend": "mask2former"}
        )
    registry = webapi.app.state.segmenters
    assert registry.loaded() == ("mask2former",), "only the requested backend is resident"
    first, _ = registry.get("mask2former")
    second, _ = registry.get("mask2former")
    assert first is second
