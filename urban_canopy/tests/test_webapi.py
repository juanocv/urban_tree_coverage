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
    assert (True, False, True) in webapi.app.state.registry._pipes


def test_single_view_without_overlays_uses_non_rgb_pipeline(client):
    response = client.post("/analyse/single", json={"lat": -23.0, "lon": -46.0})
    assert response.status_code == 200
    assert (True, False, False) in webapi.app.state.registry._pipes


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
    assert (True, False, True) in webapi.app.state.registry._pipes


def test_multi_view_without_overlays_omits_them(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90]},
    )
    assert response.status_code == 200
    assert all("overlays" not in view for view in response.json()["views"])
    assert (True, False, False) in webapi.app.state.registry._pipes


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
