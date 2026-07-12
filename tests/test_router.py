import pytest
import numpy as np
from dhi.agent.router import Router


# -- Fixtures --

@pytest.fixture
def mock_router(mocker):
    """Build a Router with mocked embeddings and no disk I/O."""
    mock_embedder_class = mocker.patch("dhi.agent.router.OllamaEmbeddings")
    mock_instance = mock_embedder_class.return_value

    # Complex anchors -> [1, 0], simple anchors -> [0, 1].
    # Use the actual anchor counts from the Router class.
    mock_instance.embed_documents.side_effect = [
        [[1.0, 0.0]] * 27,   # complex archetypes (27 anchors).
        [[0.0, 1.0]] * 22    # simple archetypes (22 anchors).
    ]

    mocker.patch("dhi.agent.router.Router._cache_is_valid", return_value=False)
    mocker.patch("dhi.agent.router.np.save")
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.makedirs")

    router = Router()
    return router, mock_instance


# -- Routing decisions --

class TestRoutingDecisions:
    def test_simple_query_routes_local(self, mock_router):
        """Verify a query close to [0, 1] (simple anchors) routes to local."""
        router, mock_instance = mock_router
        mock_instance.embed_query.return_value = [0.1, 0.9]
        result = router.route("list files")
        assert result["route"] == "local"

    def test_complex_query_routes_cloud(self, mock_router):
        """Verify a query close to [1, 0] (complex anchors) routes to cloud."""
        router, mock_instance = mock_router
        mock_instance.embed_query.return_value = [0.9, 0.1]
        result = router.route("write a python web scraper")
        assert result["route"] == "cloud"

    def test_ambiguous_query_routes_local(self, mock_router):
        """Verify a balanced query defaults to local (margin-based tiebreak)."""
        router, mock_instance = mock_router
        mock_instance.embed_query.return_value = [0.5, 0.5]
        result = router.route("do something")
        assert result["route"] == "local"


# -- Confidence scores --

class TestConfidenceScores:
    def test_confidence_is_bounded(self, mock_router):
        """Verify confidence is always between 0.0 and 1.0."""
        router, mock_instance = mock_router

        for query_vec in [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.01, 0.99]]:
            mock_instance.embed_query.return_value = query_vec
            result = router.route("test query")
            assert 0.0 <= result["confidence"] <= 1.0, f"Confidence out of bounds for vec {query_vec}"

    def test_strong_match_has_high_confidence(self, mock_router):
        """Verify a vector strongly aligned with one set produces high confidence."""
        router, mock_instance = mock_router
        mock_instance.embed_query.return_value = [0.0, 1.0]
        result = router.route("simple command")
        assert result["confidence"] > 0.5


# -- Output contract --

class TestOutputContract:
    def test_returns_required_keys(self, mock_router):
        """Verify route() always returns both 'route' and 'confidence' keys."""
        router, mock_instance = mock_router
        mock_instance.embed_query.return_value = [0.5, 0.5]
        result = router.route("anything")
        assert "route" in result
        assert "confidence" in result

    def test_route_is_valid_value(self, mock_router):
        """Verify route is either 'local' or 'cloud'."""
        router, mock_instance = mock_router
        for vec in [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]:
            mock_instance.embed_query.return_value = vec
            result = router.route("test")
            assert result["route"] in {"local", "cloud"}


# -- Vectorized computation correctness --

class TestVectorizedComputation:
    def test_normalized_embeddings(self, mock_router):
        """Verify pre-normalized embedding rows have unit length."""
        router, _ = mock_router
        for row in router._complex_normed:
            assert abs(np.linalg.norm(row) - 1.0) < 1e-6
        for row in router._simple_normed:
            assert abs(np.linalg.norm(row) - 1.0) < 1e-6

    def test_deterministic_routing(self, mock_router):
        """Verify the same input produces the same result on repeated calls."""
        router, mock_instance = mock_router
        mock_instance.embed_query.return_value = [0.3, 0.7]
        results = [router.route("test query") for _ in range(5)]
        routes = [r["route"] for r in results]
        confidences = [r["confidence"] for r in results]
        assert len(set(routes)) == 1, "Routing should be deterministic"
        assert len(set(confidences)) == 1, "Confidence should be deterministic"
