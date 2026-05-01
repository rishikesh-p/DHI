import pytest
import numpy as np
from dhi.agent.router import Router



def test_router_semantic_similarity(mocker):
    # Mock OllamaEmbeddings so we don't need a running Ollama daemon
    mock_embedder_class = mocker.patch("dhi.agent.router.OllamaEmbeddings")
    mock_instance = mock_embedder_class.return_value
    
    # mock_instance.embed_documents is called for complex first, then simple
    # complex_vectors -> [1, 0]
    # simple_vectors -> [0, 1]
    mock_instance.embed_documents.side_effect = [
        [[1.0, 0.0]] * 8,   # complex archetypes
        [[0.0, 1.0]] * 10   # simple archetypes
    ]
    
    router = Router()
    
    # Test a query that is semantically close to [0, 1] (Simple / Local)
    mock_instance.embed_query.return_value = [0.1, 0.9]
    assert router.route("some simple command") == "local"
    
    # Test a query that is semantically close to [1, 0] (Complex / Cloud)
    mock_instance.embed_query.return_value = [0.9, 0.1]
    assert router.route("some complex script") == "cloud"
