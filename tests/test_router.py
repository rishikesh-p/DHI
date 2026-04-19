import pytest
import numpy as np
from dhi.agent.router import Router

def test_router_fallback(mocker):
    # Mock embeddings to raise an exception so it falls back to heuristics
    mocker.patch("dhi.agent.router.OllamaEmbeddings", side_effect=Exception("Offline"))
    router = Router()
    assert router.online is False
    
    # Under 15 words -> local
    assert router.route("create a file") == "local"
    # Over 15 words -> cloud
    assert router.route("I want you to " * 5 + "create a python script that does something very complex") == "cloud"

def test_router_semantic_similarity(mocker):
    # Mock OllamaEmbeddings so we don't need a running Ollama daemon
    mock_embedder_class = mocker.patch("dhi.agent.router.OllamaEmbeddings")
    mock_instance = mock_embedder_class.return_value
    
    # Mock the archetypes vectors
    # local_vectors -> [1, 0]
    # cloud_vectors -> [0, 1]
    mock_instance.embed_documents.side_effect = [
        [[1.0, 0.0]] * 10,  # local archetypes
        [[0.0, 1.0]] * 8    # cloud archetypes
    ]
    
    router = Router()
    assert router.online is True
    
    # Test a query that is semantically close to [1, 0] (Local)
    mock_instance.embed_query.return_value = [0.9, 0.1]
    assert router.route("some simple command") == "local"
    
    # Test a query that is semantically close to [0, 1] (Cloud)
    mock_instance.embed_query.return_value = [0.1, 0.9]
    assert router.route("some complex script") == "cloud"
