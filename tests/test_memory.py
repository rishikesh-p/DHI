import pytest
from dhi.agent.memory import MemorySystem

def test_memory_init_and_save(tmp_path, mocker):
    # Mock OllamaEmbeddings to prevent network calls to Ollama daemon
    mock_embedder = mocker.patch("dhi.agent.memory.OllamaEmbeddings")
    mock_instance = mock_embedder.return_value
    
    # Return a dummy vector based on text length to distinguish them slightly
    mock_instance.embed_query.side_effect = lambda text: [float(len(text))] * 768

    db_dir = tmp_path / "lancedb"
    mem = MemorySystem(db_path=str(db_dir))
    
    # Assert DB and Table are initialized
    assert mem.table is not None
    assert mem.table.name == "knowledge_base"
    
    # Table starts with 'Init' record
    initial_rows = len(mem.table.to_pandas())
    
    # Teach it a new command
    mem.save("Request: execute echo -> Command: echo 'hello'")
    
    # Table should now have +1 row
    assert len(mem.table.to_pandas()) == initial_rows + 1

def test_memory_recall(tmp_path, mocker):
    mock_embedder = mocker.patch("dhi.agent.memory.OllamaEmbeddings")
    mock_instance = mock_embedder.return_value
    mock_instance.embed_query.side_effect = lambda text: [float(len(text))] * 768

    db_dir = tmp_path / "lancedb"
    mem = MemorySystem(db_path=str(db_dir))
    
    test_str = "Request: list files -> Command: ls -la"
    mem.save(test_str)
    
    # Because LanceDB does vector similarity, and our mock vector is based on length,
    # asking for the exact same string length ensures we get the exact same vector back
    recalled = mem.recall(test_str)
    
    assert test_str == recalled
