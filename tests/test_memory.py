import pytest
from dhi.agent.memory import MemorySystem, RELEVANCE_THRESHOLD, DEDUP_THRESHOLD


# -- Fixtures --

def _make_embedder(mocker):
    """Mock OllamaEmbeddings with length-based deterministic vectors."""
    mock_embedder = mocker.patch("dhi.agent.memory.OllamaEmbeddings")
    mock_instance = mock_embedder.return_value
    mock_instance.embed_query.side_effect = lambda text: [
        1.0 if i == (len(text) % 768) else 0.0 for i in range(768)
    ]
    return mock_instance


@pytest.fixture
def mem(tmp_path, mocker):
    """Provide a MemorySystem backed by a temporary LanceDB."""
    _make_embedder(mocker)
    db_dir = tmp_path / "lancedb"
    return MemorySystem(db_path=str(db_dir))


# -- Initialization --

class TestInitialization:
    def test_table_is_created(self, mem):
        """Verify the knowledge_base table is initialized on startup."""
        assert mem.table is not None
        assert mem.table.name == "knowledge_base"

    def test_seed_record_exists(self, mem):
        """Verify the __init__ seed record exists immediately after creation."""
        df = mem.table.to_pandas()
        assert len(df) >= 1
        assert "__init__" in df["intent"].values

    def test_graceful_db_failure(self, tmp_path, mocker):
        """Verify a database connection error sets table to None."""
        _make_embedder(mocker)
        mocker.patch("dhi.agent.memory.lancedb.connect", side_effect=Exception("DB error"))
        mem = MemorySystem(db_path=str(tmp_path / "bad"))
        assert mem.table is None


# -- save --

class TestSave:
    def test_save_adds_row(self, mem):
        """Verify save() adds exactly one row to the table."""
        initial = len(mem.table.to_pandas())
        mem.save("execute echo command", "echo 'hello world from the test suite'")
        assert len(mem.table.to_pandas()) == initial + 1

    def test_save_stores_intent_and_command(self, mem):
        """Verify saved intent and command are retrievable from the table."""
        mem.save("show disk usage", "df -h")
        df = mem.table.to_pandas()
        row = df[df["intent"] == "show disk usage"]
        assert len(row) == 1
        assert row.iloc[0]["command"] == "df -h"

    def test_save_stores_success_flag(self, mem):
        """Verify the success flag is stored correctly."""
        mem.save("failing cmd", "exit 1", success=False)
        df = mem.table.to_pandas()
        row = df[df["intent"] == "failing cmd"]
        assert row.iloc[0]["success"] == False

    def test_dedup_prevents_identical_save(self, mem):
        """Verify saving the same intent twice does not create a duplicate."""
        mem.save("list files", "ls -la")
        count_after_first = len(mem.table.to_pandas())
        mem.save("list files", "ls -la")
        count_after_second = len(mem.table.to_pandas())
        assert count_after_second == count_after_first

    def test_save_noop_when_table_is_none(self, tmp_path, mocker):
        """Verify save() is a no-op when the table failed to initialize."""
        _make_embedder(mocker)
        mocker.patch("dhi.agent.memory.lancedb.connect", side_effect=Exception("fail"))
        mem = MemorySystem(db_path=str(tmp_path / "bad"))
        mem.save("test", "test")  # Should not raise.


# -- recall --

class TestRecall:
    def test_recall_finds_saved_entry(self, mem):
        """Verify recall() finds a previously saved entry."""
        mem.save("list files", "ls -la")
        recalled = mem.recall("list files")
        assert "list files" in recalled
        assert "ls -la" in recalled

    def test_recall_returns_formatted_string(self, mem):
        """Verify recall() returns a string with '- ' prefix per line."""
        mem.save("list files", "ls -la")
        recalled = mem.recall("list files")
        assert recalled.startswith("- ")

    def test_recall_filters_init_seed(self, mem):
        """Verify the __init__ seed record never appears in recall results."""
        recalled = mem.recall("__init__")
        assert "__init__" not in recalled

    def test_recall_returns_empty_when_no_table(self, tmp_path, mocker):
        """Verify recall() returns empty string when table is None."""
        _make_embedder(mocker)
        mocker.patch("dhi.agent.memory.lancedb.connect", side_effect=Exception("fail"))
        mem = MemorySystem(db_path=str(tmp_path / "bad"))
        assert mem.recall("anything") == ""

    def test_recall_returns_empty_for_irrelevant_query(self, tmp_path, mocker):
        """Verify recall() returns empty when no results pass the relevance threshold."""
        mock_embedder = mocker.patch("dhi.agent.memory.OllamaEmbeddings")
        mock_instance = mock_embedder.return_value

        def mock_embed(text):
            vec = [0.0] * 768
            if "unrelated" in text:
                vec[0] = 1.0
            else:
                vec[1] = 1.0
            return vec

        mock_instance.embed_query.side_effect = mock_embed
        db_dir = tmp_path / "lancedb"
        mem = MemorySystem(db_path=str(db_dir))

        mem.save("list files", "ls -la")
        recalled = mem.recall("completely unrelated topic here")
        assert recalled == ""

    def test_recall_respects_limit(self, mem):
        """Verify recall() returns at most `limit` results."""
        for i in range(10):
            # Use varying-length intents to create distinct vectors.
            intent = f"command number {i}" + ("x" * (i * 10))
            mem.save(intent, f"cmd_{i}")

        recalled = mem.recall("command number", limit=2)
        line_count = len([l for l in recalled.strip().split("\n") if l.strip()])
        assert line_count <= 2


# -- exact_match --

class TestExactMatch:
    def test_exact_match_returns_command(self, mem):
        """Verify exact_match() returns the cached command for a near-identical query."""
        mem.save("list files", "ls -la")
        result = mem.exact_match("list files")
        assert result == "ls -la"

    def test_exact_match_returns_none_for_miss(self, tmp_path, mocker):
        """Verify exact_match() returns None when no close match exists."""
        mock_embedder = mocker.patch("dhi.agent.memory.OllamaEmbeddings")
        mock_instance = mock_embedder.return_value

        def mock_embed(text):
            vec = [0.0] * 768
            if "unrelated" in text:
                vec[0] = 1.0
            else:
                vec[1] = 1.0
            return vec

        mock_instance.embed_query.side_effect = mock_embed
        db_dir = tmp_path / "lancedb"
        mem = MemorySystem(db_path=str(db_dir))

        mem.save("list files", "ls -la")
        result = mem.exact_match("completely unrelated query here")
        assert result is None

    def test_exact_match_skips_init_record(self, mem):
        """Verify exact_match() never returns the __init__ seed record."""
        result = mem.exact_match("__init__")
        # Even if vector is close, __init__ should be filtered.
        assert result is None or result != "__init__"

    def test_exact_match_returns_none_when_no_table(self, tmp_path, mocker):
        """Verify exact_match() returns None when table is None."""
        _make_embedder(mocker)
        mocker.patch("dhi.agent.memory.lancedb.connect", side_effect=Exception("fail"))
        mem = MemorySystem(db_path=str(tmp_path / "bad"))
        assert mem.exact_match("anything") is None
