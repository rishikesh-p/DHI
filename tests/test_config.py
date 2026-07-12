import pytest
import json
import os
import stat
from dhi.config import load_config, save_config, DEFAULT_CONFIG


# -- Fixtures --

@pytest.fixture
def isolated_config(tmp_path, mocker):
    """Redirect CONFIG_FILE to a temporary directory for test isolation."""
    config_file = tmp_path / "config.json"
    mocker.patch("dhi.config.CONFIG_FILE", str(config_file))
    mocker.patch("dhi.config.CONFIG_DIR", str(tmp_path))
    return config_file


# -- load_config --

class TestLoadConfig:
    def test_creates_default_file_when_missing(self, isolated_config):
        """Verify load_config creates the config file with defaults if absent."""
        assert not isolated_config.exists()
        loaded = load_config()
        assert isolated_config.exists()
        assert loaded == DEFAULT_CONFIG

    def test_merges_missing_keys_with_defaults(self, isolated_config):
        """Verify new keys added to DEFAULT_CONFIG are backfilled into existing configs."""
        partial = {"local_model": "custom:latest"}
        isolated_config.write_text(json.dumps(partial))

        loaded = load_config()

        assert loaded["local_model"] == "custom:latest"
        for key in DEFAULT_CONFIG:
            assert key in loaded, f"Missing key after merge: {key}"

    def test_returns_defaults_on_corrupted_json(self, isolated_config):
        """Verify load_config gracefully handles a corrupted config file."""
        isolated_config.write_text("{invalid json!!")
        loaded = load_config()
        assert loaded == DEFAULT_CONFIG

    def test_preserves_extra_user_keys(self, isolated_config):
        """Verify load_config does not strip user-added keys."""
        custom = {**DEFAULT_CONFIG, "custom_key": "custom_value"}
        isolated_config.write_text(json.dumps(custom))

        loaded = load_config()
        assert loaded["custom_key"] == "custom_value"


# -- save_config --

class TestSaveConfig:
    def test_roundtrip_preserves_values(self, isolated_config):
        """Verify save -> load roundtrip preserves all values."""
        custom_cfg = {
            "local_model": "test_model:latest",
            "stateful_local": True,
            "require_confirmation": False,
            "cloud_provider": "google",
            "cloud_api_key": "sk-test-key-123"
        }
        save_config(custom_cfg)
        loaded = load_config()

        for key, value in custom_cfg.items():
            assert loaded[key] == value

    def test_file_permissions_are_restricted(self, isolated_config):
        """Verify the config file is created with 0600 permissions (user R/W only)."""
        save_config(DEFAULT_CONFIG)
        mode = os.stat(str(isolated_config)).st_mode
        assert stat.S_IMODE(mode) == 0o600

    def test_overwrites_existing_config(self, isolated_config):
        """Verify save_config replaces the entire file, not appends."""
        save_config({"local_model": "first"})
        save_config({"local_model": "second"})

        loaded = load_config()
        assert loaded["local_model"] == "second"
