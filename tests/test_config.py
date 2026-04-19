import pytest
import os
from dhi.config import load_config, save_config

def test_config_save_load(tmp_path, mocker):
    config_file = tmp_path / "config.json"
    mocker.patch("dhi.config.CONFIG_FILE", str(config_file))
    
    # Save a custom config
    custom_cfg = {
        "local_model": "test_model:latest",
        "stateful_local": True,
        "require_confirmation": False
    }
    save_config(custom_cfg)
    
    # Ensure it actually wrote the file
    assert os.path.exists(str(config_file))
    
    # Load and verify
    loaded = load_config()
    assert loaded["local_model"] == "test_model:latest"
    assert loaded["stateful_local"] is True

def test_config_default_fallback(tmp_path, mocker):
    config_file = tmp_path / "missing_config.json"
    mocker.patch("dhi.config.CONFIG_FILE", str(config_file))
    
    # load_config should create it with defaults if missing
    loaded = load_config()
    assert loaded["local_model"] == "qwen3.5:4b"
    assert os.path.exists(str(config_file))
