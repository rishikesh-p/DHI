import json
import os

CONFIG_DIR = os.path.expanduser("~/.config/dhi")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_CONFIG = {
    "local_model": "qwen3.5:4b",
    "stateful_local": False,
    "require_confirmation": True,
    "cloud_provider": "google",
    "cloud_api_key": ""
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        os.chmod(CONFIG_FILE, 0o600)  # Secure the file (User R/W only)
        return DEFAULT_CONFIG
        
    try:
        with open(CONFIG_FILE, 'r') as f:
            user_config = json.load(f)
            # Merge with default to ensure new keys exist
            for key, value in DEFAULT_CONFIG.items():
                if key not in user_config:
                    user_config[key] = value
            return user_config
    except (json.JSONDecodeError, FileNotFoundError):
        return DEFAULT_CONFIG

def save_config(config_dict):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_dict, f, indent=4)
    os.chmod(CONFIG_FILE, 0o600)  # Secure the file (User R/W only)
