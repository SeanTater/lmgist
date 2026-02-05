"""Server configuration module."""

import os

def get_config():
    """Get server configuration."""
    config = {
        'host': '0.0.0.0',
        'port': "8080",  # Bug: should be int, not string
        'debug': False,
        'workers': 4,
        'timeout': 30
    }
    return config

def validate_config(config):
    """Validate configuration."""
    required = ['host', 'port', 'debug']
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config: {key}")
    return True
