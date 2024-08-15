import yaml
import json
import toml
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import re
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

class AdvancedConfigParser:
    def __init__(self, config_file: Union[str, Path], env_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self.env_file = Path(env_file) if env_file else None
        self._load_env()

    def _load_env(self):
        if self.env_file and self.env_file.exists():
            load_dotenv(self.env_file)
        load_dotenv()  # Load from .env file in current directory if exists

    def load_config(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        loaders = {
            '.yaml': self._load_yaml,
            '.yml': self._load_yaml,
            '.json': self._load_json,
            '.toml': self._load_toml
        }

        loader = loaders.get(self.config_file.suffix.lower())
        if not loader:
            raise ValueError(f"Unsupported configuration file format: {self.config_file.suffix}")

        self.config = loader()
        self._substitute_env_vars(self.config)
        return self.config

    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def _load_json(self) -> Dict[str, Any]:
        with open(self.config_file, 'r') as file:
            return json.load(file)

    def _load_toml(self) -> Dict[str, Any]:
        with open(self.config_file, 'r') as file:
            return toml.load(file)

    def _substitute_env_vars(self, config: Dict[str, Any]):
        for key, value in config.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str):
                config[key] = os.path.expandvars(value)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

    def save_config(self, output_file: Optional[Union[str, Path]] = None) -> None:
        output_file = Path(output_file) if output_file else self.config_file

        savers = {
            '.yaml': self._save_yaml,
            '.yml': self._save_yaml,
            '.json': self._save_json,
            '.toml': self._save_toml
        }

        saver = savers.get(output_file.suffix.lower())
        if not saver:
            raise ValueError(f"Unsupported configuration file format: {output_file.suffix}")

        saver(output_file)

    def _save_yaml(self, output_file: Path) -> None:
        with open(output_file, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def _save_json(self, output_file: Path) -> None:
        with open(output_file, 'w') as file:
            json.dump(self.config, file, indent=2)

    def _save_toml(self, output_file: Path) -> None:
        with open(output_file, 'w') as file:
            toml.dump(self.config, file)

    def validate_config(self, schema: Dict[str, Any]) -> None:
        try:
            validate(instance=self.config, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Config validation failed: {e}")

    def merge_config(self, other_config: Dict[str, Any]) -> None:
        def deep_merge(source, destination):
            for key, value in source.items():
                if isinstance(value, dict):
                    node = destination.setdefault(key, {})
                    deep_merge(value, node)
                else:
                    destination[key] = value
            return destination

        self.config = deep_merge(other_config, self.config)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.config)

    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        self.config = config_dict

    def get_all_keys(self, prefix: str = '') -> List[str]:
        keys = []
        for key, value in self.config.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                keys.extend(self.get_all_keys(new_key))
            else:
                keys.append(new_key)
        return keys

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __str__(self) -> str:
        return json.dumps(self.config, indent=2)

    def __repr__(self) -> str:
        return f"AdvancedConfigParser(config_file='{self.config_file}')"

# Example usage
if __name__ == "__main__":
    # Create a sample config file
    sample_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "${DB_USER}",
            "password": "${DB_PASS}"
        },
        "app": {
            "debug": True,
            "log_level": "INFO"
        }
    }

    with open("sample_config.yaml", "w") as f:
        yaml.dump(sample_config, f)

    # Create a .env file
    with open(".env", "w") as f:
        f.write("DB_USER=admin\nDB_PASS=secret")

    # Use the AdvancedConfigParser
    parser = AdvancedConfigParser("sample_config.yaml", env_file=".env")
    config = parser.load_config()

    print("Loaded config:")
    print(parser)

    print("\nAccessing nested config:")
    print(f"Database host: {parser['database.host']}")
    print(f"App debug mode: {parser.get('app.debug')}")

    print("\nModifying config:")
    parser.set("app.log_level", "DEBUG")
    print(f"Updated log level: {parser['app.log_level']}")

    print("\nAll config keys:")
    print(parser.get_all_keys())

    # Validate config
    schema = {
        "type": "object",
        "properties": {
            "database": {"type": "object"},
            "app": {"type": "object"}
        },
        "required": ["database", "app"]
    }
    parser.validate_config(schema)

    print("\nConfig validation passed")

    # Save to a different format
    parser.save_config("sample_config.json")
    print("\nConfig saved as JSON")

    # Clean up
    os.remove("sample_config.yaml")
    os.remove("sample_config.json")
    os.remove(".env")

    print("\nAdvanced Config Parser demonstration completed!")