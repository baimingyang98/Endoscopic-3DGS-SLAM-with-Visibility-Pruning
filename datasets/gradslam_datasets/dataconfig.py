"""Dataset configuration loader."""
import yaml


def load_dataset_config(path: str) -> dict:
    """Load a YAML dataset configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
