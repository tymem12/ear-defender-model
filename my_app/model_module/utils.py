from typing import Dict
import yaml


def load_config(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        return config
