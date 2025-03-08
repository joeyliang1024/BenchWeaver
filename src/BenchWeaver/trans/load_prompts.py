import os
import json
from ..extras.constants import PROJECT_BASE_PATH

def load_prompts_from_file(
    path: str = os.path.join(PROJECT_BASE_PATH, "prompt", "translation_prompts.json")
    ) -> dict:
    """
    Load translation prompts from a file.
    """
    with open(path, "r", encoding="utf-8") as file:
        prompts = json.load(file)
    return prompts
