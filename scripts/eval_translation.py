import os
import json
import fire
import logging
from typing import Literal, Union
from BenchWeaver.extras.constants import PROJECT_BASE_PATH
from BenchWeaver.inference.client import Client

logger = logging.getLogger(__name__)   

def load_translation_data(origin_lang_dir, target_lang_dir, data_type: Literal['question', 'answer']):
    """
    Load translation data from the specified directories.
    """
    pass

def export_data(data: Union[list, dict], output_dir:str) -> None:
    """
    Export the data to json format.
    """
    assert isinstance(data, (list, dict)), "Data must be a list or a dictionary."
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Data exported to {output_dir}")
    
def main(
    
    ):
    CHECK_MODEL_NAME = "gpt-4o"
    
    client = Client(
        mode='api',
        host_name='localhost',
        port=8000,
        model_path=None,
        model_name=None,
        max_model_len=16384,
        openai_source="azure",
    )
    # evaluate question quality

if __name__ == "__main__":
    fire.Fire(main)