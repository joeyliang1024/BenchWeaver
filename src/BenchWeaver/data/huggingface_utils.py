import os
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from ..extras.constants import PROJECT_BASE_PATH

def _check_huggingface_repo_exists(repo_id: str, token: str | None = None, repo_type: str = "dataset") -> bool:
    """
    Check whether a Hugging Face repo (model, dataset, or space) exists.
    Works for both public and private repos when a token is provided.

    Args:
        repo_id (str): e.g. "meta-llama/Llama-2-7b-hf"
        token (str, optional): Hugging Face access token. Required for private repos.
        repo_type (str): One of "model", "dataset", or "space".

    Returns:
        bool: True if repo exists (and accessible), False otherwise.
    """
    api = HfApi(token=token)
    try:
        repo_id = "/".join(repo_id.split("/")[-2:]) # keep only the last two segments
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            return False  # repo not found or not accessible
        raise e  # raise other errors (auth, rate limit, etc.)

def load_hf_or_local_dataset(path:str, task_name:str, token: str | None = None, **kwargs):
    """
    Load a dataset from Hugging Face Hub if it exists there; 
    otherwise, load it from a local path.

    Args:
        path (str): Dataset path or name on Hugging Face Hub or local filesystem.
        task_name (str): Specific configuration or subset of the dataset to load.
        token (str, optional): Hugging Face access token (for private datasets).
        **kwargs: Extra arguments to pass to `datasets.load_dataset()`.

    Returns:
        DatasetDict or Dataset
    """
    # Check if the dataset exists on Hugging Face
    exists_on_hf = _check_huggingface_repo_exists(path, token=token, repo_type="dataset")

    if exists_on_hf:
        print(f"✅ Loading dataset from Hugging Face Hub: {path}")
        dataset = load_dataset(path=path, token=token, trust_remote_code=True)
    else:
        print(f"📂 Loading dataset from local path: {path}")
        local_path = os.path.join(PROJECT_BASE_PATH, path, task_name)
        dataset = load_dataset(path=local_path, **kwargs)
    return dataset
