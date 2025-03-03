import os
import sys
import json
import aiohttp
import concurrent.futures
from typing import List
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

class HFDataset:
    def __init__(self, dataset_name, num_workers=4, timeout: int = 60 * 60, token=None):
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.token = token
        self.timeout = timeout
        self.dataset = {task: {} for task in get_dataset_config_names(dataset_name)}

    @property
    def features(self, task, split):
        return self.dataset[task][split].features

    def get_subsets(self) -> List[str]:
        subsets = get_dataset_config_names(self.dataset_name)
        print("Subsets:")
        print(json.dumps(subsets, indent=2))
        return subsets

    def get_splits(self, task: str) -> List[str]:
        return get_dataset_split_names(self.dataset_name, config_name=task, token=self.token)

    def load_split(self, task: str, split: str):
        try:
            if split in self.get_splits(task):
                self.dataset[task][split] = load_dataset(
                    self.dataset_name,
                    task,
                    split=split,
                    token=self.token,
                    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=self.timeout)}}
                )
            else:
                self.dataset[task][split] = None
        except Exception as e:
            print(f"Error loading {task} - {split}: {e}")
            self.dataset[task][split] = None

    def load_task(self, task: str) -> str:
        splits = ["train", "validation", "test"]
        with tqdm(total=len(splits), dynamic_ncols=True, file=sys.stdout, desc=f"Loading {task}") as pbar:
            for split in splits:
                self.load_split(task, split)
                pbar.update(1)
        return task

    def load_data(self):
        subsets = self.get_subsets()
        with tqdm(total=len(subsets), dynamic_ncols=True, position=1, desc="Loading Data") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.load_task, subset) for subset in subsets]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        loaded_subset = future.result()
                        pbar.set_postfix_str(loaded_subset)
                        pbar.update(1)
                except concurrent.futures.CancelledError:
                    for future in futures:
                        future.cancel()
                        
if __name__ == "__main__":
    # Define the dataset name
    dataset_name = "ikala/tmmluplus"  # Replace with your dataset
    datamodule = HFDataset(dataset_name)

    # Run the asynchronous load_data method
    datamodule.load_data()
    print(datamodule.dataset.keys())
