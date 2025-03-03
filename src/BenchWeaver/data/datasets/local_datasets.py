import json
from datasets import Dataset, DatasetDict

class LocalDataset:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.data = self.load_data()

    def load_data(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
        
        dataset_dict = {}
        for subject, items_list in data['data'].items():
            subset_dict = {'train': [], 'validation': [], 'test': []}
            for item in items_list:
                subset_dict[item['split']].append(item)
            
            dataset_dict[subject] = DatasetDict({
                'train': Dataset.from_list(subset_dict['train']),
                'validation': Dataset.from_list(subset_dict['validation']),
                'test': Dataset.from_list(subset_dict['test'])
            })
        
        return dataset_dict

    def get_split(self, split: str):
        split_data = {}
        for subject, dataset in self.data.items():
            if split in dataset:
                split_data[subject] = dataset[split]
            else:
                raise ValueError(f"Split '{split}' not found in the dataset for subject '{subject}'.")
        return split_data

if __name__ == "__main__":
    import os
    from pathlib import Path
    current_dir = Path(__file__).resolve().parents[4]
    example_data_path = os.path.join(current_dir, "example_data/local_test_data.json")

    dataset = LocalDataset(name="example_dataset", path=example_data_path)
    train_data = dataset.get_split('train')
    print(train_data)
