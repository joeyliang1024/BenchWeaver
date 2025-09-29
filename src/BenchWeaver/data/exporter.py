import os
import shutil
import json
from typing import Literal
import pandas as pd

class DataExporter:
    train_split = False
    valid_split = False
    
    def __init__(self, 
                 name: str,
                 base_path: str = None,
                 subjects: list = None, 
                 train_split: bool = True, 
                 valid_split: bool = True,
                 ):
        self.name = name
        self.base_path = base_path if base_path else os.getcwd()
        subjects = subjects if subjects else ["all"]
        self.data = {}
        if train_split:
            self.train_split = True
            self.data["train"] = {subject: pd.DataFrame() for subject in subjects}
        if valid_split:
            self.valid_split = True
            self.data["valid"] = {subject: pd.DataFrame() for subject in subjects}
        self.data["test"] = {subject: pd.DataFrame() for subject in subjects}
        print("=" * 50)
        print(f"Initialized DataExporter for benchmark '{self.name}' with subjects:\n  {subjects}")
        print(f"Base path: {self.base_path}")
        print(f"Train split: {self.train_split}, Valid split: {self.valid_split}")
        print("-" * 50)
        print("Init data with structure:\n" + json.dumps(
            {split: {subject: df.__class__.__name__ for subject, df in subject_dict.items()} 
             for split, subject_dict in self.data.items()},
            indent=2
        ))
        print("Please populate the 'data' attribute with actual DataFrames before exporting.")
        print("Each instance for each subject should be a pandas DataFrame.")
        print("=" * 50)

    def export_data(self, fmt: Literal["csv", "parquet"]) -> "DataExporter":
        """
        Export datasets in the chosen format (csv or parquet).
        """
        if fmt not in ["csv", "parquet"]:
            raise ValueError("Invalid format. Supported formats: 'csv', 'parquet'")
        
        os.makedirs(os.path.join(self.base_path, "data"), exist_ok=True)
        
        # export Train Split
        if self.train_split:
            os.makedirs(os.path.join(self.base_path, "data", "dev"), exist_ok=True)
            for subject, df in self.data["train"].items():
                self._export_df(df, os.path.join(self.base_path, "data", "dev", f"{subject}_dev.{fmt}"), fmt)
        
        # export Valid Split
        if self.valid_split:
            os.makedirs(os.path.join(self.base_path, "data", "val"), exist_ok=True)
            for subject, df in self.data["valid"].items():
                self._export_df(df, os.path.join(self.base_path, "data", "val", f"{subject}_val.{fmt}"), fmt)
        
        # export Test Split
        os.makedirs(os.path.join(self.base_path, "data", "test"), exist_ok=True)
        for subject, df in self.data["test"].items():
            self._export_df(df, os.path.join(self.base_path, "data", "test", f"{subject}_test.{fmt}"), fmt)
        
        return self
    
    @staticmethod
    def _export_df(df: pd.DataFrame, file_path: str, fmt: Literal["csv", "parquet"]):
        # make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if fmt == "csv":
            df.to_csv(file_path, index=False)
        elif fmt == "parquet":
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError("Invalid format. Supported formats: 'csv', 'parquet'")
    
    def zip_data(self, del_source: bool = False):
        folder_path = os.path.join(self.base_path, "data")
        shutil.make_archive(
            folder_path,
            'zip',
            root_dir=self.base_path, 
            base_dir="data"
        )
        # rename data.zip to {self.name}.zip
        shutil.move(f"{folder_path}.zip", os.path.join(self.base_path, f"{self.name}.zip"))
        if del_source:
            shutil.rmtree(folder_path)
        return self
