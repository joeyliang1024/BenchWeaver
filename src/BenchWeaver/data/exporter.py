import os
import shutil
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
        
        
    def export(self) -> "DataExporter":
        os.makedirs(os.path.join(self.base_path, "data"), exist_ok=True)
        # export Train Split
        if self.train_split:
            os.makedirs(os.path.join(self.base_path, "data", "dev"), exist_ok=True)
            for subject, df in self.data["train"].items():
                self.export_csv(df, os.path.join(self.base_path, "data", "dev", f"{subject}_dev.csv"))
        # export Valid Split
        if self.valid_split:
            os.makedirs(os.path.join(self.base_path, "data", "val"), exist_ok=True)
            for subject, df in self.data["valid"].items():
                self.export_csv(df, os.path.join(self.base_path, "data", "val", f"{subject}_val.csv"))
        # export Test Split
        os.makedirs(os.path.join(self.base_path, "data", "test"), exist_ok=True)
        for subject, df in self.data["test"].items():
            self.export_csv(df, os.path.join(self.base_path, "data", "test", f"{subject}_test.csv"))
        return self
    
    @staticmethod
    def export_csv(df: pd.DataFrame, file_path: str):
        # make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    
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
    

