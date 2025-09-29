import os
from typing import List
import datasets
import pandas as pd

_CITATION = """\
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""

_DESCRIPTION = """\
MATH dataset from https://github.com/hendrycks/math (Measuring Mathematical Problem Solving With the MATH Dataset).
"""

_HOMEPAGE = "https://huggingface.co/datasets/EleutherAI/hendrycks_math"
_LICENSE = "CC BY 4.0"
_URL = "math.zip"

task_list = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus"
]

class MATHConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class MATH(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [MATHConfig(name=task) for task in task_list]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'level': datasets.Value(dtype='string'),
                'question': datasets.Value(dtype='string'),
                'answer': datasets.Value(dtype='string'),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", "test", f"{task_name}_test.parquet"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", "dev", f"{task_name}_dev.parquet"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        # Read in with pandas so that 'kwargs' remains a list of dicts
        df = pd.read_parquet(filepath)
        # Ensure the dataframe columns are as expected
        assert list(df.columns) == ["level", "question", "answer"]
        for idx, example in enumerate(df.to_dict(orient="records")):
            yield idx, example