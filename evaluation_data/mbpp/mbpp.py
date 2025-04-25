import os
from typing import List
import datasets
import pandas as pd

_CITATION = """\
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""

_DESCRIPTION = """\
The benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases. As described in the paper, a subset of the data has been hand-verified by us.
"""

_HOMEPAGE = "https://github.com/google-research/google-research/tree/master/mbpp"
_LICENSE = "CC BY 4.0"
_URL = "mbpp.zip"

task_list = ["full"]

class MBPPConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class MBPP(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [MBPPConfig(name=task) for task in task_list]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'task_id': datasets.Value(dtype='int32'),
                'text': datasets.Value(dtype='string'),
                'code': datasets.Value(dtype='string'),
                'test_list': datasets.Sequence(datasets.Value(dtype='string')),
                'test_setup_code': datasets.Value(dtype='string'),
                'challenge_test_list': datasets.Sequence(datasets.Value(dtype='string')),
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
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", "val", f"{task_name}_val.parquet"),
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
        df.columns = ['task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list']
        for idx, example in enumerate(df.to_dict(orient="records")):
            yield idx, example
