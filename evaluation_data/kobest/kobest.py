# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import datasets
import pandas as pd


_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2204.04541,
  doi = {10.48550/ARXIV.2204.04541},
  url = {https://arxiv.org/abs/2204.04541},
  author = {Kim, Dohyeong and Jang, Myeongjun and Kwon, Deuk Sin and Davis, Eric},
  title = {KOBEST: Korean Balanced Evaluation of Significant Tasks},
  publisher = {arXiv},
  year = {2022},
}
"""

_DESCRIPTION = """\
# Dataset Summary
KoBEST is a Korean benchmark suite consists of 5 natural language understanding tasks that requires advanced knowledge in Korean.
# Supported Tasks and Leaderboards
Boolean Question Answering, Choice of Plausible Alternatives, Words-in-Context, HellaSwag, Sentiment Negation Recognition
"""

_HOMEPAGE = "https://huggingface.co/datasets/skt/kobest_v1"

_LICENSE = "cc-by-sa-4.0"

_URL = "kobest.zip"

task_list = [
    'boolq',
    'copa',
    'hellaswag',
    'sentineg',
    'wic',
]

class KoBestConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class KoBest(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        KoBestConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "paragraph": datasets.Value("string"),
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
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
                    "filepath": os.path.join(data_dir, "data", "test", f"{task_name}_test.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", "val", f"{task_name}_val.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", "dev", f"{task_name}_dev.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = ["paragraph", "question", "A", "B", "C", "D", "answer"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
