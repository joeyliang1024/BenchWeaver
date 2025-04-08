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

"""

_DESCRIPTION = """\
歡迎來到 taide-bench，本 project 以 GPT-4 評估 LLM 的辦公室任務，例如: 中翻英、英翻中、摘要、寫文章、寫信等。 
"""

_HOMEPAGE = "https://huggingface.co/datasets/taide/taide-bench"

_LICENSE = "MIT"

_URL = "taide-bench.zip"

task_list = [
    "letter",
    "essay",
    "summary",
    "zh2en",
    "en2zh",
]

class TaideBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class TaideBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TaideBenchConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "task": datasets.Value("string"),
                "qid": datasets.Value("string"),
                "question": datasets.Value("string"),
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
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = ["task", "qid", "question", "answer"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
