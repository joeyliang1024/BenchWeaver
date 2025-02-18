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
@misc{lin2021truthfulqa,
    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
    author={Stephanie Lin and Jacob Hilton and Owain Evans},
    year={2021},
    eprint={2109.07958},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

"""

_DESCRIPTION = """\
TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts.
"""

_HOMEPAGE = "https://huggingface.co/datasets/truthfulqa/truthful_qa"

_LICENSE = "MIT"

_URL = "truthfulqa.zip"

task_list = [
    "merge",
]

class TRUTHFULQAConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class TRUTHFULQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TRUTHFULQAConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                'category': datasets.Value("string"), 
                'question': datasets.Value("string"),
                'answer': datasets.Value("string"),
                'mc1_choices': datasets.Value("string"),
                'mc1_labels': datasets.Value("string"),
                'mc2_choices': datasets.Value("string"),
                'mc2_labels': datasets.Value("string"),
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
        df.columns = ['category', 'question', "answer", 'mc1_choices', 'mc1_labels', 'mc2_choices', 'mc2_labels']
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
