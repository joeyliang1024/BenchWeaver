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
@article{sun2019investigating,
  title={Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension},
  author={Sun, Kai and Yu, Dian and Yu, Dong and Cardie, Claire},
  journal={Transactions of the Association for Computational Linguistics},
  year={2020},
  url={https://arxiv.org/abs/1904.09679v3}
}
"""

_DESCRIPTION = """\
Machine reading comprehension tasks require a machine reader to answer questions relevant to the given document. In this paper, we present the first free-form multiple-Choice Chinese machine reading Comprehension dataset (C^3), containing 13,369 documents (dialogues or more formally written mixed-genre texts) and their associated 19,577 multiple-choice free-form questions collected from Chinese-as-a-second-language examinations.
We present a comprehensive analysis of the prior knowledge (i.e., linguistic, domain-specific, and general world knowledge) needed for these real-world problems. We implement rule-based and popular neural methods and find that there is still a significant performance gap between the best performing model (68.5%) and human readers (96.0%), especially on problems that require prior knowledge. We further study the effects of distractor plausibility and data augmentation based on translated relevant datasets for English on model performance. We expect C^3 to present great challenges to existing systems as answering 86.8% of questions requires both knowledge within and beyond the accompanying document, and we hope that C^3 can serve as a platform to study how to leverage various kinds of prior knowledge to better understand a given written or orally oriented text.
"""

_HOMEPAGE = "https://github.com/nlpdata/c3"

_LICENSE = "non-commercial research"

_URL = "c3.zip"

task_list = [
    "mixed",
    "dialogue",
]


class C3Config(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class C3(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        C3Config(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "paragraph": datasets.Value("string"),
                "question": datasets.Value("string"),
                "choices": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "idx": datasets.Value("string"),
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
        df.columns = ["paragraph", "question", "choices", "answer", "idx"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
