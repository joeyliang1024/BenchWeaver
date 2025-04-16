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
@misc{hsu2023advancing,
    title={Advancing the Evaluation of Traditional Chinese Language Models: Towards a Comprehensive Benchmark Suite}, 
    author={Chan-Jan Hsu and Chang-Le Liu and Feng-Ting Liao and Po-Chun Hsu and Yi-Chang Chen and Da-shan Shiu},
    year={2023},
    eprint={2309.08448},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
MT-Bench-TW is a Traditional Chinese version of MT-bench, which is a series of open-ended questions that evaluate a chatbot’s multi-turn conversational and instruction-following ability. MT-Bench-TW inherits the categorization of MT-Bench, which includes a wide variety of core capabilities, such as reasoning and writing.
"""

_HOMEPAGE = "https://huggingface.co/datasets/MediaTek-Research/TCEval-v2"

_LICENSE = "None"

_URL = "mt-bench-tw.zip"

task_list = [
    'writing',
    'roleplay',
    'reasoning',
    'math',
    'coding',
    'extraction',
    'stem',
    'humanities'
]

class MTBenchTWConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class MTBenchTW(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MTBenchTWConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question_id": datasets.Value("string"),
                "question_turns": datasets.Value("string"),
                "answer_turns": datasets.Value("string"),
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
        df.columns = ["question_id", "question_turns", "answer_turns"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
