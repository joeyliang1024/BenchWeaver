# coding=utf-8
# Copyright 2024 The P-MMEval Authors.
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
from typing import List
import datasets
import pandas as pd

_CITATION = """\
@misc{zhang2024pmmevalparallelmultilingualmultitask,
      title={P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs}, 
      author={Yidan Zhang and Yu Wan and Boyi Deng and Baosong Yang and Haoran Wei and Fei Huang and Bowen Yu and Junyang Lin and Fei Huang and Jingren Zhou},
      year={2024},
      eprint={2411.09116},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.09116}, 
}
"""

_DESCRIPTION = """\
We introduce a multilingual benchmark, P-MMEval, covering effective fundamental and capability-specialized datasets. We extend the existing benchmarks, ensuring consistent language coverage across all datasets and providing parallel samples among multiple languages, supporting up to 10 languages from 8 language families (i.e., en, zh, ar, es, ja, ko, th, fr, pt, vi). As a result, P-MMEval facilitates a holistic assessment of multilingual capabilities and comparative analysis of cross-lingual transferability.
"""

_HOMEPAGE = "https://huggingface.co/datasets/Qwen/P-MMEval"
_LICENSE = "Apache-2.0"
_URL = "humaneval-xl.zip"

task_list = ["all"]

class HumanEvalXLConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class HumanEvalXL(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [HumanEvalXLConfig(name=task) for task in task_list]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'task_id': datasets.Value(dtype='string'),
                'text': datasets.Value(dtype='string'),
                'test_list': datasets.Sequence(datasets.Value(dtype='string')),
                'prompt': datasets.Value(dtype='string'),
                'test': datasets.Value(dtype='string'),
                'entry_point': datasets.Value(dtype='string'),
                'description': datasets.Value(dtype='string'),
                'language': datasets.Value(dtype='string'),
                'canonical_solution': datasets.Value(dtype='string'),
                'declaration': datasets.Value(dtype='string'),
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
            )
        ]

    def _generate_examples(self, filepath):
        # Read in with pandas so that 'kwargs' remains a list of dicts
        df = pd.read_parquet(filepath)
        # Ensure the dataframe columns are as expected
        df.columns = ["task_id", "text", "test_list", "prompt", "test", "entry_point", "description", "language", "canonical_solution", "declaration"]
        for idx, example in enumerate(df.to_dict(orient="records")):
            yield idx, example
