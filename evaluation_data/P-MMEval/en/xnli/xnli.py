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
_URL = "xnli.zip"

task_list = ["all"]

class XNLIConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class XNLI(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        XNLIConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "premise": datasets.Value("string"),
                "statement": datasets.Value("string"),
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
                    "filepath": os.path.join(data_dir, "data", "test", f"{task_name}_test.parquet"),
                },
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_parquet(filepath)
        df.columns = ["premise", "statement", "answer"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
