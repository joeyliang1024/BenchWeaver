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
_URL = "mifeval.zip"

task_list = ["all"]

class MIFEvalConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class MIFEval(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [MIFEvalConfig(name=task) for task in task_list]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "key": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "instruction_id_list": datasets.Sequence(datasets.Value("string")),
                "kwargs": datasets.Features({
                        "num_highlights":      datasets.Value("float64"),
                        "relation":            datasets.Value("string"),
                        "num_words":           datasets.Value("float64"),
                        "num_placeholders":    datasets.Value("float64"),
                        "prompt_to_repeat":    datasets.Value("string"),
                        "num_bullets":         datasets.Value("float64"),
                        "section_spliter":     datasets.Value("string"),
                        "num_sections":        datasets.Value("float64"),
                        "capital_relation":    datasets.Value("string"),
                        "capital_frequency":   datasets.Value("float64"),
                        "keywords":            datasets.Sequence(datasets.Value("string")),
                        "num_paragraphs":      datasets.Value("float64"),
                        "language":            datasets.Value("string"),
                        "let_relation":        datasets.Value("string"),
                        "letter":              datasets.Value("string"),
                        "let_frequency":       datasets.Value("float64"),
                        "end_phrase":          datasets.Value("string"),
                        "forbidden_words":     datasets.Sequence(datasets.Value("string")),
                        "keyword":             datasets.Value("string"),
                        "frequency":           datasets.Value("float64"),
                        "num_sentences":       datasets.Value("float64"),
                        "postscript_marker":   datasets.Value("string"),
                        "first_word":          datasets.Value("string"),
                        "nth_paragraph":       datasets.Value("float64"),
                    })
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        path = os.path.join(data_dir, "data", "test", f"{self.config.name}_test.parquet")
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": path})
        ]

    def _generate_examples(self, filepath):
        # Read in with pandas so that 'kwargs' remains a list of dicts
        df = pd.read_parquet(filepath)
        # Ensure the dataframe columns are as expected
        df.columns = ["key", "question", "instruction_id_list", "kwargs"]
        def format_new_dict(x: List[dict]):
            record = {}
            for _dict in x:
                for key, value in _dict.items():
                    if key not in record:
                        if key in ["keywords", "forbidden_words"]:
                            record[key] = [_ for _ in value] if value is not None else None
                        record[key] = value
                    
                    if value is not None:
                        if key in ["keywords", "forbidden_words"]:
                            record[key] = [_ for _ in value]
                        record[key] = value
            return record
        df['kwargs'] = df['kwargs'].apply(format_new_dict)
        for idx, example in enumerate(df.to_dict(orient="records")):
            yield idx, example