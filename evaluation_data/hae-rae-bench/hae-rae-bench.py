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
@misc{polyglot-ko,
  title = {{Polyglot-Ko: Open-Source Korean Autoregressive Language Model}},
  author = {Ko, Hyunwoong and Yang, Kichang and Ryu, Minho and Choi, Taekyoon and Yang, Seungmu and Hyun, jiwung and Park, Sungho},
  url = {https://www.github.com/eleutherai/polyglot},
  month = {9},
  year = {2022},
}
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
@misc{kullm,
  author = {NLP & AI Lab and Human-Inspired AI research},
  title = {KULLM: Korea University Large Language Model Project},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nlpai-lab/kullm}},
}
"""

_DESCRIPTION = """\
The HAE_RAE_BENCH 1.1 is an ongoing project to develop a suite of evaluation tasks designed to test the understanding of models regarding Korean cultural and contextual nuances. Currently, it comprises 13 distinct tasks, with a total of 4900 instances.
"""

_HOMEPAGE = "https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.1"

_LICENSE = "cc-by-nc-nd-4.0"

_URL = "hae-rae-bench.zip"

task_list = ['lyrics_denoising', 
             'proverbs_denoising',
             'correct_definition_matching', 
             'csat_geo', 
             'csat_law', 
             'csat_socio', 
             'date_understanding', 
             'general_knowledge', 
             'history', 
             'loan_words', 
             'rare_words', 
             'standard_nomenclature', 
             'reading_comprehension'
             ]


class HAE_RAE_BENCHConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class HAE_RAE_BENCH(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        HAE_RAE_BENCHConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "E": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "categories": datasets.Value("string"),
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
        df.columns = ["question", "A", "B", "C", "D", "E", "answer", "categories"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
