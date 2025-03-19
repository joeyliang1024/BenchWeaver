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
@misc{kim2024click,
      title={CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean}, 
      author={Eunsu Kim and Juyoung Suk and Philhoon Oh and Haneul Yoo and James Thorne and Alice Oh},
      year={2024},
      eprint={2403.06412},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

"""

_DESCRIPTION = """\
CLIcK (Cultural and Linguistic Intelligence in Korean) is a comprehensive dataset designed to evaluate cultural and linguistic intelligence in the context of Korean language models. In an era where diverse language models are continually emerging, there is a pressing need for robust evaluation datasets, especially for non-English languages like Korean. CLIcK fills this gap by providing a rich, well-categorized dataset focusing on both cultural and linguistic aspects, enabling a nuanced assessment of Korean language models.
"""

_HOMEPAGE = "https://huggingface.co/datasets/EunsuKim/CLIcK"

_LICENSE = "Not specified"

_URL = "click.zip"

task_list = [
    'PSAT', 
    'KHB', 
    'Kedu', 
    'PSE', 
    'KIIP', 
    'TK', 
    'CSAT',
]

class CLIcKConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class CLIcK(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CLIcKConfig(
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
        df.columns = ["paragraph", "question", "choices", "answer", "categories"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
