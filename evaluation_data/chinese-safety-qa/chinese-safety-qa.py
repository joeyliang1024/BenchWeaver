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
@misc{tan2024chinesesafetyqasafetyshortform,
      title={Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large Language Models}, 
      author={Yingshui Tan and Boren Zheng and Baihui Zheng and Kerui Cao and Huiyun Jing and Jincheng Wei and Jiaheng Liu and Yancheng He and Wenbo Su and Xiangyong Zhu and Bo Zheng},
      year={2024},
      eprint={2412.15265},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.15265}, 
}
"""

_DESCRIPTION = """\
Chinese SafetyQA is an innovative benchmark designed to evaluate the factuality ability of large language models, specifically for short-form factual questions in the Chinese safety domain.
"""

_HOMEPAGE = "https://github.com/OpenStellarTeam/ChineseSafetyQA"

_LICENSE = "MIT"

_URL = "chinese-safety-qa.zip"

task_list = [
    "theoretical_and_technical_knowledge",
    "ethical_and_moral_risks",
    "bias_and_discrimination_risks",
    "abuse_and_hate_speech_risks",
    "physical_and_mental_health_risks",
    "legal_and_regulatory_risks",
    "rumor_and_misinformation_risks"
]


class ChineseSafetyQAConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class ChineseSafetyQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ChineseSafetyQAConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "choices": datasets.Value("string"),
                "main_cate": datasets.Value("string"),
                "sub_cate": datasets.Value("string"),
                "sub_sub_cate": datasets.Value("string"),
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
        df.columns = ["question", "answer", "choices", "main_cate", "sub_cate", "sub_sub_cate"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
