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
@misc{wei2023cmath,
      title={CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?}, 
      author={Tianwen Wei and Jian Luan and Wei Liu and Shuang Dong and Bin Wang},
      year={2023},
      eprint={2306.16636},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
本项目中我们提出了CMATH数据集，包括1.7k个小学水平的数学应用题和详细的注释。本数据集旨在提供一个基准工具来评估以下问题：当前流行的大模型的数学能力对应小学数学几年级的水平？我们评估了各种流行的大模型，发现只有GPT-4能通过所有六个年级的数学考试(准确率>=60%)。此外，我们通过在CMATH数据集中添加干扰信息来评估大模型的稳健性。我们的研究结果表明，GPT-4是唯一保持鲁棒性的模型。
"""

_HOMEPAGE = "https://github.com/XiaoMi/cmath"

_LICENSE = "CC-BY-4.0"

_URL = "cmath.zip"

task_list = [
    "main",
    "distractor",
]

class CMATHConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class CMATH(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CMATHConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "distractor": datasets.Value("int32"),
                "original": datasets.Value("string"),
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
        df.columns = ["question", "answer", "distractor", "original"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
