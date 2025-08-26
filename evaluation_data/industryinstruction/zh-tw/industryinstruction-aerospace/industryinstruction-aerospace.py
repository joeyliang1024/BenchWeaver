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
@misc {beijing_academy_of_artificial_intelligence,
    author= { Xiaofeng Shi and Lulu Zhao and Hua Zhou and Donglin Hao and Yonghua Lin },
    title = { IndustryInstruction},
    year  = 2024,
    url   = { https://huggingface.co/datasets/BAAI/IndustryInstruction },
    doi   = { 10.57967/hf/3487 },
    publisher = { Hugging Face }
}
"""

_DESCRIPTION = """\
本数据集为行业指令数据集，目前包含的行业中英文对照名称如下，本次数据旨在补充当前行业指令数据的空白，并挖掘BAAI/IndustryCorpus2预训练数据集中高质量预训练语料中包含的行业高价值知识。
汽车 : Automobiles
航空航天 : Aerospace
人工智能_机器学习 : Artificial-Intelligence
交通运输 : Transportation
科技_科学研究 : Technology-Research
法律_司法 : Law-Justice
金融_经济 : Finance-Economics
文学_情感 : Literature-Emotions
旅游_地理 : Travel-Geography
住宿_餐饮_酒店 : Hospitality-Catering
医疗 : Health-Medicine
学科教育 : Subject-Education
"""

_HOMEPAGE = "https://huggingface.co/datasets/BAAI/IndustryInstruction_Aerospace"

_LICENSE = "apache-2.0"

_URL = "industryinstruction.zip"

task_list = [
    "all",
]

class ZhTwAerospaceConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class ZhTwAerospace(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ZhTwAerospaceConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
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
                    "filepath": os.path.join(data_dir, "data", "test", f"{task_name}_test.csv"),
                },
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = ["question", "answer"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
