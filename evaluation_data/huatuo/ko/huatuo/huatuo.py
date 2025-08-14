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
@misc{li2023huatuo26m,
      title={Huatuo-26M, a Large-scale Chinese Medical QA Dataset}, 
      author={Jianquan Li and Xidong Wang and Xiangbo Wu and Zhiyi Zhang and Xiaolong Xu and Jie Fu and Prayag Tiwari and Xiang Wan and Benyou Wang},
      year={2023},
      eprint={2305.01526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
We are pleased to announce the release of our evaluation dataset, a subset of the Huatuo-26M. This dataset contains 6,000 entries that we used for Natural Language Generation (NLG) experimentation in our associated research paper.

We encourage researchers and developers to use this evaluation dataset to gauge the performance of their own models. This is not only a chance to assess the accuracy and relevancy of generated responses but also an opportunity to investigate their model's proficiency in understanding and generating complex medical language.

Note: All the data points have been anonymized to protect patient privacy, and they adhere strictly to data protection and privacy regulations.
Caution: This dataset is refined and formatted by Qwen model and translated by GPT-4o to EN, ZH-TW, and KO.
"""

_HOMEPAGE = "FreedomIntelligence/huatuo26M-testdatasets"

_LICENSE = "apache-2.0"

_URL = "huatuo.zip"

task_list = [
    "all",
]

class KoHuatuoConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class KoHuatuo(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        KoHuatuoConfig(
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
        df.columns = ["question", "answer"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
