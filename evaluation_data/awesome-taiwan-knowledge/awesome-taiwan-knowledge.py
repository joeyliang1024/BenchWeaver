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


_CITATION = """
@article{DBLP:journals/corr/abs-2311-17487,
  author       = {Yen{-}Ting Lin and
                  Yun{-}Nung Chen},
  title        = {Taiwan {LLM:} Bridging the Linguistic Divide with a Culturally Aligned
                  Language Model},
  journal      = {CoRR},
  volume       = {abs/2311.17487},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2311.17487},
  doi          = {10.48550/ARXIV.2311.17487},
  eprinttype    = {arXiv},
  eprint       = {2311.17487},
  timestamp    = {Tue, 05 Dec 2023 14:40:42 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2311-17487.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@article{DBLP:journals/corr/abs-2403-20180,
  author       = {Po{-}Heng Chen and
                  Sijia Cheng and
                  Wei{-}Lin Chen and
                  Yen{-}Ting Lin and
                  Yun{-}Nung Chen},
  title        = {Measuring Taiwanese Mandarin Language Understanding},
  journal      = {CoRR},
  volume       = {abs/2403.20180},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2403.20180},
  doi          = {10.48550/ARXIV.2403.20180},
  eprinttype    = {arXiv},
  eprint       = {2403.20180},
  timestamp    = {Wed, 10 Apr 2024 17:37:45 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2403-20180.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
GPQA is a multiple-choice, Q&A dataset of very hard questions written and validated by experts in biology, physics, and chemistry. When attempting questions out of their own domain (e.g., a physicist answers a chemistry question), these experts get only 34% accuracy, despite spending >30m with full access to Google.

We request that you do not reveal examples from this dataset in plain text or images online, to reduce the risk of leakage into foundation model training corpora.
"""

_HOMEPAGE = "https://github.com/MiuLab/Taiwan-LLM/blob/main/evaluation/TTQA_1.0.0_tw_llama_v1.0.json"

_LICENSE = "Apache-2.0 License"

_URL = "awesome-taiwan-knowledge.zip"

task_list = [
    "all",
]

class AwesomeTaiwanKnowledgeConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class AwesomeTaiwanKnowledge(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        AwesomeTaiwanKnowledgeConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "title": datasets.Value("string"),
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
        df.columns = ["title", "question", "answer"]

        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
