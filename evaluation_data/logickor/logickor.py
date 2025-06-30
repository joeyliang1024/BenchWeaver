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
@article{logickor,
  title={LogicKor},
  doi={doi:10.57967/hf/2440}
  author={Jeonghwan Park},
  year={2024},
  url={https://github.com/instructkr/LogicKor}
}

"""

_DESCRIPTION = """\
LogicKor는 한국어 언어모델 다양한 분야에서의 사고력을 측정하기위해 구성된 LLM-as-a-judge 방식의 멀티턴 벤치마크 데이터셋입니다. 본 데이터셋은 6가지(추론, 수학, 글쓰기, 코딩, 이해, 국어)의 카테고리의 멀티턴 프롬프트 총 42개로 구성되어있습니다.

추론 및 평가 코드는 https://github.com/instructkr/LogicKor 저장소를 참고해주세요.
"""

_HOMEPAGE = "https://github.com/instructkr/LogicKor"

_LICENSE = "None"

_URL = "logickor.zip"

task_list = [
    'Reasoning',
    'Math',
    'Writing',
    'Coding',
    'Understanding',
    'Grammar',
]

class MTBenchTWConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class MTBenchTW(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MTBenchTWConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question_id": datasets.Value("string"),
                "question_turns": datasets.Value("string"),
                "answer_turns": datasets.Value("string"),
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
        df.columns = ["question_id", "question_turns", "answer_turns"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
