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
@article{suzgun2022challenging,
  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
"""

_DESCRIPTION = """\
BIG-Bench (Srivastava et al., 2022) is a diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models. Language models have already made good progress on this benchmark, with the best model in the BIG-Bench paper outperforming average reported human-rater results on 65% of the BIG-Bench tasks via few-shot prompting. But on what tasks do language models fall short of average human-rater performance, and are those tasks actually unsolvable by current language models
"""

_HOMEPAGE = "https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main"

_LICENSE = "MIT"

_URL = "big-bench-hard.zip"

task_list = [
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "object_counting",
    "penguins_in_a_table",
    "salient_translation_error_detection",
    "tracking_shuffled_objects_five_objects",
]

class TRUTHFULQAConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class TRUTHFULQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TRUTHFULQAConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                'question': datasets.Value("string"),
                'answer': datasets.Value("string"),
                'choices': datasets.Value("string"),
                'labels': datasets.Value("string"),
                'explanation': datasets.Value("string"),
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
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = ['question', 'answer', 'choices', 'labels', 'explanation']
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
