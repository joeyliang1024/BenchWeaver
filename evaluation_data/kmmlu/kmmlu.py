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
@article{son2024kmmlu,
  title={Kmmlu: Measuring massive multitask language understanding in korean},
  author={Son, Guijin and Lee, Hanwool and Kim, Sungdong and Kim, Seungone and Muennighoff, Niklas and Choi, Taekyoon and Park, Cheonbok and Yoo, Kang Min and Biderman, Stella},
  journal={arXiv preprint arXiv:2402.11548},
  year={2024}
}
"""

_DESCRIPTION = """\
We propose KMMLU, a new Korean benchmark with 35,030 expert-level multiple-choice questions across 45 subjects ranging from humanities to STEM. Unlike previous Korean benchmarks that are translated from existing English benchmarks, KMMLU is collected from original Korean exams, capturing linguistic and cultural aspects of the Korean language. We test 26 publically available and proprietary LLMs, identifying significant room for improvement. The best publicly available model achieves 50.54% on KMMLU, far below the average human performance of 62.6%. This model was primarily trained for English and Chinese, not Korean. Current LLMs tailored to Korean, such as Polyglot-Ko, perform far worse. Surprisingly, even the most capable proprietary LLMs, e.g., GPT-4 and HyperCLOVA X, achieve 59.95% and 53.40%, respectively. This suggests that further work is needed to improve Korean LLMs, and KMMLU offers the right tool to track this progress. We make our dataset publicly available on the Hugging Face Hub and integrate the benchmark into EleutherAI's Language Model Evaluation Harness.
"""

_HOMEPAGE = "https://huggingface.co/datasets/HAERAE-HUB/KMMLU"

_LICENSE = "cc-by-nd-4.0"

_URL = "kmmlu.zip"

task_list = [
    'accounting',
    'agricultural_sciences',
    'aviation_engineering_and_maintenance',
    'biology',
    'chemical_engineering',
    'chemistry',
    'civil_engineering',
    'computer_science',
    'construction',
    'criminal_law',
    'ecology',
    'economics',
    'education',
    'electrical_engineering',
    'electronics_engineering',
    'energy_management',
    'environmental_science',
    'fashion',
    'food_processing',
    'gas_technology_and_engineering',
    'geomatics',
    'health',
    'industrial_engineer',
    'information_technology',
    'interior_architecture_and_design',
    'law',
    'machine_design_and_manufacturing',
    'management',
    'maritime_engineering',
    'marketing',
    'materials_engineering',
    'mechanical_engineering',
    'nondestructive_testing',
    'patent',
    'political_science_and_sociology',
    'psychology',
    'public_safety',
    'railway_and_automotive_engineering',
    'real_estate',
    'refrigerating_machinery',
    'social_welfare',
    'taxation',
    'telecommunications_and_wireless_technology',
    'korean_history',
    'math'
]


class KMMLUConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class KMMLU(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        KMMLUConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "Category": datasets.Value("string"),
                "Human Accuracy": datasets.Value("string"), 
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
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", "val", f"{task_name}_val.csv"),
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
        df.columns = ["question", "answer", "A", "B", "C", "D", "Category", "Human Accuracy"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
