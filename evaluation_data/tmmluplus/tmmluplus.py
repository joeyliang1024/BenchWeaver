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
@article{ikala2024improved,
  title={An Improved Traditional Chinese Evaluation Suite for Foundation Model},
  author={Tam, Zhi-Rui and Pai, Ya-Ting and Lee, Yen-Wei and Cheng, Sega and Shuai, Hong-Han},
  journal={arXiv preprint arXiv:2403.01858},
  year={2024}
}
"""

_DESCRIPTION = """\
We present TMMLU+, a traditional Chinese massive multitask language understanding dataset. TMMLU+ is a multiple-choice question-answering dataset featuring 66 subjects, ranging from elementary to professional level.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ikala/tmmluplus"

_LICENSE = "MIT"

_URL = "tmmluplus.zip"

task_list = [
    'engineering_math', 
    'dentistry', 
    'traditional_chinese_medicine_clinical_medicine', 
    'clinical_psychology', 
    'technical', 
    'culinary_skills', 
    'mechanical', 
    'logic_reasoning', 
    'real_estate',
    'general_principles_of_law', 
    'finance_banking', 
    'anti_money_laundering', 
    'ttqav2', 
    'marketing_management', 
    'business_management', 
    'organic_chemistry', 
    'advance_chemistry',
    'physics', 
    'secondary_physics', 
    'human_behavior', 
    'national_protection', 
    'jce_humanities', 
    'politic_science', 
    'agriculture', 
    'official_document_management',
    'financial_analysis', 
    'pharmacy', 
    'educational_psychology', 
    'statistics_and_machine_learning', 
    'management_accounting', 
    'introduction_to_law', 
    'computer_science', 
    'veterinary_pathology',
    'accounting', 
    'fire_science', 
    'optometry', 
    'insurance_studies', 
    'pharmacology', 
    'taxation', 
    'trust_practice', 
    'geography_of_taiwan', 
    'physical_education', 
    'auditing', 
    'administrative_law',
    'education_(profession_level)', 
    'economics', 
    'veterinary_pharmacology', 
    'nautical_science', 
    'occupational_therapy_for_psychological_disorders',
    'basic_medical_science', 
    'macroeconomics', 
    'trade', 
    'chinese_language_and_literature', 
    'tve_design', 
    'junior_science_exam', 
    'junior_math_exam', 
    'junior_chinese_exam',
    'junior_social_studies', 
    'tve_mathematics', 
    'tve_chinese_language', 
    'tve_natural_sciences', 
    'junior_chemistry', 
    'music', 
    'education', 
    'three_principles_of_people',
    'taiwanese_hokkien'
]


class MMLUConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class MMLU(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MMLUConfig(
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
                    "filepath": os.path.join(
                        data_dir, "test", f"{task_name}_test.csv"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "val", f"{task_name}_val.csv"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "dev", f"{task_name}_dev.csv"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = ["question", "A", "B", "C", "D", "answer"]

        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
