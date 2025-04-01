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
@misc{li2023cmmlu,
      title={CMMLU: Measuring massive multitask language understanding in Chinese}, 
      author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
      year={2023},
      eprint={2306.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
CMMLU is a comprehensive Chinese assessment suite specifically designed to evaluate the advanced knowledge and reasoning abilities of LLMs within the Chinese language and cultural context. CMMLU covers a wide range of subjects, comprising 67 topics that span from elementary to advanced professional levels. It includes subjects that require computational expertise, such as physics and mathematics, as well as disciplines within humanities and social sciences. Many of these tasks are not easily translatable from other languages due to their specific contextual nuances and wording. Furthermore, numerous tasks within CMMLU have answers that are specific to China and may not be universally applicable or considered correct in other regions or languages.
"""

_HOMEPAGE = "https://huggingface.co/datasets/haonan-li/cmmlu"

_LICENSE = "cc-by-nc-4.0"

_URL = "cmmlu.zip"

task_list = [
    'agronomy', 
    'anatomy', 
    'ancient_chinese', 
    'arts', 
    'astronomy', 
    'business_ethics', 
    'chinese_civil_service_exam', 
    'chinese_driving_rule', 
    'chinese_food_culture', 
    'chinese_foreign_policy', 
    'chinese_history',
    'chinese_literature', 
    'chinese_teacher_qualification', 
    'clinical_knowledge', 
    'college_actuarial_science', 
    'college_education', 
    'college_engineering_hydrology', 
    'college_law', 
    'college_mathematics', 
    'college_medical_statistics', 
    'college_medicine', 
    'computer_science',
    'computer_security', 
    'conceptual_physics', 
    'construction_project_management', 
    'economics', 
    'education', 
    'electrical_engineering', 
    'elementary_chinese', 
    'elementary_commonsense', 
    'elementary_information_and_technology', 
    'elementary_mathematics', 
    'ethnology', 
    'food_science', 
    'genetics', 
    'global_facts', 
    'high_school_biology', 
    'high_school_chemistry', 
    'high_school_geography', 
    'high_school_mathematics', 
    'high_school_physics', 
    'high_school_politics', 
    'human_sexuality',
    'international_law', 
    'journalism', 
    'jurisprudence', 
    'legal_and_moral_basis', 
    'logical', 
    'machine_learning', 
    'management', 
    'marketing', 
    'marxist_theory', 
    'modern_chinese', 
    'nutrition', 
    'philosophy', 
    'professional_accounting', 
    'professional_law', 
    'professional_medicine', 
    'professional_psychology', 
    'public_relations', 
    'security_study', 
    'sociology', 
    'sports_science', 
    'traditional_chinese_medicine', 
    'virology', 
    'world_history', 
    'world_religions'
]


class CMMLUConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class CMMLU(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CMMLUConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
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
        df.columns = ["A", "B", "C", "D", "question", "answer"]
        
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
