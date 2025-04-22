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
@article{nllb-24,
    author="{NLLB Team} and Costa-juss{\`a}, Marta R. and Cross, James and {\c{C}}elebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and Youngblood, Al and Akula, Bapi and Barrault, Loic and Gonzalez, Gabriel Mejia and Hansanti, Prangthip and Hoffman, John and Jarrett, Semarley and Sadagopan, Kaushik Ram and Rowe, Dirk and Spruit, Shannon and Tran, Chau and Andrews, Pierre and Ayan, Necip Fazil and Bhosale, Shruti and Edunov, Sergey and Fan, Angela and Gao, Cynthia and Goswami, Vedanuj and Guzm{\'a}n, Francisco and Koehn, Philipp and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and Schwenk, Holger and Wang, Jeff",
    title="Scaling neural machine translation to 200 languages",
    journal="Nature",
    year="2024",
    volume="630",
    number="8018",
    pages="841--846",
    issn="1476-4687",
    doi="10.1038/s41586-024-07335-x",
    url="https://doi.org/10.1038/s41586-024-07335-x"
}
"""

_DESCRIPTION = """\
FLORES+ is a multilingual machine translation benchmark released under CC BY-SA 4.0. This dataset was originally released by FAIR researchers at Meta under the name FLORES. Further information about these initial releases can be found in Dataset Sources below. The data is now being managed by OLDI, the Open Language Data Initiative. The + has been added to the name to disambiguate between the original datasets and this new actively developed version.
"""

_HOMEPAGE = "https://huggingface.co/datasets/openlanguagedata/flores_plus"

_LICENSE = "cc-by-sa-4.0"

_URL = "flores-plus.zip"

task_list = [
    "cmn_Hans",
    "cmn_Hant",
    "kor_Hang",
    "eng_Latn",
]

class FloresPlusConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class FloresPlus(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        FloresPlusConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("int64"),
                "iso_639_3": datasets.Value("string"),
                "iso_15924": datasets.Value("string"),
                "glottocode": datasets.Value("string"),
                "text": datasets.Value("string"),
                "url": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "topic": datasets.Value("string"),
                "has_image": datasets.Value("string"),
                "has_hyperlink": datasets.Value("string"),
                "last_updated": datasets.Value("string"),
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
        df.columns = ["id", "iso_639_3", "iso_15924", "glottocode", "text", "url", "domain", "topic", "has_image", "has_hyperlink", "last_updated"]

        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
