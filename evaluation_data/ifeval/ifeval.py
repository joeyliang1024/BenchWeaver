import os
from typing import List
import datasets
import pandas as pd

_CITATION = """\
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
"""

_DESCRIPTION = """\
One core capability of Large Language Models (LLMs) is to follow natural language instructions. However, the evaluation of such abilities is not standardized:
Human evaluations are expensive, slow, and not objectively reproducible, while
LLM-based auto-evaluation is potentially biased or limited by the ability of the
evaluator LLM. To overcome these issues, we introduce Instruction-Following
Eval (IFEval) for large language models. IFEval is a straightforward and easy-to-
reproduce evaluation benchmark. It focuses on a set of “verifiable instructions”
such as “write in more than 400 words” and “mention the keyword of AI at least
3 times”. We identified 25 types of those verifiable instructions and constructed
around 500 prompts, with each prompt containing one or more verifiable instructions. We show evaluation results of two widely available LLMs on the market.
"""

_HOMEPAGE = "https://github.com/google-research/google-research/tree/master/instruction_following_eval"
_LICENSE = "Apache License 2.0"
_URL = "ifeval.zip"

task_list = ["all"]

class IFEvalConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class IFEval(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [IFEvalConfig(name=task) for task in task_list]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "key": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "instruction_id_list": datasets.Sequence(datasets.Value("string")),
                "kwargs": datasets.Features({
                        "num_highlights":      datasets.Value("float64"),
                        "relation":            datasets.Value("string"),
                        "num_words":           datasets.Value("float64"),
                        "num_placeholders":    datasets.Value("float64"),
                        "prompt_to_repeat":    datasets.Value("string"),
                        "num_bullets":         datasets.Value("float64"),
                        "section_spliter":     datasets.Value("string"),
                        "num_sections":        datasets.Value("float64"),
                        "capital_relation":    datasets.Value("string"),
                        "capital_frequency":   datasets.Value("float64"),
                        "keywords":            datasets.Sequence(datasets.Value("string")),
                        "num_paragraphs":      datasets.Value("float64"),
                        "language":            datasets.Value("string"),
                        "let_relation":        datasets.Value("string"),
                        "letter":              datasets.Value("string"),
                        "let_frequency":       datasets.Value("float64"),
                        "end_phrase":          datasets.Value("string"),
                        "forbidden_words":     datasets.Sequence(datasets.Value("string")),
                        "keyword":             datasets.Value("string"),
                        "frequency":           datasets.Value("float64"),
                        "num_sentences":       datasets.Value("float64"),
                        "postscript_marker":   datasets.Value("string"),
                        "first_word":          datasets.Value("string"),
                        "nth_paragraph":       datasets.Value("float64"),
                    })
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        path = os.path.join(data_dir, "data", "test", f"{self.config.name}_test.parquet")
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": path})
        ]

    def _generate_examples(self, filepath):
        # Read in with pandas so that 'kwargs' remains a list of dicts
        df = pd.read_parquet(filepath)
        # Ensure the dataframe columns are as expected
        df.columns = ["key", "question", "instruction_id_list", "kwargs"]
        def format_new_dict(x: List[dict]):
            record = {}
            for _dict in x:
                for key, value in _dict.items():
                    if key not in record:
                        if key in ["keywords", "forbidden_words"]:
                            record[key] = [_ for _ in value] if value is not None else None
                        record[key] = value
                    
                    if value is not None:
                        if key in ["keywords", "forbidden_words"]:
                            record[key] = [_ for _ in value]
                        record[key] = value
            return record
        df['kwargs'] = df['kwargs'].apply(format_new_dict)
        for idx, example in enumerate(df.to_dict(orient="records")):
            yield idx, example
