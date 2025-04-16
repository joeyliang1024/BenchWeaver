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
@article{li2024cifbench,
        title={CIF-Bench: A Chinese Instruction-Following Benchmark for Evaluating the Generalizability of Large Language Models}, 
        author={Yizhi LI and Ge Zhang and Xingwei Qu and Jiali Li and Zhaoqun Li and Zekun Wang and Hao Li and Ruibin Yuan and Yinghao Ma and Kai Zhang and Wangchunshu Zhou and Yiming Liang and Lei Zhang and Lei Ma and Jiajun Zhang and Zuowen Li and Stephen W. Huang and Chenghua Lin and Wenhu Chen and Jie Fu},
        year={2024},
        eprint={2402.13109},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
      }
"""

_DESCRIPTION = """\
he advancement of large language models (LLMs) has enhanced the ability to generalize across a wide range of unseen natural language processing (NLP) tasks through instruction-following. Yet, their effectiveness often diminishes in low-resource languages like Chinese, exacerbated by biased evaluations from data leakage, casting doubt on their true generalizability to new linguistic territories. In response, we introduce the Chinese Instruction-Following Benchmark (CIF-Bench), designed to evaluate the zero-shot generalizability of LLMs to the Chinese language. CIF-Bench comprises 150 tasks and 15,000 input-output pairs, developed by native speakers to test complex reasoning and Chinese cultural nuances across 20 categories. To mitigate evaluation bias, we release only half of the dataset publicly, with the remainder kept private, and introduce diversified instructions to minimize score variance, totaling 45,000 data instances. Our evaluation of 28 selected LLMs reveals a noticeable performance gap, with the best model scoring only 52.9%, highlighting the limitations of LLMs in less familiar language and task contexts. This work aims to uncover the current limitations of LLMs in handling Chinese tasks, pushing towards the development of more culturally informed and linguistically diverse models with the released data and benchmark.
"""

_HOMEPAGE = "https://github.com/yizhilll/CIF-Bench"

_LICENSE = "CC-BY-SA-4.0"

_URL = "cif-bench.zip"

task_list = [
    "chinese_attractions_list",
    "intimacy_score_prediction",
    "word_semantics",
    "speaker_identification",
    "chinese_figurative_detection",
    "multilabel_chinese_humor_categorization",
    "argument_mining",
    "code_to_text",
    "social_norms_detection",
    "paraphrasing",
    "theory_of_mind",
    "stereotype_detection",
    "sentence_ordering",
    "sentence_expansion",
    "fill_in_the_blank",
    "discourse_relation_classification",
    "chinese_dialect_translation",
    "game_playing",
    "chinese_modern_abbreviation_explanation",
    "entity_relation_classification",
    "rhyme_aligned_generationn",
    "text_categorization",
    "chinese_winograd_schema_challenge",
    "stance_detection",
    "mind_tree_generation",
    "grammar_error_correction",
    "chinese_epigraph_detection",
    "discourse_connective_identification",
    "explanation",
    "word_relation_classification",
    "discreteoperationqa",
    "style_transfer",
    "iq_test",
    "story_composition",
    "chinese_fiction_characteristic_detection",
    "poem_generation",
    "code_debug",
    "first_order_logic",
    "make_positive",
    "chinese_pinyin_detection",
    "flowchart_generation",
    "commonsense",
    "preposition_prediction",
    "outline_generation",
    "number_conversion",
    "title_generation",
    "question_generation",
    "english_translation",
    "recipe_generation",
    "dialogue_generation",
    "chinese_ambiguity_sentence_location",
    "intent_identification",
    "data_to_text",
    "named_entity_recognition",
    "wrong_candidate_generation",
    "advertising",
    "gender_classification",
    "event_type_detection",
    "text_matching",
    "joke_telling",
    "spam_classification",
    "spanish_translation",
    "chinese_typo_categorization",
    "name_allusion_detection",
    "chinese_wubi_written",
    "spelling_error_detection",
    "information_extraction",
    "affordance",
    "question_rewriting",
    "review_generation",
    "bengali_translation",
    "bias_detoxication",
    "emotion_prediction",
    "nationality_detection",
    "text_de_identification",
    "commonsenseqa",
    "region_detection",
    "cause_effect_classification",
    "commonsense_explanation",
    "fact_verification",
    "readingcomprehensionqa",
    "function_explanation",
    "chinese_medicine_detection",
    "program_execution",
    "summarization",
    "question_answering",
    "question_understanding",
    "grammar_error_detection",
    "ancient_chinese_poem_retrieval",
    "patronizing_condescending_multilabel",
    "negotiation_strategy_detection",
    "keyword_tagging",
    "dialogue_act_recognition",
    "irony_detection",
    "table_generation",
    "tool_use",
    "sentence_compression",
    "text_quality_evaluation",
    "chinese_heteronomous_language_detection",
    "entity_generation",
    "pos_tagging",
    "commonsense_classification",
    "sentence_composition",
    "textual_entailment",
    "legal_term_retrieval",
    "draw_figure_with_symbol",
    "chinese_idiom_explanation",
    "dialogue_state_tracking",
    "role_playing",
    "critical_thinking",
    "chinese_relative_identification",
    "linguistic_probing",
    "arabic_translation",
    "french_translation",
    "text_to_code",
    "sentiment_analysis",
    "answer_verification",
    "japanese_translation",
    "material_synthesis",
    "chinese_metaphor_explanation",
    "emphathy_detection",
    "concept_abstraction",
    "pros_cons_listing",
    "conversationalqa",
    "personality_detection",
    "overlap_extraction",
    "translate_to_ancient_chinese",
    "text_completion",
    "speaker_relation_classification",
    "gujarati_translation",
    "sentence_perturbation",
    "compositional_reasoning",
    "mathqa",
    "coreference_resolution",
    "mathematics",
    "text_simplification",
    "joke_explanation",
    "multihopqa",
    "question_decomposition",
    "commonsensenli",
    "toxic_language_detection",
    "chinese_rhyme_detection",
    "code_translate",
    "imagination",
    "tamil_translation",
    "answerability_classification",
    "sentence_level_uncertainty_judgement",
    "ancient_chinese_translation",
    "punctuation_error_detection"
]

class CifBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class CifBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CifBenchConfig(
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
