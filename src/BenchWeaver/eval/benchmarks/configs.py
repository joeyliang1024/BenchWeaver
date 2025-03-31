from typing import Any, Dict
from ...extras.constants import OPTION_CODES

# Define the BENCHMARK_CONFIG with None for evaluators initially
BENCHMARK_CONFIG = {
    "mmlu": {
        "language": "en",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", "STEM", "Social Sciences", "Humanities", "Other"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,
    },
    "arc-challenge": {
        "language": "en",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", "challenge"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,
    },
    "gpqa": {
        "language": "en",
        "mode": ["opqa"],
        "display_scores": ["Average", "diamond", "extended", "main"],
        "mcqa_choices": None,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,
    },
    "gsm8k": {
        "language": "en",
        "mode": ["opqa"],
        "display_scores": ["Average", "main", "socratic"],
        "mcqa_choices": None,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,
    },
    "truthfulqa": {
        "language": "en",
        "mode": ["mix"],
        "display_scores": ["Average", "generation", "mcqa-mc1", "mcqa-mc2"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": True,
    },
    "big-bench-hard": {
        "language": "en",
        "mode": ["mix"],
        "display_scores": ["Average", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "object_counting", "penguins_in_a_table", "salient_translation_error_detection", "tracking_shuffled_objects_five_objects"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 3,
        "support_chain_of_thought": True,
    },
    "click": {
        "language": "ko",
        "mode": ["mcqa-oq"],
        "display_scores": ["Average", "TK", "Kedu", "PSE", "PSAT", "CSAT", "KHB", "KIIP"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "hae-rae-bench": {
        "language": "ko",
        "mode": ["mix"],
        "display_scores": ['Average', 'lyrics_denoising', 'proverbs_denoising', 'correct_definition_matching', 'csat_geo', 'csat_law', 'csat_socio', 'date_understanding', 'general_knowledge', 'history', 'loan_words', 'rare_words', 'standard_nomenclature', 'reading_comprehension'],
        "mcqa_choices": ["A", "B", "C", "D", "E"],
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False
    },
    "kmmlu": {
        "language": "ko",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", 'STEM', 'Applied Science', 'HUMSS', 'Other'],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": False,
    },
    "kmmlu-hard": {
        "language": "ko",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", 'STEM', 'Applied Science', 'HUMSS', 'Other'],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,
    },
    "tmmluplus": {
        "language": "zh-tw",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", 'STEM', 'Social Sciences', 'Humanities', 'Other'],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": False,
    },
    "tmlu": {
        "language": "zh-tw",
        "mode": ["mcqa-oq"],
        "display_scores": ["Average", "Social Science", "STEM", "Humanities", "Taiwan Specific", "Others"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,
    },
    "drcd": {
        "language": "zh-tw",
        "mode": ["opqa"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": None,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": False,
    },
    "awesome-taiwan-knowledge": {
        "language": "zh-tw",
        "mode": ["opqa"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    }
}


def get_evaluators(task_name:str) -> Dict[str, Any]:
    """
    Dynamically import evaluators for the specified task. This is use only for cli showing all support benchmarks.
    """
    if task_name == "mmlu":
        from .en.mmlu import MMLUOQEvaluator, MMLUProbEvaluator
        return {
            "mcqa-prob": MMLUProbEvaluator,
            "mcqa-oq": MMLUOQEvaluator,
        }
    elif task_name == "arc-challenge":
        from .en.arc_challenge import ArcChallengeProbEvaluator, ArcChallengeOQEvaluator
        return {
            "mcqa-prob": ArcChallengeProbEvaluator,
            "mcqa-oq": ArcChallengeOQEvaluator,
        }
    elif task_name == "gpqa":
        from .en.gpqa import GPQAEvaluator
        return {
            "opqa": GPQAEvaluator,
        }
    elif task_name == "gsm8k":
        from .en.gsm8k import GSM8KEvaluator
        return {
            "opqa": GSM8KEvaluator,
        }
    elif task_name == "truthfulqa":
        from .en.truthfulqa import TruthfulQAEvaluator
        return {
            "mix": TruthfulQAEvaluator,
        }
    elif task_name == "big-bench-hard":
        from .en.big_bench_hard.mix_eval import BigBenchHardEvaluator
        return {
            "mix": BigBenchHardEvaluator,
        }
    elif task_name == "click":
        from .ko.click.oq_eval import CLIcKEvaluator
        return {
            "mcqa-oq": CLIcKEvaluator,
        }
    elif task_name == "hae-rae-bench":
        from .ko.hae_rae_bench.mix_eval import HAE_RAE_BENCHEvaluator
        return {
            "mix": HAE_RAE_BENCHEvaluator,
        }
    elif task_name == "kmmlu":
        from .en.mmlu import MMLUProbEvaluator, MMLUOQEvaluator
        return {
            "mcqa-prob": MMLUProbEvaluator,
            "mcqa-oq": MMLUOQEvaluator,
        }
    elif task_name == "kmmlu-hard":
        from .en.mmlu import MMLUProbEvaluator, MMLUOQEvaluator
        return {
            "mcqa-prob": MMLUProbEvaluator,
            "mcqa-oq": MMLUOQEvaluator,
        }
    elif task_name == "tmmluplus":
        from .en.mmlu import MMLUOQEvaluator, MMLUProbEvaluator
        return {
            "mcqa-prob": MMLUProbEvaluator,
            "mcqa-oq": MMLUOQEvaluator,
        }
    elif task_name == "tmlu":
        from .zhtw.tmlu.oq_eval import TMLUEvaluator
        return {
            "mcqa-oq": TMLUEvaluator,
        }
    elif task_name == "drcd":
        from .zhtw.drcd.opqa_eval import DRCDEvaluator
        return {
            "opqa": DRCDEvaluator,
        }
    elif task_name == "awesome-taiwan-knowledge":
        from .zhtw.awesome_taiwan_knowledge.opqa_eval import AwesomeTaiwanKnowledgeEvaluator
        return {
            "opqa": AwesomeTaiwanKnowledgeEvaluator,
        }
    return {}
