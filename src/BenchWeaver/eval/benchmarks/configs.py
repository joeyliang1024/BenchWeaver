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
    },
    "cmmlu": {
        "language": "zh",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", 'STEM', 'Humanities', 'Social Science', 'Other'],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": False,
    },
    "taide-bench": {
        "language": "zh-tw",
        "mode": ["opqa"],
        "display_scores": ["Average", "letter", "essay", "summary", "zh2en", "en2zh",],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "ccpm": {
        "language": "zh",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "cmath": {
        "language": "zh",
        "mode": ["opqa"],
        "display_scores": ["Average", "main", "distractor",],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "cif-bench": {
        "language": "zh",
        "mode": ["opqa"],
        "display_scores": ["Average", 'Grammar', 'Style Transfer', 'Commonsense', 'Motion', 'NLI', 'NLG', 'Summarization', 'Toxic', 'Creative', 'NER', 'Code', 'Translation', 'Structured Data', 'Linguistic', 'Evaluation', 'Chinese Culture', 'QA', 'Role Playing', 'Sentiment', 'Reasoning', 'Classification', 'Detection'],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "c3": {
        "language": "zh",
        "mode": ["mcqa-oq"],
        "display_scores": ["Average", 'mixed', "dialogue"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "chinese-safety-qa": {
        "language": "zh",
        "mode": ["mcqa-oq"],
        "display_scores": ["Average", '理论技术知识', '伦理道德风险', '偏见歧视风险', '辱骂仇恨风险', '身心健康风险', '违法违规风险', '谣言错误风险'],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "mt-bench-tw": {
        "language": "zh-tw",
        "mode": ["multi-turn"],
        "display_scores": ["Average", "writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "hellaswag": {
        "language": "en",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "ifeval": {
        "language": "en",
        "mode": ["opqa"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "flores-plus": {
        "language": "en",
        "mode": ["trans"],
        "display_scores": ["Average"],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "mbpp": {
        "language": "en",
        "mode": ["code"],
        "display_scores": ["Average", "full"],
        "mcqa_choices": None,
        "sugguest_num_shots": 3,
        "support_chain_of_thought": False,
    },
    "xnli": {
        "language": "en",
        "mode": ["opqa"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "logiqa": {
        "language": "en",
        "mode": [ "mcqa-oq"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "humaneval-xl": {
        "language": "en",
        "mode": ["code"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": None,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "logickor": {
        "language": "ko",
        "mode": ["multi-turn"],
        "display_scores": ["Average", "Reasoning", "Math", "Writing", "Coding", "Understanding", "Grammar", "Single Turn", "Multi Turn"],
        "mcqa_choices": None,
        "sugguest_num_shots": 1,
        "support_chain_of_thought": True,
    },
    "medqa": {
        "language": "en",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
    "medmcqa": {
        "language": "en",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", "all"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 0,
        "support_chain_of_thought": False,
    },
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
    elif task_name == "cmmlu":
        from .en.mmlu import MMLUOQEvaluator, MMLUProbEvaluator
        return {
            "mcqa-prob": MMLUProbEvaluator,
            "mcqa-oq": MMLUOQEvaluator,
        }
    elif task_name == "taide-bench":
        from .zhtw.taide_bench.opqa_eval import TaideBenchEvaluator
        return {
            "opqa": TaideBenchEvaluator,
        }
    elif task_name == "ccpm":
        from .zh.ccpm.oq_eval import CCPMOQEvaluator
        from .zh.ccpm.prob_eval import CCPMProbEvaluator
        return {
            "mcqa-prob": CCPMProbEvaluator,
            "mcqa-oq": CCPMOQEvaluator,
        }
    elif task_name == "cmath":
        from .zh.cmath.opqa_eval import CMATHEvaluator
        return {
            "opqa": CMATHEvaluator,
        }
    elif task_name == "cif-bench":
        from .zh.cif_bench.opqa_eval import CifBenchEvaluator
        return {
            "opqa": CifBenchEvaluator,
        }
    elif task_name == "c3":
        from .zh.c3.oq_eval import C3OQEvaluator
        return {
            "mcqa-oq": C3OQEvaluator,
        }
    elif task_name == "chinese-safety-qa":
        from .zh.chinese_safety_qa.oq_eval import ChineseSafetyQAOQEvaluator
        return {
            "mcqa-oq": ChineseSafetyQAOQEvaluator,
        }
    elif task_name == "mt-bench-tw":
        from .zhtw.mt_bench_tw.multi_turn_eval import MTBenchTWEvaluator
        return {
            "multi-turn": MTBenchTWEvaluator,
        }
    elif task_name == "hellaswag":
        from .en.hellaswag.oq_eval import HellaSwagOQEvaluator
        from .en.hellaswag.prob_eval import HellaSwagProbEvaluator
        return {
            "mcqa-prob": HellaSwagProbEvaluator,
            "mcqa-oq": HellaSwagOQEvaluator,
        }
    elif task_name == "ifeval":
        from .en.ifeval.opqa_eval import IFEvalEvaluator
        return {
            "opqa": IFEvalEvaluator,
        }
    elif task_name == "flores-plus":
        from .en.flores_plus.trans_eval import FloresPlusEvaluator
        return {
            "trans": FloresPlusEvaluator,
        }
    elif task_name == "mbpp":
        from .en.mbpp.code_eval import MBPPEvaluator
        return {
            "code": MBPPEvaluator,
        }
    elif task_name == "xnli":
        from .en.xnli.opqa_eval import XNLIEvaluator
        return {
            "opqa": XNLIEvaluator,
        }
    elif task_name == "logiqa":
        from .en.logiqa.oq_eval import LogiQAEvaluator
        return {
            "mcqa-oq": LogiQAEvaluator,
        }
    elif task_name == "humaneval-xl":
        from .en.humaneval_xl.code_eval import HumanEvalXLEvaluator
        return {
            "code": HumanEvalXLEvaluator,
        }
    elif task_name == "logickor":
        from .ko.logickor.multi_turn_eval import LogicKorEvaluator
        return {
            "multi-turn": LogicKorEvaluator,
        }
    elif task_name == "medqa":
        from .en.medqa.prob_eval import MedQAProbEvaluator
        from .en.medqa.oq_eval import MedQAOQEvaluator
        return {
            "mcqa-prob": MedQAProbEvaluator,
            "mcqa-oq": MedQAOQEvaluator,
        }
    elif task_name == "medmcqa":
        from .en.medmcqa.prob_eval import MedMCQAProbEvaluator
        from .en.medmcqa.oq_eval import MedMCQAOQEvaluator
        return {
            "mcqa-prob": MedMCQAProbEvaluator,
            "mcqa-oq": MedMCQAOQEvaluator,
        }
    return {}
