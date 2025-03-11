from ...extras.constants import OPTION_CODES
from .en.mmlu import MMLUOQEvaluator, MMLUProbEvaluator
from .en.arc_challenge import ArcChallengeOQEvaluator, ArcChallengeProbEvaluator
from .en.gpqa import GPQAEvaluator
from .en.gsm8k import GSM8KEvaluator
from .en.truthfulqa import TruthfulQAEvaluator
from .en.big_bench_hard.mix_eval import BigBenchHardEvaluator

BENCHMARK_CONFIG = {
    "mmlu":{
        "language": "en",
        "evaluators": {
            "mcqa-prob": MMLUProbEvaluator,
            "mcqa-oq": MMLUOQEvaluator,
        },
        "display_scores": ["Average", "STEM", "Social Sciences", "Humanities", "Other"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        },     
    "arc_challenge":{
        "language": "en",
        "evaluators": {
            "mcqa-prob": ArcChallengeProbEvaluator,
            "mcqa-oq": ArcChallengeOQEvaluator,
        },
        "display_scores": ["Average", "challenge"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        },
    "gpqa":{
        "language": "en",
        "evaluators": {
            "opqa": GPQAEvaluator,
        },
        "display_scores": ["Average", "diamond", "extended", "main"],
        "mcqa_choices": None,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        }, 
    "gsm8k":{
        "language": "en",
        "evaluators": {
            "opqa": GSM8KEvaluator,
        },
        "display_scores": ["Average", "main", "socratic"],
        "mcqa_choices": None,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        }, 
    "truthfulqa":{
        "language": "en",
        "evaluators": {
            "mix": TruthfulQAEvaluator,
        },
        "display_scores": ["Average", "generation", "mcqa-mc1", "mcqa-mc2"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": True,   
        }, 
    "big_bench_hard":{
        "language": "en",
        "evaluators": {
            "mix": BigBenchHardEvaluator,
        },
        "display_scores": ["Average", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "object_counting", "penguins_in_a_table", "salient_translation_error_detection", "tracking_shuffled_objects_five_objects"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 3,
        "support_chain_of_thought": True,   
    }
}