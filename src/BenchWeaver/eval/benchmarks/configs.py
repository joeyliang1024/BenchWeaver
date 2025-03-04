from ...extras.constants import OPTION_CODES

benchmark_configs = {
    "mmlu":{
        "language": "en",
        "evaluators": [
            "mcqa-prob",
            "mcqa-oq"
            ],
        "display_scores": ["Average", "STEM", "Social Sciences", "Humanities", "Other"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        },     
    "arc_challenge":{
        "language": "en",
        "evaluators": [
            "mcqa-prob",
            "mcqa-oq"
            ],
        "display_scores": ["Average", "challenge"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        },
    "gpqa":{
        "language": "en",
        "evaluators": [
            "opqa"
            ],
        "display_scores": ["Average", "diamond", "extended", "main"],
        "mcqa_choices": None,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        }, 
    "gsm8k":{
        "language": "en",
        "evaluators": [
            "opqa"
            ],
        "display_scores": ["Average", "main", "socratic"],
        "mcqa_choices": None,
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,   
        }, 
    "truthfulqa":{
        "language": "en",
        "evaluators": [
            "mix"
            ],
        "display_scores": ["Average", "generation", "mcqa-mc1", "mcqa-mc2"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 0,
        "support_chain_of_thought": True,   
        }, 
    "big_bench_hard":{
        "language": "en",
        "evaluators": [
            "mix"
            ],
        "display_scores": ["Average", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "object_counting", "penguins_in_a_table", "salient_translation_error_detection", "tracking_shuffled_objects_five_objects"],
        "mcqa_choices": OPTION_CODES,
        "sugguest_num_shots": 3,
        "support_chain_of_thought": True,   
    }
}