from .template import EvalTemplate
from .mcqa_template import MCQA_Template
from .opqa_template import OPQA_Template
from .mmlu_template import get_mmlu_eval_template
from .arc_challenge_template import get_arc_challenge_eval_template
from .gpqa_template import get_gpqa_eval_template
from .gsm8k_template import get_gsm8k_eval_template
from .truthfulqa_template import get_truthfulqa_eval_template, TruthfulQA_Template
from .big_bench_hard_template import get_big_bench_hard_eval_template, BigBenchHard_Template

__all__ = [
    # class
    "EvalTemplate",
    "MCQA_Template",
    "OPQA_Template",
    "TruthfulQA_Template",
    "BigBenchHard_Template",
    # function
    "get_mmlu_eval_template",
    "get_arc_challenge_eval_template",
    "get_gpqa_eval_template",
    "get_gsm8k_eval_template",
    "get_truthfulqa_eval_template",
    "get_big_bench_hard_eval_template",
    "_register_eval_template",
]