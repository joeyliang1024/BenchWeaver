from .template import EvalTemplate, TransTemplate, _register_eval_template, _register_trans_template
from .eval.mcqa_template import MCQA_Template
from .eval.opqa_template import OPQA_Template
from .eval.mmlu_template import get_mmlu_eval_template
from .eval.arc_challenge_template import get_arc_challenge_eval_template
from .eval.gpqa_template import get_gpqa_eval_template
from .eval.gsm8k_template import get_gsm8k_eval_template
from .eval.truthfulqa_template import get_truthfulqa_eval_template, TruthfulQA_Template
from .eval.big_bench_hard_template import get_big_bench_hard_eval_template, BigBenchHard_Template
from .trans.advance_template import AdvancedTransTemplate
from .trans.trans_template import get_translation_template
__all__ = [
    # class
    "EvalTemplate",
    "TransTemplate",
    "MCQA_Template",
    "OPQA_Template",
    "AdvancedTransTemplate",
    "TruthfulQA_Template",
    "BigBenchHard_Template",
    # function
    "get_mmlu_eval_template",
    "get_arc_challenge_eval_template",
    "get_gpqa_eval_template",
    "get_gsm8k_eval_template",
    "get_truthfulqa_eval_template",
    "get_big_bench_hard_eval_template",
    "get_translation_template",
    # register
    "_register_eval_template",
    "_register_trans_template",
]