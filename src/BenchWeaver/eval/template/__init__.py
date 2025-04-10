from .template import EvalTemplate, TransTemplate, _register_eval_template, _register_trans_template
from .eval.mcqa_template import MCQA_Template
from .eval.opqa_template import OPQA_Template
from .eval.mmlu_template import get_mmlu_eval_template
from .eval.arc_challenge_template import get_arc_challenge_eval_template
from .eval.gpqa_template import get_gpqa_eval_template
from .eval.gsm8k_template import get_gsm8k_eval_template
from .eval.truthfulqa_template import get_truthfulqa_eval_template, TruthfulQA_Template
from .eval.big_bench_hard_template import get_big_bench_hard_eval_template, BigBenchHard_Template
from .eval.click_template import CLIcK_Template, get_click_eval_template
from .eval.hae_rae_bench_template import HAE_RAE_BENCH_Template, get_hae_rae_bench_eval_template
from .eval.tmlu_template import TMLU_Template, get_tmlu_eval_template
from .eval.drcd_template import DRCD_Template, get_drcd_eval_template
from .eval.awesome_taiwan_knowledge_template import get_awesome_taiwan_knowledge_eval_template
from .eval.taide_bench_template import get_taide_bench_eval_template
from .eval.ccpm_template import get_ccpm_eval_template
from .eval.cmath_template import get_cmath_eval_template
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
    "CLIcK_Template",
    "HAE_RAE_BENCH_Template",
    "TMLU_Template",
    "DRCD_Template",
    # function
    "get_mmlu_eval_template",
    "get_arc_challenge_eval_template",
    "get_gpqa_eval_template",
    "get_gsm8k_eval_template",
    "get_truthfulqa_eval_template",
    "get_big_bench_hard_eval_template",
    "get_click_eval_template",
    "get_hae_rae_bench_eval_template",
    "get_tmlu_eval_template",
    "get_drcd_eval_template",
    "get_awesome_taiwan_knowledge_eval_template",
    "get_taide_bench_eval_template",
    "get_ccpm_eval_template",
    "get_cmath_eval_template",
    # translation
    "get_translation_template",
    # register
    "_register_eval_template",
    "_register_trans_template",
]