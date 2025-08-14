from .template import EvalTemplate, TransTemplate, _register_eval_template, _register_trans_template
from .eval.mcqa_template import MCQA_Template
from .eval.opqa_template import OPQA_Template
from .eval.multi_turn_template import Multi_Turn_Template
from .eval.trans_template import Trans_Template
from .eval.code_template import Code_Template
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
from .eval.cif_bench_template import get_cif_bench_eval_template
from .trans.advance_template import AdvancedTransTemplate
from .trans.trans_template import get_translation_template
from .eval.c3_template import C3_Template, get_c3_eval_template
from .eval.chinese_safety_qa_template import ChineseSafetyQA_Template, get_chinese_safety_qa_eval_template
from .eval.mt_bench_tw_template import MT_Bench_TW_Template, get_mt_bench_tw_eval_template
from .eval.hellaswag_template import get_hellaswag_eval_template
from .eval.ifeval_template import IFEval_Template, get_ifeval_eval_template
from .eval.flores_template import get_flores_eval_template
from .eval.mbpp_template import get_mbpp_eval_template
from .eval.xnli_template import XNLI_Template, get_xnli_eval_template
from .eval.logiqa_template import LogiQA_Template, get_logiqa_eval_template
from .eval.humaneval_xl_template import get_humaneval_xl_eval_template
from .eval.logickor_template import LogicKor_Template, get_logickor_eval_template
from .eval.medqa_template import get_medqa_eval_template
from .eval.medmcqa_template import get_medmcqa_eval_template
from .eval.kobest_template import KoBest_Template, get_kobest_eval_template
from .eval.huatuo_template import get_huatuo_eval_template

__all__ = [
    # class
    "EvalTemplate",
    "TransTemplate",
    "MCQA_Template",
    "OPQA_Template",
    "Multi_Turn_Template",
    "Trans_Template",
    "Code_Template",
    "AdvancedTransTemplate",
    "TruthfulQA_Template",
    "BigBenchHard_Template",
    "CLIcK_Template",
    "HAE_RAE_BENCH_Template",
    "TMLU_Template",
    "DRCD_Template",
    "C3_Template",
    "ChineseSafetyQA_Template",
    "MT_Bench_TW_Template",
    "IFEval_Template",
    "XNLI_Template",
    "LogiQA_Template",
    "LogicKor_Template",
    "KoBest_Template",
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
    "get_cif_bench_eval_template",
    "get_c3_eval_template",
    "get_chinese_safety_qa_eval_template",
    "get_mt_bench_tw_eval_template",
    "get_hellaswag_eval_template",
    "get_ifeval_eval_template",
    "get_flores_eval_template",
    "get_mbpp_eval_template",
    "get_xnli_eval_template",
    "get_humaneval_xl_eval_template",
    "get_logiqa_eval_template",
    "get_logickor_eval_template",
    "get_medqa_eval_template",
    "get_medmcqa_eval_template",
    "get_kobest_eval_template",
    "get_huatuo_eval_template"
    # translation
    "get_translation_template",
    # register
    "_register_eval_template",
    "_register_trans_template",
]