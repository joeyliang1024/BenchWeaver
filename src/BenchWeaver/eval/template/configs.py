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

EVAL_TEMPLATE_CONFIG = {
    "mmlu":{
        "class": MCQA_Template,
        "func": get_mmlu_eval_template,
        },
    "arc-challenge":{
        "class": MCQA_Template,
        "func": get_arc_challenge_eval_template,
        },
    "gpqa":{
       "class": OPQA_Template,
       "func": get_gpqa_eval_template,
        },
    "gsm8k":{
        "class": OPQA_Template,
        "func": get_gsm8k_eval_template,
        }, 
    "truthfulqa":{
         "class": TruthfulQA_Template,
         "func": get_truthfulqa_eval_template,
        }, 
    "big-bench-hard":{
        "class": BigBenchHard_Template,
        "func": get_big_bench_hard_eval_template,
        },
    "click":{
        "class": CLIcK_Template,
        "func": get_click_eval_template,
        },
    "hae-rae-bench":{
        "class": HAE_RAE_BENCH_Template,
        "func": get_hae_rae_bench_eval_template,
        },
    "kmmlu":{
        "class": MCQA_Template,
        "func": get_mmlu_eval_template,
        },
    "kmmlu-hard":{
        "class": MCQA_Template,
        "func": get_mmlu_eval_template,
        },
    "tmmluplus":{
        "class": MCQA_Template,
        "func": get_mmlu_eval_template,
        },
    "tmlu":{
        "class": TMLU_Template,
        "func": get_tmlu_eval_template,
        },
    "drcd":{
        "class": DRCD_Template,
        "func": get_drcd_eval_template,
        },
    "awesome-taiwan-knowledge":{
        "class": OPQA_Template,
        "func": get_awesome_taiwan_knowledge_eval_template,
        },
    "taide-bench":{
        "class": OPQA_Template,
        "func": get_taide_bench_eval_template,
        },
    "ccpm":{
        "class": MCQA_Template,
        "func": get_ccpm_eval_template,
        },
}