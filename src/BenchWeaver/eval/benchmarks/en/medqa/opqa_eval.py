from ....evaluator import OPQAEvaluator
from ....template.eval.medqa_template import get_medqa_eval_template

class MedQAOPQAEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_medqa_eval_template(self.eval_args.lang)