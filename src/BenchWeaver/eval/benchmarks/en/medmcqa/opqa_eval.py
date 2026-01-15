from ....evaluator import OPQAEvaluator
from ....template.eval.medmcqa_template import get_medmcqa_eval_template

class MedMCQAEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_medmcqa_eval_template(self.eval_args.lang)