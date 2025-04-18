from ....evaluator import OQEvaluator
from ....template import get_hellaswag_eval_template

class HellaSwagOQEvaluator(OQEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_hellaswag_eval_template(self.eval_args.lang)