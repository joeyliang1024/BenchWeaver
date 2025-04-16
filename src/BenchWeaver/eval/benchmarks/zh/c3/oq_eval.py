from ....evaluator import OQEvaluator
from ....template import get_c3_eval_template

class C3OQEvaluator(OQEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_c3_eval_template(self.eval_args.lang)