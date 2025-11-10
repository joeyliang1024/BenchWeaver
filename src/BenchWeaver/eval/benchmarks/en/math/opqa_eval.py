from ....evaluator import OPQAEvaluator
from ....template.eval.math_template import get_math_eval_template

class MATHEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_math_eval_template(self.eval_args.lang)
