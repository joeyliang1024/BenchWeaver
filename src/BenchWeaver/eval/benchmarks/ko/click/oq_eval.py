from ....evaluator import OQEvaluator
from ....template import get_click_eval_template

class CLIcKEvaluator(OQEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_click_eval_template(self.eval_args.lang)