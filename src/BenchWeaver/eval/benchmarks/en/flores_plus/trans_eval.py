from ....evaluator.trans.trans_evaluator import TransEvaluator
from ....template import get_flores_eval_template

class FloresPlusEvaluator(TransEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_flores_eval_template(self.eval_args.lang)