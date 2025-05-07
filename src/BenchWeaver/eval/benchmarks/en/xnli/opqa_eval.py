from ....evaluator import OPQAEvaluator
from ....template import get_xnli_eval_template

class XNLIEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_xnli_eval_template(self.eval_args.lang)
        
