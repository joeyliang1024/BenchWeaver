from ....evaluator import OPQAEvaluator
from ....template.eval.drcd_template import get_drcd_eval_template

class DRCDEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_drcd_eval_template(self.eval_args.lang)
        