from ....evaluator import OPQAEvaluator
from ....template.eval.gpqa_template import get_gpqa_eval_template

class GPQAEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_gpqa_eval_template(self.eval_args.lang)