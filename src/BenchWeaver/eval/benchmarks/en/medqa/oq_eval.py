from ....evaluator import OQEvaluator
from ....template.eval.medqa_template import get_medqa_eval_template

class MedQAOQEvaluator(OQEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_medqa_eval_template(self.eval_args.lang)
        self.options_per_question = 4