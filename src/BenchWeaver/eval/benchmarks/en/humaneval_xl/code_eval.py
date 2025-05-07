from ....evaluator.code.code_evaluator import CodeEvaluator
from ....template import get_humaneval_xl_eval_template

class HumanEvalXLEvaluator(CodeEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_humaneval_xl_eval_template(self.eval_args.lang)