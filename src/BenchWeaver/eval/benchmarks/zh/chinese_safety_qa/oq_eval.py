from ....evaluator import OQEvaluator
from ....template import get_chinese_safety_qa_eval_template

class ChineseSafetyQAOQEvaluator(OQEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_chinese_safety_qa_eval_template(self.eval_args.lang)