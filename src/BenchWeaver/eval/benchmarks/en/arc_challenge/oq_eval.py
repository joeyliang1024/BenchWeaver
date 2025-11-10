from ....evaluator import OQEvaluator
from ....template.eval.arc_challenge_template import get_arc_challenge_eval_template

class ArcChallengeOQEvaluator(OQEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_arc_challenge_eval_template(self.eval_args.lang)
