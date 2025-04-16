from ....evaluator import OPQAEvaluator
from ....template import get_cif_bench_eval_template
class CifBenchEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_cif_bench_eval_template(self.eval_args.lang)
