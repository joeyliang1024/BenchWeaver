from ....evaluator import OPQAEvaluator
from ....template import get_awesome_taiwan_knowledge_eval_template

class AwesomeTaiwanKnowledgeEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_awesome_taiwan_knowledge_eval_template(self.eval_args.lang)
        