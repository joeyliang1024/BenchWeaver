from typing import Dict, Tuple
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

class IFEval_Template(OPQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response)
        
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", ...}
        output: a tuple of (prompt, None)
        """
        # format question
        question = (example["question"] + (self.cot if use_cot else self.answer)).strip()
        
        return question, None
    
ifeval_eval_templates: Dict[str, "IFEval_Template"] = {}

def get_ifeval_eval_template(name: str) -> "IFEval_Template":
    eval_template = ifeval_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=ifeval_eval_templates,
    template_class=IFEval_Template,
    criteria_prompt="",
    response=""
)
    