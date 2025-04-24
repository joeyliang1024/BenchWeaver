from typing import Dict
from ..template import _register_eval_template
from .code_template import Code_Template

mbpp_eval_templates: Dict[str, "Code_Template"] = {}

def get_mbpp_eval_template(name: str) -> "Code_Template":
    eval_template = mbpp_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mbpp_eval_templates,
    template_class=Code_Template,
    criteria_prompt="",
    response=""
)