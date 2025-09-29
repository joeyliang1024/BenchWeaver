from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

math_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_math_eval_template(name: str) -> "OPQA_Template":
    eval_template = math_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="",
    choice="",
    answer="",
    cot="\nLet's think step by step.\nAnswer:",
    templates=math_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="Determine whether the LLM Response correctly answer the question.\n\nQuestion: {question}\n\nReference Answer: {answer}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'.",
    response="{answer}"
)