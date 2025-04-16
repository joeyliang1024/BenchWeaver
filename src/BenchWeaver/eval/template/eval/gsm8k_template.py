from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

gsm8k_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_gsm8k_eval_template(name: str) -> "OPQA_Template":
    eval_template = gsm8k_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="Problem: \n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=gsm8k_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="Determine whether the LLM Response correctly answer the question.\n\nQuestion: {question}\n\nReference Answer: {answer}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'.",
    response="The correct answer is ({answer})."
)
