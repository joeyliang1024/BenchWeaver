from typing import Dict
from .mcqa_template import MCQA_Template
from ..template import _register_eval_template

arc_challenge_eval_templates: Dict[str, "MCQA_Template"] = {}

def get_arc_challenge_eval_template(name: str) -> "MCQA_Template":
    eval_template = arc_challenge_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers).\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=arc_challenge_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt='Determine whether the following response identifies "{option}" as the answer to the multiple-choice question.\n\nQuestion: {question}\n\nLLM Response: {llm_response}\n\nIf the LLM Response is correct, just response \'True\', else response \'False\'.',
)