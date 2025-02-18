from typing import Dict
from .template import _register_eval_template
from .opqa_template import OPQA_Template

gpqa_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_gpqa_eval_template(name: str) -> "OPQA_Template":
    eval_template = gpqa_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="Question:  \n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=gpqa_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="Determine whether the LLM Response correctly answer the question.\n\nQuestion: {answer}\n\nReference Answer: {question}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'.",
)
