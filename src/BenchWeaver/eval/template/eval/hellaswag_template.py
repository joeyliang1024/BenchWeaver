from typing import Dict
from .mcqa_template import MCQA_Template
from ..template import _register_eval_template

hellaswag_eval_templates: Dict[str, "MCQA_Template"] = {}

def get_hellaswag_eval_template(name: str) -> "MCQA_Template":
    eval_template = hellaswag_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


_register_eval_template(
    name="en",
    system="Choose the most plausible ending for the given context from the options below. Read the context carefully and select the answer that best completes or continues the scenario in a coherent and natural way.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=hellaswag_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="You are an evaluation model and will be given a multiple-choice question, an LLM response, and an option {option}.  \nYour task is not to answer the question but to determine whether the LLM response explicitly selects {option} as one of the answers.  \n\nQuestion: {question}  \n\nLLM Response: {llm_response}  \n\nEvaluation criteria:  \n\nIf {option} is clearly and directly expressed as a selected answer in the LLM response, or if the LLM response contains only an option (A, B, C, D, etc.) or {option} itself, respond with 'True'.  \n\nIf {option} is not selected, or the LLM response does not clearly express an answer, respond with 'False'.  \n\nIf the LLM response is empty or it is unclear whether {option} is a selected answer, respond with 'Unknown'.",
    response = "The correct answer is ({answer})."
)