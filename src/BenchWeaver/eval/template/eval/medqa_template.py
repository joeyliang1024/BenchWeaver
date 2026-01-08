# from typing import Dict
# from .mcqa_template import MCQA_Template
# from ..template import _register_eval_template
# 
# medqa_eval_templates: Dict[str, "MCQA_Template"] = {}
# 
# def get_medqa_eval_template(name: str) -> "MCQA_Template":
#     eval_template = medqa_eval_templates.get(name, None)
#     assert eval_template is not None, "Template {} does not exist.".format(name)
#     return eval_template
# 
# _register_eval_template(
#     name="en",
#     system="The following are multiple choice questions (with answers).\n\n",
#     choice="\n{choice}. {content}",
#     answer="\nAnswer:",
#     cot="\nLet's think step by step.\nAnswer:",
#     templates=medqa_eval_templates,
#     template_class=MCQA_Template,
#     criteria_prompt="You are an evaluation model and will be given a multiple-choice question, an LLM response, and an option {option}.  \nYour task is not to answer the question but to determine whether the LLM response explicitly selects {option} as one of the answers.  \n\nQuestion: {question}  \n\nLLM Response: {llm_response}  \n\nEvaluation criteria:  \n\nIf {option} is clearly and directly expressed as a selected answer in the LLM response, or if the LLM response contains only an option (A, B, C, D, etc.) or {option} itself, respond with 'True'.  \n\nIf {option} is not selected, or the LLM response does not clearly express an answer, respond with 'False'.  \n\nIf the LLM response is empty or it is unclear whether {option} is a selected answer, respond with 'Unknown'.",
#     response = "The correct answer is ({answer})."
# )
from typing import Dict, Tuple
from .opqa_template import OPQA_Template
from ..template import _register_eval_template

class MedQA_Template(OPQA_Template):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "answer", ...}
        output: a tuple of (prompt, response)
        """
        # format question
        question = "".join([example["question"]] + 
                           [self.choice.format(choice=ch, content=example[ch]) for ch in ["A", "B", "C", "D"] if ch in example] +
                           [self.cot if use_cot else self.answer]
                           ).strip()
        # question = (example["question"] + (self.cot if use_cot else self.answer)).strip()
        # format answer
        answer = ((example.get("explanation") if use_cot and example.get("explanation") else "") + "\n" + 
                  self.response.format(answer=example.get("answer"))).strip()

        return question, answer

medqa_eval_templates: Dict[str, "MedQA_Template"] = {}

def get_medqa_eval_template(name: str) -> "MedQA_Template":
    eval_template = medqa_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers).\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=medqa_eval_templates,
    template_class=MedQA_Template,
    criteria_prompt="Determine whether the LLM Response correctly answer the single-choice question.\n\nQuestion: {question}\n\nReference Answer: {answer}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'.",
    response="The correct answer is ({answer})."
)