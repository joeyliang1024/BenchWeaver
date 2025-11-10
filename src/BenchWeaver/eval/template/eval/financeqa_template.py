from typing import Dict, Tuple
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

class FinanceQA_Template(OPQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response)
        
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"QUERY", "ANSWER", "CONTEXT",...}
        output: a tuple of (prompt, response)
        """
        # format question
        question = (
            "Context:\n"
            + example["CONTEXT"]
            + "\n\n"
            + example["QUERY"]
            + (self.cot if use_cot else self.answer)
        ).strip()
        # format answer
        answer = self.response.format(answer=example.get("ANSWER")).strip()

        return question, answer

financeqa_eval_templates: Dict[str, "FinanceQA_Template"] = {}

def get_financeqa_eval_template(name: str) -> "FinanceQA_Template":
    eval_template = financeqa_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="Please read the following article and answer the given question:\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=financeqa_eval_templates,
    template_class=FinanceQA_Template,
    criteria_prompt="Determine whether the LLM Response correctly answer the question.\n\nQuestion: {question}\n\nReference Answer: {answer}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'.",
    response="{answer}"
)