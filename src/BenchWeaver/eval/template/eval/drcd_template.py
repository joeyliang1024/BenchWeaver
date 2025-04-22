from typing import Dict, Tuple
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

class DRCD_Template(OPQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response)

    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"context", "question", "answer", ...}
        output: a tuple of (prompt, response)
        """
        # format question
        question = (
            self.system + 
            example["context"] + "\n" +
            "問題：" + 
            example["question"] + 
            (self.cot if use_cot else self.answer)
            ).strip()
        # format answer
        answer = ((example.get("explanation") if use_cot and example.get("explanation") else "") + "\n" + 
                  self.response.format(answer=example.get("answer"))).strip()

        return question, answer

drcd_eval_templates: Dict[str, "DRCD_Template"] = {}

def get_drcd_eval_template(name: str) -> "DRCD_Template":
    eval_template = drcd_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="zh-tw",
    system="請閱讀以下的文章回答給定的問題：\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n讓我們一步一步來思考。\n答案：",
    templates=drcd_eval_templates,
    template_class=DRCD_Template,
    criteria_prompt="判斷 LLM 回應是否正確回答了問題。\n\n問題: {question}\n\n參考答案: {answer}\n\nLLM 回應: {llm_response}\n\n如果 LLM 回應正確，請回答 \'True\'，否則回答 \'False\'。",
    response="正確答案是{answer}。"
)
