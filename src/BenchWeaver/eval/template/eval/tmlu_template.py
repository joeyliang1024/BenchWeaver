import ast
from typing import Dict, List, Tuple
from ..template import _register_eval_template
from ...template import MCQA_Template
from ....extras.constants import OPTION_CODES

class TMLU_Template(MCQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system=system
        self.choice=choice
        self.answer=answer
        self.cot=cot
        self.criteria_prompt=criteria_prompt
        self.response=response
        
    def _parse_example(self, example, choices, use_cot = False, **kwargs):
        """
        input: a dict with keys {"question", "choices", "answer"}
        output: a tuple of (prompt, response)
        """
        candidates_list = ast.literal_eval((example['choices']))
        question_candidates = [self.choice.format(choice=option, content=content) for option, content in zip(OPTION_CODES, candidates_list)]
        question = " ".join(
            ['問題：{question}\n'.format(question=example['question'])] +
            question_candidates + 
            [self.cot if use_cot else self.answer]
        ).strip()
        
        answer = self.response.format(answer=candidates_list.index(example['answer']))
        return question, answer
    
tmlu_eval_templates: Dict[str, "TMLU_Template"] = {}

def get_tmlu_eval_template(name: str) -> "TMLU_Template":
    eval_template = tmlu_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="zh-tw",
    system="以下是台灣關於{subject}考試的單項選擇題，請選出其中的正確答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n讓我們一步一步來思考。\n答案：",
    templates=tmlu_eval_templates,
    template_class=TMLU_Template,
    criteria_prompt='判斷以下 LLM 回應是否包含 "{option}" 作為多項選擇題的答案。\n\n問題: {question}\n\nLLM 回應: {llm_response}\n\n如果 "{option}" 確實是 LLM 回應中明確選擇的答案，請回答 \'True\'，否則回答 \'False\'。',
    response="正確答案是 ({answer})。",
)
        