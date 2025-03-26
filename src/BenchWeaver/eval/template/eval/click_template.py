import ast
from typing import Dict, List, Tuple
from ..template import _register_eval_template
from ...template import MCQA_Template
from ....extras.constants import OPTION_CODES

class CLIcK_Template(MCQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.criteria_prompt = criteria_prompt
        self.response = response
        
    # override
    def _parse_example(self, example: Dict[str, str], choices: List[str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        """
        input: a dict with keys {"question", "paragraph", "choices", "answer"}
        output: a tuple of (prompt, response)
        """
        candidates_list = ast.literal_eval((example['choices']))
        question_candidates = [self.choice.format(choice=option, content=content) for option, content in zip(OPTION_CODES, candidates_list)]
        question = " ".join(
            [
                "" if example['paragraph'] == "" else "주어진 문맥을 읽고, 질문에 대한 올바른 답을 A, B, C, D 등에서 선택하세요 중에 골라 알파벳 하나로 답하시오.\n",
                "" if example['paragraph'] == "" else "맥락: {paragraph}\n".format(paragraph=example['paragraph']),
                "질문: {question}\n".format(question=example['question']),    
                "보기:\n"
            ] + 
            question_candidates + 
            [
                "정답:"
            ]
        ).strip()
        answer = self.response.format(answer=example['answer'])
        return question, answer
    
click_eval_templates: Dict[str, "CLIcK_Template"] = {}

def get_click_eval_template(name: str) -> "CLIcK_Template":
    eval_template = click_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="ko",
    system="다음은 {subject} 에 대한 객관식 질문(정답 포함)입니다.\n\n",
    choice="\n{choice}. {content}",
    answer="\n정답:",
    cot="\n단계별로 생각해 봅시다.\n답변:",
    templates=click_eval_templates,
    template_class=CLIcK_Template,
    criteria_prompt='다음 LLM 응답이 "{option}"을(를) 객관식 질문의 답으로 포함하는지 판단하세요.\n\n질문: {question}\n\nLLM 응답: {llm_response}\n\n"{option}"이(가) LLM 응답에서 명확하게 선택된 답이면 \'True\'를, 그렇지 않으면 \'False\'를 답하세요.',
    response="정답은 ({answer})입니다."
)