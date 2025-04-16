import ast
from typing import Dict, List, Tuple
from ..template import _register_eval_template
from ...template import MCQA_Template
from ....extras.constants import OPTION_CODES

class C3_Template(MCQA_Template):
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
        question = "\n".join(
            [
                self.system,
                "",
                "{paragraph}".format(paragraph=example['paragraph']),
                "问题：{question}".format(question=example['question']),
                "",
            ] + 
            question_candidates + 
            [
                self.answer
            ]
        ).strip()
        answer = self.response.format(answer=example['answer'])
        return question, answer


c3_eval_templates: Dict[str, "C3_Template"] = {}

def get_c3_eval_template(name: str) -> "C3_Template":
    eval_template = c3_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="zh",
    system="请根据所提供的文章或对话，选择最符合问题的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n让我们一步一步来思考。\n答案：",
    templates=c3_eval_templates,
    template_class=C3_Template,
    criteria_prompt="你是一个评估模型，将被给予一个选择题问题、一个 LLM 回应，以及一个选项 {option}。  \n你的任务不是回答问题，而是判断 LLM 回应中是否明确选择 {option} 作为答案之一。  \n\n问题: {question}  \n\nLLM 回应: {llm_response}  \n\n判断标准：  \n\n如果 {option} 在 LLM 回应中被清楚且直接表达为选择的答案，或 LLM 回应仅包含选项（A、B、C、D 等）或本身（{option}），则请回答 'True'。  \n\n如果 {option} 未被选择，或 LLM 回应未表达出明确的答案，请回答 'False'。  \n\n如果 LLM 回应为空，或其内容无法确定 {option} 是否为选择的答案，请回答 'Unknown'。",
    response="正确答案是 ({answer})。",
)