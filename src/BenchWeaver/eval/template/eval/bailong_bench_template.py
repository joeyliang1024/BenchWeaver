from typing import Dict, List
from ..template import _register_eval_template
from ....data.data_utils import Role
from .multi_turn_template import Multi_Turn_Template

class Bailong_Bench_Template(Multi_Turn_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str, **kwargs):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response, **kwargs)
        # Store any unknown kwargs as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def format_checker_example(
        self, target_data: Dict[str, str], history: List[dict], **kwargs
    ) -> List[Dict[str, str]]:
                
        # get the question and answer turns
        question_turns, _ = self._parse_example(example=target_data)
        # get the assistant response
        assistant_turns = [turn["content"] for turn in history if turn["role"] == Role.ASSISTANT]
        
        if target_data["type"] == "Multi-turn":
            # use list extend for collecting prompts
            return  [[
                        {
                            "role": Role.USER.value, 
                            "content": self.criteria_prompt.format_map({
                                "question1": question_turns[0],
                                "answer1": assistant_turns[0],
                                "question2": question_turns[1],
                                "answer2": assistant_turns[1],
                                "question3": question_turns[2],
                                "answer3": assistant_turns[2],
                            })
                        }
                    ]]
        else:
            return  [[
                        {
                            "role": Role.USER.value, 
                            "content": self.criteria_prompt.format_map({
                                "question": question_turns[0], 
                                "answer": assistant_turns[0],
                            })
                        }
                    ]]
            
bailong_bench_eval_templates: Dict[str, "Bailong_Bench_Template"] = {}

def get_bailong_bench_eval_template(name: str) -> "Bailong_Bench_Template":
    eval_template = bailong_bench_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

sigle_turn_names = [
    "creative_writing",
    "mail_assistant",
    "health_consultation",
    "translation",
    "copywriting_generation",
    "knowledge_based_question",
    "summarization",
    "proofreading",
    "open_question",
    "morality_and_ethics",
    "general",
    "english_instruction",
    "arithemetic",
]
for name in sigle_turn_names:
    _register_eval_template(
        name=name,
        system="",
        choice="",
        answer="",
        cot="",
        templates=bailong_bench_eval_templates,
        template_class=Bailong_Bench_Template,
        criteria_prompt="請充當一位公正的評審，評估AI助手對以下顯示的用戶問題的回應品質。你的評估應該考慮回答的有用性、相關性、準確性、深度、創造力和細節水平。除非用戶要求，否則若用戶以英語提問而助理用中文回答則直接給0分，若用戶以中文提問而助理用英語回答也直接給0分。注意，回答中絕對不能出現簡體中文(simplified chinese)，如果助理使用簡體中文請直接給0分。提供解釋後，請按照嚴格的格式在0到10的範圍內評分，分數必須是整數，例如：“評分：[[5]]”。\n\n[用戶問題]\n{question}\n\n[助手回答開始]\n{answer}\n[助手回答結束]",
        response=""
    )

_register_eval_template(
    name="multi_turn",
    system="",
    choice="",
    answer="",
    cot="",
    templates=bailong_bench_eval_templates,
    template_class=Bailong_Bench_Template,
    criteria_prompt="請充當一位公正的評審，評估AI助手對以下顯示的用戶問題的回應品質。你的評估應該考慮回答的有用性、相關性、準確性、深度、創造力和細節水平。除非用戶要求，若用戶以中文提問而助理用英語回答也直接給0分。若用戶英語提問而助理用中文回答則直接給0分。注意，回答中絕對不能出現簡體中文(simplified chinese)，如果助理使用簡體中文請直接給0分。提供解釋後，請按照嚴格的格式在0到10的範圍內評分，分數必須是整數，例如：“評分：[[5]]”。\n\n[用戶問題]\n{question1}\n\n[助手回答]\n{answer1}\n\n[用戶問題]\n{question2}\n\n[助手回答]\n{answer2}\n\n[用戶問題]\n{question3}\n\n[助手回答]\n{answer3}",
    response=""
)