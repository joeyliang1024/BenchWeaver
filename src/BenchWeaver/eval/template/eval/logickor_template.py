from typing import Dict, List
from ..template import _register_eval_template
from ....data.data_utils import Role
from .multi_turn_template import Multi_Turn_Template
from ..source.logickor_template import PROMPT_STRATEGY, JUDGE_TEMPLATE

class LogicKor_Template(Multi_Turn_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str, **kwargs):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response, **kwargs)
        # Store any unknown kwargs as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def format_inference_example(
        self, target_data: Dict[str, str], history: List[dict], use_cot: bool=False, **kwargs
    ) -> List[Dict[str, str]]:
        """
        Format the inference example for multi-turn evaluation.
        Can specify `use_cot` to use chain-of-thought reasoning.
        """
        # count the number of turns in the history
        turns_idx = 0 if (history == [] or history is None) else sum([1 for turn in history if turn["role"] == Role.ASSISTANT])
        # get the question and answer turns
        question_turns, _ = self._parse_example(example=target_data)
        if turns_idx == 0:
            history = PROMPT_STRATEGY["cot-1-shot"] if use_cot else PROMPT_STRATEGY["1-shot"]
        history.append({
            "role": Role.USER,
            "content": question_turns[turns_idx]
        })
        return history
    
    def format_checker_example(
        self, target_data: Dict[str, str], history: List[dict], **kwargs
    ) -> List[List[Dict[str, str]]]:
        """
        Prompt consturction for the LogicKor evaluation.
        Source code: https://github.com/instructkr/LogicKor/blob/main/evaluator.py
        """
        # get the question and answer turns
        question_turns, answer_turns = self._parse_example(example=target_data)
        # get the assistant response
        assistant_turns = [turn["content"] for turn in history if turn["role"] == Role.ASSISTANT]
        turn1_prompt = (
            f"아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. 특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
            f"**Question**\n{question_turns[0]}"
        )
        # Format turn 1 prompt
        turn1_prompt += f"\n\n**Additional Reference**\n{answer_turns[0]}"
        turn1_prompt += f"\n\n**Model's Response**\n{assistant_turns[0]}"
        # Format turn 2 prompt
        turn2_prompt = turn1_prompt
        turn2_prompt += f"\n\n**Follow-up Question.**\n{question_turns[1]}"
        turn2_prompt += f"\n\n**Additional Reference**\n{answer_turns[1]}"
        turn2_prompt += f"\n\n**Model's Response**\n{assistant_turns[1]}"
        # End the 2 prompts
        turn1_prompt += "\n\n[[대화 종료. 평가 시작.]]"
        turn2_prompt += "\n\n[[대화 종료. 평가 시작.]]"
        
        return [
            # turn 1 message
            [
                {
                    "role":  Role.SYSTEM.value,
                    "content": JUDGE_TEMPLATE['single_turn']
                },
                {
                    "role": Role.USER.value, 
                    "content": turn1_prompt
                }
            ],
            # turn 2 message
            [
                {
                    "role": Role.SYSTEM.value,
                    "content": JUDGE_TEMPLATE['multi_turn']
                },
                {
                    "role": Role.USER.value, 
                    "content": turn2_prompt
                }
            ]
        ]
        
logickor_eval_templates: Dict[str, "LogicKor_Template"] = {}

def get_logickor_eval_template(name: str) -> "LogicKor_Template":
    eval_template = logickor_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

##### Register the LogicKor evaluation template #####
_register_eval_template(
    name="Reasoning",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=logickor_eval_templates,
    template_class=LogicKor_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='',
    turn_2_criteria_prompt='',
)

_register_eval_template(
    name="Math",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=logickor_eval_templates,
    template_class=LogicKor_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='',
    turn_2_criteria_prompt='',
)

_register_eval_template(
    name="Writing",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=logickor_eval_templates,
    template_class=LogicKor_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='',
    turn_2_criteria_prompt='',
)

_register_eval_template(
    name="Coding",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=logickor_eval_templates,
    template_class=LogicKor_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='',
    turn_2_criteria_prompt='',
)

_register_eval_template(
    name="Understanding",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=logickor_eval_templates,
    template_class=LogicKor_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='',
    turn_2_criteria_prompt='',
)

_register_eval_template(
    name="Grammar",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=logickor_eval_templates,
    template_class=LogicKor_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='',
    turn_2_criteria_prompt='',
)