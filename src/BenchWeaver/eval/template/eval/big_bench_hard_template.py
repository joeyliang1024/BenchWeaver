import json
import ast
from typing import Dict, List, Literal, Sequence, Tuple, Union
from ..template import EvalTemplate, _register_eval_template
from ....data.data_utils import Role
from ....extras.constants import OPTION_CODES

class BigBenchHard_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.mcqa_criteria_prompt = 'Determine whether the following LLM response includes "{option}" as the answer to the multiple-choice question.\n\nQuestion: {question}\n\nLLM Response: {llm_response}\n\nIf "{option}" is explicitly selected as the answer in the LLM response, please answer \'True\', otherwise answer \'False\'.'
        self.opqa_criteria_prompt = "Determine whether the LLM Response correctly answer the question.\n\nQuestion: {answer}\n\nReference Answer: {question}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'."
        self.response = response
        
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, *args) -> Tuple[str, str]:
        # format question
        question = example["question"] + (self.cot if use_cot else self.answer)
        # format answer
        answer =  example.get('explanation') if use_cot else example["answer"]
        return question, answer
    
    def format_inference_example(
         self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], user_prompt: str, use_cot: bool
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        if support_set is not None:
            for k in range(len(support_set)):
                prompt, response = self._parse_example(example=support_set[k], use_cot=use_cot)
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": response})
        
        prompt, response = self._parse_example(example=target_data, use_cot=use_cot)
        messages.append({"role": Role.USER.value, "content": prompt})
        if user_prompt is not None:
            messages[0]["content"] = user_prompt + messages[0]["content"]
        else:
            messages[0]["content"] = self.system + messages[0]["content"]
        return messages
    
    def format_checker_example(
        self, target_data: Dict[str, str], is_mcqa:bool, llm_response: str) -> Union[
            List[Dict[str, str]],
            Tuple[List[List[Dict[str, str]]], List[str]]
            ]:
        # mcqa
        if is_mcqa:
            check_msg_list = []
            answer_list = []
            choices:list =  ast.literal_eval(target_data['choices'])
            lables:list = json.loads(target_data['labels'])
            assert len(choices) == len(lables), f"Error: choices and labels length mismatch.\nchoices: {choices}, labels: {lables}"
            for content, label in zip(choices, lables):
                check_msg_list.append([
                    {
                        "role": Role.USER.value, 
                        "content": self.mcqa_criteria_prompt.format(
                            option=content,
                            question=target_data['question'],
                            llm_response=llm_response,
                            )
                    }
                ])
                answer_list.append(
                    "True".lower() if label == 1 else "False".lower()
                )
            return check_msg_list, answer_list
        # opqa
        else:
            return [
                 {
                     "role": Role.USER.value, 
                     "content": self.opqa_criteria_prompt.format(
                         answer=target_data['answer'],
                         question=target_data['question'],
                         llm_response=llm_response,
                         )
                 }
             ]
           
big_bench_hard_eval_templates: Dict[str, "BigBenchHard_Template"] = {}

def get_big_bench_hard_eval_template(name: str) -> "BigBenchHard_Template":
    eval_template = big_bench_hard_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="disambiguation_qa",
    system="Clarify the meaning of sentences with ambiguous pronouns.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)
_register_eval_template(
    name="formal_fallacies",
    system="Distinguish deductively valid arguments from formal fallacies.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)
_register_eval_template(
    name="geometric_shapes",
    system="Name geometric shapes from their SVG paths.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)
_register_eval_template(
    name="hyperbaton",
    system="Order adjectives correctly in English sentences.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)
_register_eval_template(
    name="object_counting",
    system="Questions that involve enumerating objects and asking the model to count them.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)
_register_eval_template(
    name="penguins_in_a_table",
    system="Answer questions about a table of penguins and their attributes.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)
_register_eval_template(
    name="salient_translation_error_detection",
    system="Detect the type of error in an English translation of a German source sentence.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)

_register_eval_template(
    name="tracking_shuffled_objects_five_objects",
    system="A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=big_bench_hard_eval_templates,
    template_class=BigBenchHard_Template,
    criteria_prompt="",
    response="{answer}"
)