from typing import Dict, List
from ..template import _register_eval_template
from ....data.data_utils import Role
from .multi_turn_template import Multi_Turn_Template

class MT_Bench_TW_Template(Multi_Turn_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str, **kwargs):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response, **kwargs)
        # Store any unknown kwargs as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def format_checker_example(
        self, target_data: Dict[str, str], history: List[dict], **kwargs
    ) -> List[List[Dict[str, str]]]:
                
        # get the question and answer turns
        question_turns, answer_turns = self._parse_example(example=target_data)
        # get the assistant response
        assistant_turns = [turn["content"] for turn in history if turn["role"] == Role.ASSISTANT]
        # format ref_block
        ref_block = "\n".join(
            f"User:\n{q}\n\nReference answer:\n{r}" for q, r in zip(question_turns, answer_turns)
        )

        assistant_block = "\n".join(
            f"User:\n{q}\n\nAssistant:\n{a}" for q, a in zip(question_turns, assistant_turns)
        )
        return [
            # turn 1 message
            [
                {
                    "role": Role.USER.value, 
                    "content": self.turn_1_criteria_prompt.format(
                        ref_answer=answer_turns[0],
                        question=question_turns[0],
                        llm_response=assistant_turns[0]
                        )
                }
            ],
            # turn 2 message
            [
                {
                    "role": Role.USER.value, 
                    "content": self.turn_2_criteria_prompt.format(
                        ref_block=ref_block,
                        assistant_block=assistant_block
                        )
                }
            ]
        ]
        
mt_bench_tw_eval_templates: Dict[str, "MT_Bench_TW_Template"] = {}

def get_mt_bench_tw_eval_template(name: str) -> "MT_Bench_TW_Template":
    eval_template = mt_bench_tw_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="writing",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on the assistant\'s answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)

_register_eval_template(
    name="roleplay",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on the assistant\'s answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)

_register_eval_template(
    name="reasoning",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Your evaluation should focus on the assistant\'s answer to the second question. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Reference Answer]\n\n{ref_block}\n\n[The End of Reference Answer]\n\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)

_register_eval_template(
    name="math",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Your evaluation should focus on the assistant\'s answer to the second question. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Reference Answer]\n\n{ref_block}\n\n[The End of Reference Answer]\n\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)

_register_eval_template(
    name="coding",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Your evaluation should focus on the assistant\'s answer to the second question. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Reference Answer]\n\n{ref_block}\n\n[The End of Reference Answer]\n\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)

_register_eval_template(
    name="extraction",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on the assistant\'s answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)

_register_eval_template(
    name="stem",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on the assistant\'s answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)

_register_eval_template(
    name="humanities",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=mt_bench_tw_eval_templates,
    template_class=MT_Bench_TW_Template,
    criteria_prompt="",
    response="",
    turn_1_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    turn_2_criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on the assistant\'s answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Assistant\'s Conversation with User]\n\n{assistant_block}\n\n[The End of Assistant\'s Conversation with User]',
)