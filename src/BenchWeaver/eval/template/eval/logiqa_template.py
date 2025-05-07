from typing import Dict, List, Sequence, Tuple
from ..template import _register_eval_template
from .mcqa_template import MCQA_Template
from ....data.data_utils import Role

class LogiQA_Template(MCQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response)

    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"context", "question", "answer", ...}
        output: a tuple of (prompt, response)
        """
        # format question
        question = self.system.format(
            context=example["context"],
            question=example["question"],
            option_a=example["A"],
            option_b=example["B"],
            option_c=example["C"],
            option_d=example["D"]
        )
        # format answer
        answer = ((example.get("explanation") if use_cot and example.get("explanation") else "") + "\n" + 
                  self.response.format(answer=example.get("answer"))).strip()

        return question, answer
    
    def format_inference_example(
        self, target_data: Dict[str, str], choices: List[str], support_set: Sequence[Dict[str, str]], subject_name: str, user_prompt: str, use_cot: bool
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        if support_set is not None:
            for k in range(len(support_set)):
                prompt, response = self._parse_example(support_set[k], choices)
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data, choices, use_cot=use_cot)
        messages.append({"role": Role.USER.value, "content": prompt})
        return messages
    
logiqa_eval_templates: Dict[str, "LogiQA_Template"] = {}

def get_logiqa_eval_template(name: str) -> "LogiQA_Template":
    eval_template = logiqa_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="Passage: {context}\n Question: {question} \n Choices:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD.{option_d}\nPlease choose the most suitable one among A, B, C and D as the answer to this question.",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    cot="\nLet's think step by step.\nAnswer: ",
    templates=logiqa_eval_templates,
    template_class=LogiQA_Template,
    criteria_prompt="You are an evaluation model and will be given a multiple-choice question, an LLM response, and an option {option}.  \nYour task is not to answer the question but to determine whether the LLM response explicitly selects {option} as one of the answers.  \n\nQuestion: {question}  \n\nLLM Response: {llm_response}  \n\nEvaluation criteria:  \n\nIf {option} is clearly and directly expressed as a selected answer in the LLM response, or if the LLM response contains only an option (A, B, C, D, etc.) or {option} itself, respond with 'True'.  \n\nIf {option} is not selected, or the LLM response does not clearly express an answer, respond with 'False'.  \n\nIf the LLM response is empty or it is unclear whether {option} is a selected answer, respond with 'Unknown'.",
    response="The correct answer is ({answer})."
)

_register_eval_template(
    name="zh",
    system="段落: {context}\n问题: {question}\n选择:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n请在 A、B、C 和 D 中选择最合适的一个作为此问题的答案",
    choice="\n{choice}. {content}",
    answer="\n答案: ",
    cot="\n让我们一步一步地思考。\n答案: ",
    templates=logiqa_eval_templates,
    template_class=LogiQA_Template,
    criteria_prompt="你是一个评估模型，将被给予一个选择题问题、一个 LLM 回应，以及一个选项 {option}。  \n你的任务不是回答问题，而是判断 LLM 回应中是否明确选择 {option} 作为答案之一。  \n\n问题: {question}  \n\nLLM 回应: {llm_response}  \n\n判断标准：  \n\n如果 {option} 在 LLM 回应中被清楚且直接表达为选择的答案，或 LLM 回应仅包含选项（A、B、C、D 等）或本身（{option}），则请回答 'True'。  \n\n如果 {option} 未被选择，或 LLM 回应未表达出明确的答案，请回答 'False'。  \n\n如果 LLM 回应为空，或其内容无法确定 {option} 是否为选择的答案，请回答 'Unknown'。",
    response="正确答案是 ({answer})。",
)

_register_eval_template(
    name="ko",
    system="구문: {context}\n질문: {question}\n선택:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n이 질문의 답으로 A, B, C 및 D 중 가장 적합한 것을 선택하고",
    choice="\n{choice}. {content}",
    answer="\n답: ",
    cot="\n단계별로 생각해 보겠습니다.\n답: ",
    templates=logiqa_eval_templates,
    template_class=LogiQA_Template,
    criteria_prompt="당신은 평가 모델이며, 하나의 객관식 질문, LLM 응답, 그리고 하나의 선택지 {option}을 받게 됩니다.  \n당신의 임무는 질문에 답하는 것이 아니라, LLM 응답에서 {option}이 명확하게 선택된 답변 중 하나인지 판단하는 것입니다.  \n\n질문: {question}  \n\nLLM 응답: {llm_response}  \n\n판단 기준:  \n\n{option}이 LLM 응답에서 명확하고 직접적으로 선택된 답변으로 표현되었거나, LLM 응답이 선택지(A, B, C, D 등) 또는 {option}만 포함하는 경우 'True'를 답하십시오.  \n\n{option}이 선택되지 않았거나, LLM 응답이 명확한 답을 표현하지 않았다면 'False'를 답하십시오.  \n\nLLM 응답이 비어 있거나, {option}이 선택된 답변인지 판단할 수 없다면 'Unknown'을 답하십시오.",
    response="정답은 ({answer})입니다.",
)