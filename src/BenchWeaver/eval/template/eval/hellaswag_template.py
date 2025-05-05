from typing import Dict
from .mcqa_template import MCQA_Template
from ..template import _register_eval_template

hellaswag_eval_templates: Dict[str, "MCQA_Template"] = {}

def get_hellaswag_eval_template(name: str) -> "MCQA_Template":
    eval_template = hellaswag_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


_register_eval_template(
    name="en",
    system="Choose the most plausible ending for the given context from the options below. Read the context carefully and select the answer that best completes or continues the scenario in a coherent and natural way.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=hellaswag_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="You are an evaluation model and will be given a multiple-choice question, an LLM response, and an option {option}.  \nYour task is not to answer the question but to determine whether the LLM response explicitly selects {option} as one of the answers.  \n\nQuestion: {question}  \n\nLLM Response: {llm_response}  \n\nEvaluation criteria:  \n\nIf {option} is clearly and directly expressed as a selected answer in the LLM response, or if the LLM response contains only an option (A, B, C, D, etc.) or {option} itself, respond with 'True'.  \n\nIf {option} is not selected, or the LLM response does not clearly express an answer, respond with 'False'.  \n\nIf the LLM response is empty or it is unclear whether {option} is a selected answer, respond with 'Unknown'.",
    response = "The correct answer is ({answer})."
)

_register_eval_template(
    name="zh",
    system="选择给定上下文中最合理的结尾。仔细阅读上下文，并选择最能完成或继续场景的答案，使其连贯自然。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案:",
    cot="\n让我们一步一步地思考。\n答案:",
    templates=hellaswag_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="你是一个评估模型，将会得到一个多选问题，一个LLM的回答和一个选项{option}。\n你的任务不是回答问题，而是判断LLM的回答是否明确选择了{option}作为其中一个答案。\n\n问题: {question}  \n\nLLM的回答: {llm_response}  \n\n评估标准:  \n\n如果{option}在LLM的回答中被明确和直接地表达为选定的答案，或者LLM的回答中仅包含一个选项（A、B、C、D等）或{option}本身，则回答'True'。  \n\n如果{option}没有被选中，或者LLM的回答没有明确表达出答案，则回答'False'。  \n\n如果LLM的回答为空，或者不清楚{option}是否是选定的答案，则回答'Unknown'。",
    response = "正确答案是 ({answer})."
)

_register_eval_template(
    name="ko",
    system="주어진 맥락에 대해 가장 그럴듯한 결말을 선택하세요. 맥락을 주의 깊게 읽고 시나리오를 일관되고 자연스럽게 완성하거나 계속하는 답변을 선택하세요.\n\n",
    choice="\n{choice}. {content}",
    answer="\n답변:",
    cot="\n단계별로 생각해 봅시다.\n답변:",
    templates=hellaswag_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="당신은 평가 모델이며 다중 선택 질문, LLM 응답 및 옵션 {option}을 받게 됩니다.\n당신의 임무는 질문에 답하는 것이 아니라 LLM 응답이 {option}을 선택된 답변 중 하나로 명시적으로 선택했는지 여부를 판단하는 것입니다.\n\n질문: {question}  \n\nLLM 응답: {llm_response}  \n\n평가 기준:  \n\nLLM 응답에서 {option}이 선택된 답변으로 명확하고 직접적으로 표현되거나 LLM 응답에 옵션(A, B, C, D 등) 또는 {option} 자체만 포함되어 있는 경우 'True'로 응답합니다.  \n\n{option}이 선택되지 않았거나 LLM 응답이 명확한 답변을 표현하지 않는 경우 'False'로 응답합니다.  \n\nLLM 응답이 비어 있거나 {option}이 선택된 답변인지 불분명한 경우 'Unknown'으로 응답합니다.",
    response = "정답은 ({answer})입니다."
)