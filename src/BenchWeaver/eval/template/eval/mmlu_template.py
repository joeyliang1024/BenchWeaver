from typing import Dict
from .mcqa_template import MCQA_Template
from ..template import _register_eval_template

mmlu_eval_templates: Dict[str, "MCQA_Template"] = {}

def get_mmlu_eval_template(name: str) -> "MCQA_Template":
    eval_template = mmlu_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=mmlu_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="You are an evaluation model and will be given a multiple-choice question, an LLM response, and an option {option}.  \nYour task is not to answer the question but to determine whether the LLM response explicitly selects {option} as the answer.  \n\nQuestion: {question}  \n\nLLM Response: {llm_response}  \n\nEvaluation criteria:  \n\nIf {option} is clearly and directly expressed as the chosen answer in the LLM response, respond with 'True'.  \n\nIf {option} is not selected, or the LLM response does not clearly express an answer, respond with 'False'.  \n\nIf the LLM response is empty or it is unclear whether {option} is the chosen answer, respond with 'Unknown'.  \n\n",
    response = "The correct answer is ({answer})."
)

_register_eval_template(
    name="zh-tw",
    system="以下是台灣關於{subject}考試的單項選擇題，請選出其中的正確答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n讓我們一步一步來思考。\n答案：",
    templates=mmlu_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="你是一個評估模型，將被給予一個選擇題問題、一個 LLM 回應，以及一個選項 {option}。\n你的任務不是回答問題，而是判斷 LLM 回應中是否明確選擇了 {option} 作為答案。\n\n問題: {question}\n\nLLM 回應: {llm_response}\n\n判斷標準：\n\n如果 {option} 在 LLM 回應中清楚且直接被表達為選擇的答案，請回答 'True'。\n\n如果 {option} 未被選擇，或 LLM 回應未表達出明確的答案，請回答 'False'。\n\n如果 LLM 回應為空，或其內容無法確定 {option} 是否為選擇的答案，請回答 'Unknown'。",
    response="正確答案是 ({answer})。",
)

_register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n让我们一步一步来思考。\n答案：",
    templates=mmlu_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="你是一个评估模型，将被给予一个选择题问题、一个 LLM 回应，以及一个选项 {option}。  \n你的任务不是回答问题，而是判断 LLM 回应中是否明确选择了 {option} 作为答案。  \n\n问题: {question}  \n\nLLM 回应: {llm_response}  \n\n判断标准：  \n\n如果 {option} 在 LLM 回应中清楚且直接被表达为选择的答案，请回答 'True'。  \n\n如果 {option} 未被选择，或 LLM 回应未表达出明确的答案，请回答 'False'。  \n\n如果 LLM 回应为空，或其内容无法确定 {option} 是否为选择的答案，请回答 'Unknown'。",
    response="正确答案是 ({answer})。",
)

_register_eval_template(
    name="ko",
    system="다음은 {subject}에 대한 객관식 질문(및 정답)입니다.\n\n",
    choice="\n{choice}. {content}",
    answer="\n정답:",
    cot="\n차근차근 생각해 봅시다.\n정답:",
    templates=mmlu_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt="당신은 평가 모델이며, 하나의 객관식 질문, LLM 응답, 그리고 하나의 선택지 {option}을 받게 됩니다.  \n당신의 임무는 질문에 답하는 것이 아니라, LLM 응답에서 {option}이 명확하게 선택된 답변인지 판단하는 것입니다.  \n\n질문: {question}  \n\nLLM 응답: {llm_response}  \n\n판단 기준:  \n\n{option}이 LLM 응답에서 명확하고 직접적으로 선택된 답변으로 표현되었다면 'True'를 답하십시오.  \n\n{option}이 선택되지 않았거나, LLM 응답이 명확한 답을 표현하지 않았다면 'False'를 답하십시오.  \n\nLLM 응답이 비어 있거나, {option}이 선택된 답변인지 판단할 수 없다면 'Unknown'을 답하십시오.\n",
    response="정답은 ({answer})입니다.",
)