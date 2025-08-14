from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

huatuo_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_huatuo_eval_template(name: str) -> "OPQA_Template":
    eval_template = huatuo_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=huatuo_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Question]\n{question}\n[The End of Question]\n\n[The Start of Reference Answer]\n{answer}\n[The End of Reference Answer]\n\n[The Start of Assistant\'s Answer]\n{llm_response}\n[The End of Assistant\'s Answer]',
    response="{answer}"
)

_register_eval_template(
    name="zh",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n让我们一步一步地思考。\n",
    templates=huatuo_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n请扮演一位公正的评审员，评估 AI 助手对下方用户问题所提供回复的质量。你的评估应考虑回复的有用性、相关性、准确性、深度、创意，以及细节程度。请先提供简短的说明作为评估理由，并尽可能保持客观。在提供说明后，你必须按照以下格式给出 1 到 10 分的评分："[[rating]]"，例如："Rating: [[5]]"。\n\n[问题开始]\n{question}\n[问题结束]\n\n[参考答案开始]\n{answer}\n[参考答案结束]\n\n[AI 助手回复开始]\n{llm_response}\n[AI 助手回复结束]',
    response="{answer}"
)

_register_eval_template(
    name="zh-tw",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n讓我們一步一步地思考。\n",
    templates=huatuo_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n請扮演一位公正的評審員，評估 AI 助手對下方使用者問題所提供回覆的品質。你的評估應考慮回覆的有用性、相關性、準確性、深度、創意，以及細節程度。請先提供簡短的說明作為評估理由，並盡可能保持客觀。在提供說明後，你必須按照以下格式給出 1 到 10 分的評分： "[[rating]]"，例如："Rating: [[5]]"。\n\n[問題開始]\n{question}\n[問題結束]\n\n[參考答案開始]\n{answer}\n[參考答案結束]\n\n[AI 助手回覆開始]\n{llm_response}\n[AI 助手回覆結束]',
    response="{answer}"
)

_register_eval_template(
    name="ko",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n단계별로 생각해 봅시다.\n",
    templates=huatuo_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[지침]\n아래 사용자 질문에 대해 AI 어시스턴트가 제공한 응답의 품질을 공정한 심사위원으로서 평가해 주십시오. 평가는 유용성, 관련성, 정확성, 깊이, 창의성, 세부 정보 수준 등을 고려해야 합니다. 먼저 간단한 설명으로 평가 이유를 제시하고, 가능한 한 객관적으로 작성하십시오. 설명 후 반드시 다음 형식에 따라 1에서 10까지 점수를 부여해야 합니다: "[[rating]]", 예: "Rating: [[5]]".\n\n[질문 시작]\n{question}\n[질문 끝]\n\n[참고 답변 시작]\n{answer}\n[참고 답변 끝]\n\n[AI 어시스턴트 응답 시작]\n{llm_response}\n[AI 어시스턴트 응답 끝]',
    response="{answer}"
)
