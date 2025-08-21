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
    criteria_prompt='[Instruction]\nPlease act as a professional reviewer in the biomedical domain and evaluate the quality of the AI assistant\'s response to the user’s question below.\nYour evaluation should primarily consider the following aspects:\n\n1. Accuracy: Does the response match the reference answer? Does it contain errors or misleading information?  \n2. Completeness: Does the response cover the key points in the reference answer, without missing important information?  \n3. Relevance: Does the response stay focused on the question and the reference answer, avoiding unnecessary or irrelevant content?  \n4. Professionalism: Does the wording align with biomedical professional standards, avoiding vague or imprecise statements?  \n5. Clarity: Is the response well-structured and easy to understand?  \n\nAfter providing a brief evaluation explanation, please assign a score from 1 to 10 based on overall quality, in the format: "Rating: [[X]]", e.g., "Rating: [[5]]".\n\n[Question Start]\n{question}\n[Question End]\n\n[Reference Answer Start]\n{answer}\n[Reference Answer End]\n\n[AI Assistant Response Start]\n{llm_response}\n[AI Assistant Response End]\n',
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
    criteria_prompt='[指令]\n请扮演一位专业的生物医学领域评审员，评估 AI 助手对下方用户问题所提供回复的质量。\n你的评估应主要考虑以下方面：\n\n1. 正确性：回答是否与标准答案一致，是否存在错误或误导信息。  \n2. 完整性：回答是否涵盖标准答案中的关键要点，是否遗漏重要信息。  \n3. 相关性：回答是否聚焦于问题与标准答案，避免冗余或无关内容。  \n4. 专业性：用词是否符合生物医学专业表达，是否避免模糊或不严谨的说法。  \n5. 表达清晰度：回答是否结构清晰、易于理解。  \n\n在提供简短的评估说明后，请根据整体表现给出 1 到 10 分的评分，格式为："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[问题开始]\n{question}\n[问题结束]\n\n[参考答案开始]\n{answer}\n[参考答案结束]\n\n[AI 助手回复开始]\n{llm_response}\n[AI 助手回复结束]\n',
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
    criteria_prompt='[指令]\n請扮演一位專業的生醫領域評審員，評估 AI 助手對下方使用者問題所提供回覆的品質。\n你的評估應主要考慮以下面向：\n\n1. 正確性：回答是否與標準答案一致，是否存在錯誤或誤導資訊。  \n2. 完整性：回答是否涵蓋標準答案中的關鍵要點，是否有遺漏重要資訊。  \n3. 相關性：回答是否聚焦於問題與標準答案，避免冗餘或無關內容。  \n4. 專業性：用詞是否符合生醫專業表達，是否避免模糊或不嚴謹的說法。  \n5. 表達清晰度：回答是否結構清楚、易於理解。  \n\n在提供簡短的評估說明後，請根據整體表現給出 1 到 10 分的評分，格式為："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[問題開始]\n{question}\n[問題結束]\n\n[參考答案開始]\n{answer}\n[參考答案結束]\n\n[AI 助手回覆開始]\n{llm_response}\n[AI 助手回覆結束]\n',
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
    criteria_prompt='[지시]\n당신은 생의학 분야의 전문 심사위원 역할을 맡아, 아래 사용자 질문에 대한 AI 어시스턴트의 답변 품질을 평가하십시오.\n평가 시 주요하게 고려해야 할 요소는 다음과 같습니다:\n\n1. 정확성: 답변이 표준 정답과 일치하는가? 오류나 오해의 소지가 있는 정보가 있는가?  \n2. 완전성: 답변이 표준 정답의 핵심 포인트를 모두 포함하는가? 중요한 정보를 누락하지 않았는가?  \n3. 관련성: 답변이 질문과 표준 정답에 집중하고 불필요하거나 무관한 내용을 피하고 있는가?  \n4. 전문성: 표현이 생의학 전문 용어와 기준에 부합하는가? 모호하거나 부정확한 표현을 피했는가?  \n5. 명확성: 답변의 구조가 명확하고 이해하기 쉬운가?  \n\n간단한 평가 설명을 제공한 후, 전체적인 품질을 기준으로 1점에서 10점까지 점수를 부여하십시오. 형식은 다음과 같습니다: "Rating: [[X]]", 예: "Rating: [[5]]".\n\n[질문 시작]\n{question}\n[질문 종료]\n\n[참고 정답 시작]\n{answer}\n[참고 정답 종료]\n\n[AI 어시스턴트 답변 시작]\n{llm_response}\n[AI 어시스턴트 답변 종료]\n',
    response="{answer}"
)
