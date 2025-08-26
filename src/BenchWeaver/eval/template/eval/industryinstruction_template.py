from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

industryinstruction_aerospace_eval_templates: Dict[str, "OPQA_Template"] = {}
industryinstruction_finance_eval_templates: Dict[str, "OPQA_Template"] = {}
industryinstruction_law_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_industryinstruction_aerospace_eval_template(name: str) -> "OPQA_Template":
    eval_template = industryinstruction_aerospace_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

def get_industryinstruction_finance_eval_template(name: str) -> "OPQA_Template":
    eval_template = industryinstruction_finance_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

def get_industryinstruction_law_eval_template(name: str) -> "OPQA_Template":
    eval_template = industryinstruction_law_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

# Aerospace
_register_eval_template(
    name="zh-tw",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n讓我們一步一步地思考。\n",
    templates=industryinstruction_aerospace_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n請扮演一位專業的航空航天領域評審員，評估 AI 助手對下方使用者問題所提供回覆的品質。\n你的評估應主要考慮以下面向：\n\n1. 正確性：回答是否與標準答案一致，是否存在錯誤或可能誤導的航空資訊。  \n2. 完整性：回答是否涵蓋標準答案中的關鍵要點，是否遺漏重要技術細節或規範。  \n3. 相關性：回答是否聚焦於問題主題與參考答案，避免無關或多餘的敘述。  \n4. 專業性：是否使用正確的航空航天專業術語與表達，避免模糊或非專業的說法。  \n5. 表達清晰度：回答是否結構清晰，能幫助讀者準確理解航空知識或規程。  \n\n在提供簡短的評估說明後，請根據整體表現給出 1 到 10 分的評分，格式為："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[問題開始]\n{question}\n[問題結束]\n\n[參考答案開始]\n{answer}\n[參考答案結束]\n\n[AI 助手回覆開始]\n{llm_response}\n[AI 助手回覆結束]\n',
    response="{answer}"
)
_register_eval_template(
    name="en",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=industryinstruction_aerospace_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[Instruction]\nPlease act as a professional aerospace domain reviewer and evaluate the quality of the AI assistant\'s response to the following user question.\nYour evaluation should mainly consider the following aspects:\n\n1. Accuracy: Does the answer align with the reference answer, and is there any incorrect or misleading aerospace information?  \n2. Completeness: Does the answer cover the key points of the reference answer, or does it miss important technical details or regulations?  \n3. Relevance: Does the answer stay focused on the question and the reference answer, avoiding irrelevant or redundant content?  \n4. Professionalism: Does the response use proper aerospace terminology and precise expressions, avoiding vague or non-professional language?  \n5. Clarity of Expression: Is the answer clearly structured and easy to understand, helping the reader accurately grasp aerospace knowledge or procedures?  \n\nAfter providing a brief evaluation, please assign a score from 1 to 10 based on the overall performance, in the format: "Rating: [[X]]", for example, "Rating: [[5]]".\n\n[Question Start]\n{question}\n[Question End]\n\n[Reference Answer Start]\n{answer}\n[Reference Answer End]\n\n[AI Assistant Response Start]\n{llm_response}\n[AI Assistant Response End]\n',
    response="{answer}"
)
_register_eval_template(
    name="ko",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n단계별로 생각해 봅시다.\n",
    templates=industryinstruction_aerospace_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[지시사항]\n전문적인 항공우주 분야 평가자로서, 아래 사용자 질문에 대한 AI 조수의 답변 품질을 평가해 주십시오.\n평가는 주로 다음 기준을 고려해야 합니다:\n\n1. 정확성: 답변이 기준 답변과 일치하는지, 잘못되거나 오해의 소지가 있는 항공우주 정보가 있는지.  \n2. 완전성: 기준 답변의 핵심 요점을 모두 포함하는지, 중요한 기술적 세부사항이나 규정을 누락하지 않았는지.  \n3. 관련성: 답변이 질문과 기준 답변에 집중했는지, 불필요하거나 관련 없는 내용을 피했는지.  \n4. 전문성: 항공우주 전문 용어와 표현을 정확히 사용했는지, 모호하거나 비전문적인 언어를 피했는지.  \n5. 명확성: 답변이 구조적으로 명확하고 이해하기 쉬워서 독자가 항공우주 지식이나 절차를 정확히 파악할 수 있는지.  \n\n간단한 평가 설명을 제공한 후, 전체적인 성과에 따라 1에서 10 사이의 점수를 "Rating: [[X]]" 형식으로 제시하십시오. 예: "Rating: [[5]]".\n\n[질문 시작]\n{question}\n[질문 끝]\n\n[기준 답변 시작]\n{answer}\n[기준 답변 끝]\n\n[AI 조수 답변 시작]\n{llm_response}\n[AI 조수 답변 끝]\n',
    response="{answer}"
)
_register_eval_template(
    name="zh",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n让我们一步一步地思考。\n",
    templates=industryinstruction_aerospace_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n请扮演一位专业的航空航天领域评审员，评估 AI 助手对下方用户问题所提供回复的质量。\n你的评估应主要考虑以下方面：\n\n1. 正确性：回答是否与参考答案一致，是否存在错误或可能误导的航空信息。  \n2. 完整性：回答是否涵盖参考答案中的关键要点，是否遗漏重要技术细节或规范。  \n3. 相关性：回答是否聚焦于问题和参考答案，避免无关或冗余的内容。  \n4. 专业性：是否使用正确的航空航天专业术语与表达，避免模糊或不专业的说法。  \n5. 表达清晰度：回答是否结构清晰、易于理解，能帮助读者准确掌握航空知识或规程。  \n\n在提供简短的评估说明后，请根据整体表现给出 1 到 10 分的评分，格式为："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[问题开始]\n{question}\n[问题结束]\n\n[参考答案开始]\n{answer}\n[参考答案结束]\n\n[AI 助手回复开始]\n{llm_response}\n[AI 助手回复结束]\n',
    response="{answer}"
)

# Finance
_register_eval_template(
    name="zh-tw",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n讓我們一步一步地思考。\n",
    templates=industryinstruction_finance_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n請扮演一位專業的金融領域評審員，評估 AI 助手對下方使用者問題所提供回覆的品質。\n你的評估應主要考慮以下面向：\n\n1. 正確性：回答是否與標準答案一致，是否存在錯誤或誤導的金融資訊。  \n2. 完整性：回答是否涵蓋標準答案中的核心要點，是否遺漏重要數據、規則或分析。  \n3. 相關性：回答是否緊扣問題與參考答案，避免不必要的延伸或偏離主題。  \n4. 專業性：金融術語與表達是否準確，是否展現嚴謹的專業水準。  \n5. 表達清晰度：回答是否條理分明，讓人能輕易理解財務概念或數據。  \n\n在提供簡短的評估說明後，請根據整體表現給出 1 到 10 分的評分，格式為："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[問題開始]\n{question}\n[問題結束]\n\n[參考答案開始]\n{answer}\n[參考答案結束]\n\n[AI 助手回覆開始]\n{llm_response}\n[AI 助手回覆結束]\n',
    response="{answer}"
)
_register_eval_template(
    name="en",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=industryinstruction_finance_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[Instruction]\nPlease act as a professional finance domain reviewer and evaluate the quality of the AI assistant\'s response to the following user question.\nYour evaluation should mainly consider the following aspects:\n\n1. Accuracy: Does the answer align with the reference answer, and is there any incorrect or misleading financial information?  \n2. Completeness: Does the answer cover the key points of the reference answer, or does it miss important data, rules, or analysis?  \n3. Relevance: Does the answer stay focused on the question and the reference answer, avoiding unnecessary extension or deviation?  \n4. Professionalism: Are financial terms and expressions accurate, demonstrating a rigorous professional standard?  \n5. Clarity of Expression: Is the answer well-structured and easy to understand, helping readers grasp financial concepts or data clearly?  \n\nAfter providing a brief evaluation, please assign a score from 1 to 10 based on the overall performance, in the format: "Rating: [[X]]", for example, "Rating: [[5]]".\n\n[Question Start]\n{question}\n[Question End]\n\n[Reference Answer Start]\n{answer}\n[Reference Answer End]\n\n[AI Assistant Response Start]\n{llm_response}\n[AI Assistant Response End]\n',
    response="{answer}"
)
_register_eval_template(
    name="ko",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n단계별로 생각해 봅시다.\n",
    templates=industryinstruction_finance_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[지시사항]\n전문적인 금융 분야 평가자로서, 아래 사용자 질문에 대한 AI 조수의 답변 품질을 평가해 주십시오.\n평가는 주로 다음 기준을 고려해야 합니다:\n\n1. 정확성: 답변이 기준 답변과 일치하는지, 잘못되거나 오해의 소지가 있는 금융 정보가 있는지.  \n2. 완전성: 기준 답변의 핵심 요점을 모두 포함하는지, 중요한 데이터·규칙·분석을 누락하지 않았는지.  \n3. 관련성: 답변이 질문과 기준 답변에 집중했는지, 불필요한 확장이나 주제 이탈을 피했는지.  \n4. 전문성: 금융 용어와 표현이 정확하고, 엄격한 전문적 수준을 보여주는지.  \n5. 명확성: 답변이 구조적으로 명확하고 이해하기 쉬워 독자가 금융 개념이나 데이터를 쉽게 이해할 수 있는지.  \n\n간단한 평가 설명을 제공한 후, 전체적인 성과에 따라 1에서 10 사이의 점수를 "Rating: [[X]]" 형식으로 제시하십시오. 예: "Rating: [[5]]".\n\n[질문 시작]\n{question}\n[질문 끝]\n\n[기준 답변 시작]\n{answer}\n[기준 답변 끝]\n\n[AI 조수 답변 시작]\n{llm_response}\n[AI 조수 답변 끝]\n',
    response="{answer}"
)
_register_eval_template(
    name="zh",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n让我们一步一步地思考。\n",
    templates=industryinstruction_finance_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n请扮演一位专业的金融领域评审员，评估 AI 助手对下方用户问题所提供回复的质量。\n你的评估应主要考虑以下方面：\n\n1. 正确性：回答是否与参考答案一致，是否存在错误或误导的金融信息。  \n2. 完整性：回答是否涵盖参考答案中的核心要点，是否遗漏重要数据、规则或分析。  \n3. 相关性：回答是否紧扣问题和参考答案，避免不必要的延伸或偏离主题。  \n4. 专业性：金融术语和表达是否准确，是否体现严谨的专业水准。  \n5. 表达清晰度：回答是否条理清楚，是否便于理解财务概念或数据。  \n\n在提供简短的评估说明后，请根据整体表现给出 1 到 10 分的评分，格式为："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[问题开始]\n{question}\n[问题结束]\n\n[参考答案开始]\n{answer}\n[参考答案结束]\n\n[AI 助手回复开始]\n{llm_response}\n[AI 助手回复结束]\n',
    response="{answer}"
)

# Law
_register_eval_template(
    name="zh-tw",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n讓我們一步一步地思考。\n",
    templates=industryinstruction_law_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n請扮演一位專業的法律領域評審員，評估 AI 助手對下方使用者問題所提供回覆的品質。\n你的評估應主要考慮以下面向：\n\n1. 正確性：回答是否與標準答案一致，是否存在法律解釋錯誤或誤導。  \n2. 完整性：回答是否涵蓋參考答案中的核心觀點，是否遺漏關鍵條款或法律要點。  \n3. 相關性：回答是否緊扣問題與參考答案，避免不必要的延伸或不相關的法律內容。  \n4. 專業性：法律用詞是否準確嚴謹，是否符合法律專業表述。  \n5. 表達清晰度：回答是否邏輯嚴謹、層次清晰，便於理解法律依據與推理。  \n\n在提供簡短的評估說明後，請根據整體表現給出 1 到 10 分的評分，格式為："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[問題開始]\n{question}\n[問題結束]\n\n[參考答案開始]\n{answer}\n[參考答案結束]\n\n[AI 助手回覆開始]\n{llm_response}\n[AI 助手回覆結束]\n',
    response="{answer}"
)
_register_eval_template(
    name="en",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=industryinstruction_law_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[Instruction]\nPlease act as a professional law domain reviewer and evaluate the quality of the AI assistant\'s response to the following user question.\nYour evaluation should mainly consider the following aspects:\n\n1. Accuracy: Does the answer align with the reference answer, and are there any legal interpretation errors or misleading points?  \n2. Completeness: Does the answer cover the core viewpoints in the reference answer, or does it miss key provisions or legal arguments?  \n3. Relevance: Does the answer stay focused on the question and the reference answer, avoiding unnecessary extension or unrelated legal content?  \n4. Professionalism: Are legal terms and expressions precise, and do they adhere to the standards of legal writing?  \n5. Clarity of Expression: Is the answer logically sound and well-structured, making it easy to understand the legal basis and reasoning?  \n\nAfter providing a brief evaluation, please assign a score from 1 to 10 based on the overall performance, in the format: "Rating: [[X]]", for example, "Rating: [[5]]".\n\n[Question Start]\n{question}\n[Question End]\n\n[Reference Answer Start]\n{answer}\n[Reference Answer End]\n\n[AI Assistant Response Start]\n{llm_response}\n[AI Assistant Response End]\n',
    response="{answer}"
)
_register_eval_template(
    name="ko",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n단계별로 생각해 봅시다.\n",
    templates=industryinstruction_law_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[지시사항]\n전문적인 법률 분야 평가자로서, 아래 사용자 질문에 대한 AI 조수의 답변 품질을 평가해 주십시오.\n평가는 주로 다음 기준을 고려해야 합니다:\n\n1. 정확성: 답변이 기준 답변과 일치하는지, 법적 해석 오류나 오해의 소지가 있는 부분이 있는지.  \n2. 완전성: 기준 답변의 핵심 관점을 모두 포함하는지, 중요한 조항이나 법률 요점을 누락하지 않았는지.  \n3. 관련성: 답변이 질문과 기준 답변에 집중했는지, 불필요한 확장이나 관련 없는 법률 내용을 피했는지.  \n4. 전문성: 법률 용어와 표현이 정확하고, 법률 전문적 서술 기준을 충족하는지.  \n5. 명확성: 답변이 논리적으로 타당하고 구조가 분명하여 법적 근거와 추론을 이해하기 쉬운지.  \n\n간단한 평가 설명을 제공한 후, 전체적인 성과에 따라 1에서 10 사이의 점수를 "Rating: [[X]]" 형식으로 제시하십시오. 예: "Rating: [[5]]".\n\n[질문 시작]\n{question}\n[질문 끝]\n\n[기준 답변 시작]\n{answer}\n[기준 답변 끝]\n\n[AI 조수 답변 시작]\n{llm_response}\n[AI 조수 답변 끝]\n',
    response="{answer}"
)
_register_eval_template(
    name="zh",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\n让我们一步一步地思考。\n",
    templates=industryinstruction_law_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[指令]\n请扮演一位专业的法律领域评审员，评估 AI 助手对下方用户问题所提供回复的质量。\n你的评估应主要考虑以下方面：\n\n1. 正确性：回答是否与参考答案一致，是否存在法律解释错误或误导。  \n2. 完整性：回答是否涵盖参考答案中的核心观点，是否遗漏关键条款或法律要点。  \n3. 相关性：回答是否紧扣问题与参考答案，避免不必要的延伸或不相关的法律内容。  \n4. 专业性：法律用词是否准确严谨，是否符合法律专业表述。  \n5. 表达清晰度：回答是否逻辑严谨、结构清晰，便于理解法律依据与推理。  \n\n在提供简短的评估说明后，请根据整体表现给出 1 到 10 分的评分，格式为："Rating: [[X]]"，例如："Rating: [[5]]"。\n\n[问题开始]\n{question}\n[问题结束]\n\n[参考答案开始]\n{answer}\n[参考答案结束]\n\n[AI 助手回复开始]\n{llm_response}\n[AI 助手回复结束]\n',
    response="{answer}"
)
