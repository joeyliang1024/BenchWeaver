from typing import Dict, List, Tuple
from ....extras.constants import SCHEMA_TYPES, TEMPLATE_EXAMPLE
from ..template import _register_eval_template
from .opqa_template import OPQA_Template
from ....data.data_utils import Role

class History_Template(OPQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response)
    
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "answer", ...}
        output: a tuple of (prompt, response)
        """
        question = (
            "【任務】根據下列『論文本體』，產出兩個欄位：annotation、overall。\n"
            "【輸出模板（非約束示例）】請仿照下列鍵名與語義，但不要輸出尖括號與說明文字：\n"
            f"{TEMPLATE_EXAMPLE}\n"
            "【型別約束（必須符合）】你的輸出在型別上必須符合以下 JSON Schema：\n"
            f"{SCHEMA_TYPES}\n"
            "【注意事項】\n"
            "1) 僅輸出一個 JSON 物件；不得有額外說明或 Markdown。"
            "2) 必用繁體中文。"
            "3) 請使用文中的 '--- Page N ---' 分頁標記來判斷 annotation.page 的頁碼；若無法定位，填 0。"
            "4) 若資訊不足，對應欄位填空字串 \"\" 或空陣列 []，不得臆測補齊。\n"
            "【論文本體】\n"
            f"{example['paper_text']}"
        ).strip()
        answer = example["ref_comment"].strip()
        return question, answer
    
    def format_checker_example(
        self, target_data: Dict[str, str], llm_response: str, criteria_prompt:str
    ) -> List[Dict[str, str]]:
        if criteria_prompt:
            assert "{answer}" in criteria_prompt and \
                   "{llm_response}" in criteria_prompt \
                ,ValueError("Criteria prompt format incorrect, must contain '{answer}', '{llm_response}'")
            self.criteria_prompt = criteria_prompt
        assert self.criteria_prompt is not None, ValueError("`criteria_prompt` should not be empty.")
        return [
                {
                    "role": Role.USER.value, 
                    "content": self.criteria_prompt.format(
                        answer=target_data['ref_comment'].strip(),
                        llm_response=llm_response,
                        )
                }
            ]

history_eval_templates: Dict[str, "History_Template"] = {}

def get_history_eval_template(name: str) -> "History_Template":
    eval_template = history_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="zh-tw",
    system="",
    choice="",
    answer="",
    cot="",
    templates=history_eval_templates,
    template_class=History_Template,
    criteria_prompt='[指令]\n請扮演一位專業的歷史領域評審員，評估模型對用戶撰寫的小論文之評分的品質與專業教師的評分之相似程度。\n你的評估應主要考慮以下面向：\n1. 題目評語相似度：評估模型對報告題目的評論是否與教師觀點一致\n2. 研究動機評語相似度：比較模型對研究動機部分的批改深度與準確性是否與教師觀點一致\n3. 研究目標評語相似度：檢視模型對研究目標評告的評價合理性是否與教師觀點一致\n4. 研究探討評語相似度：分析模型對主體內容評語的專業性與完整性是否與教師觀點一致\n5. 參考文獻評語相似度：評估模型對文獻引用規範的批改準確度是否與教師觀點一致\n\n每個項目採用 1-10 分的評分量表：\n\n9-10 分：高度相似，專業觀點一致\n7-8 分：相似度良好，主要論點相符\n5-6 分：基本相似，部分觀點有差異\n3-4 分：相似度偏低，需要改進\n1-2 分：差異顯著，需重新訓練\n\n在提供簡短的評估說明後，請根據各項目表現給出 1 到 10 分的評分。\n格式為："Rating: [[X]]"，例如："Rating: [[8]]"\n\n[模型的評語開始]\n{answer}\n[模型的評語結束]\n\n[教師的評語開始]\n{llm_response}\n[教師的評語結束]',
    response=""
)


