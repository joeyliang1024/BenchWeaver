from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

taide_bench_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_taide_bench_eval_template(name: str) -> "OPQA_Template":
    eval_template = taide_bench_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en2zh",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=taide_bench_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[評估指南]\n\n請針對Assistant的翻譯進行評估。考慮到翻譯是給台灣的讀者看，請特別留意以下細項：\n\n語法正確性：檢查翻譯的句子是否在語法上正確無誤。\n\n用詞選擇：檢查使用的詞彙是否正確且適當，並符合台灣中文的習慣。\n\n保留原文意思：翻譯是否忠實於原文，並保留其主要意思和細節。\n\n文化和語境適應性：檢查翻譯是否考慮到台灣的文化和語境，特別是當原文中有可能產生誤解或與台灣文化有出入的內容。\n\n使用目標語言：檢查是否完全使用了目標語言（繁體中文），並避免了不必要的原文語言內容。\n\n[問題開始]\n{question}\n\n[問題結束]\n[標準答案開始]\n\n{answer}\n\n[標準答案結束]\n\n[Assistant 回答開始]\n{llm_response}\n\n[Assistant 回答結束]\n\n在給出分數時，請按照以下指引：\n\n1-3分：大部分的翻譯都存在問題。\n4-6分：翻譯中存在一些明顯的錯誤或遺漏。\n7-8分：翻譯大致上是正確的，但還有一些小問題。\n9-10分：翻譯非常精確，幾乎沒有任何問題。\n請於最後一行輸出 Assistant 的分數，並且格式為 Score:分數',
    response="The correct answer is ({answer})."
)

_register_eval_template(
    name="zh2en",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=taide_bench_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[評估指南]\n\n請針對Assistant的翻譯進行評估。考慮到翻譯是給台灣的讀者看，請特別留意以下細項：\n\n語法正確性：檢查翻譯的句子是否在語法上正確無誤。\n\n用詞選擇：檢查使用的詞彙是否正確且適當，並符合台灣中文的習慣。\n\n保留原文意思：翻譯是否忠實於原文，並保留其主要意思和細節。\n\n文化和語境適應性：檢查翻譯是否考慮到台灣的文化和語境，特別是當原文中有可能產生誤解或與台灣文化有出入的內容。\n\n使用目標語言：檢查是否完全使用了目標語言（英文），並避免了不必要的原文語言內容。\n\n[問題開始]\n{question}\n\n[問題結束]\n[標準答案開始]\n\n{answer}\n\n[標準答案結束]\n\n[Assistant 回答開始]\n{llm_response}\n\n[Assistant 回答結束]\n\n在給出分數時，請按照以下指引：\n\n1-3分：大部分的翻譯都存在問題。\n4-6分：翻譯中存在一些明顯的錯誤或遺漏。\n7-8分：翻譯大致上是正確的，但還有一些小問題。\n9-10分：翻譯非常精確，幾乎沒有任何問題。\n請於最後一行輸出 Assistant 的分數，並且格式為 Score:分數',
    response="The correct answer is ({answer})."
)

_register_eval_template(
    name="essay",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=taide_bench_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[問題]\n{question}\n\n[標準回答開始]\n{answer}\n\n[標準回答結束]\n\n[Assistant 回答開始]\n{llm_response}\n\n[Assistant 回答結束]\n\n[system]\n我們希望你針對上述問題和標準回答，對 Assistant 的表現提供回應。\n請對其回答的幫助性、相關性、準確性、詳細程度以及是否使用中文進行評分。不必要的英文內容應該避免，並會受到扣分。即使在中翻英任務中，系統說明例如「這是將中文句子翻譯成英文的結果」之敘述也應以繁體中文提供。Assistant 將在 1 到 10 的範圍上獲得總體分數，較高的分數表示總體表現較好。\n請首先針對你的評估提供全面的解釋，避免任何潛在偏見，並確保回答呈現的順序不影響你的判斷。請於最後一行輸出 Assistant 的分數，並且格式為 Score:分數',
    response="The correct answer is ({answer})."
)

_register_eval_template(
    name="letter",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=taide_bench_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[問題]\n{question}\n\n[標準回答開始]\n{answer}\n\n[標準回答結束]\n\n[Assistant 回答開始]\n{llm_response}\n\n[Assistant 回答結束]\n\n[system]\n我們希望你針對上述問題和標準回答，對 Assistant 的表現提供回應。\n請對其回答的幫助性、相關性、準確性、詳細程度以及是否使用中文進行評分。不必要的英文內容應該避免，並會受到扣分。即使在中翻英任務中，系統說明例如「這是將中文句子翻譯成英文的結果」之敘述也應以繁體中文提供。Assistant 將在 1 到 10 的範圍上獲得總體分數，較高的分數表示總體表現較好。\n請首先針對你的評估提供全面的解釋，避免任何潛在偏見，並確保回答呈現的順序不影響你的判斷。請於最後一行輸出 Assistant 的分數，並且格式為 Score:分數',
    response="The correct answer is ({answer})."
)

_register_eval_template(
    name="summary",
    system="",
    choice="\n{choice}. {content}",
    answer="",
    cot="\nLet's think step by step.\n",
    templates=taide_bench_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt='[評估指南]\n\n請針對Assistant的摘要進行評估。請特別留意以下細項：\n\n簡單明瞭：檢查是否簡單明瞭的保留原始文章大致內容，避免陷入不重要的細節。\n\n用詞選擇：檢查使用的詞彙是否符合台灣所使用的繁體中文習慣，且該使用原文時保留原始語言。\n\n[問題開始]\n{question}\n\n[問題結束]\n[標準答案開始]\n{answer}\n\n[標準答案結束]\n\n[Assistant 回答開始]\n{llm_response}\n\n[Assistant 回答結束]\n\n在給出分數時，請按照以下指引：\n\n1-3分：摘要明顯有誤或回答原文。\n4-6分：摘要存在一些明顯的錯誤或遺漏。\n7-8分：摘要大致上是正確的，但還有一些小問題。\n9-10分：摘要非常精確，幾乎沒有任何問題。\n請於最後一行輸出 Assistant 的分數，並且格式為 Score:分數',
    response="The correct answer is ({answer})."
)





