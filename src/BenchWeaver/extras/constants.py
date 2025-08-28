import os
from pathlib import Path
PROJECT_BASE_PATH = str(Path(__file__).resolve().parents[3])
TRANSLATION_PROMPT_PATH = os.path.join(PROJECT_BASE_PATH, "prompt", "translation_prompt.json")

# COMET model
COMET_MODEL_NAME_OR_PATH = "Unbabel/wmt20-comet-qe-da"
# Word Tokenizer
ENABLE_INIT_TOKENIZER = False
# MMLU
MMLU_CHOICES = ["A", "B", "C", "D"]
MMLU_SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
KMMLU_SUBJECTS = ["Average", 'STEM', 'Applied Science', 'HUMSS', 'Other']
TMLU_SUBJECTS = ["Average", "Social Science", "STEM", "Humanities", "Taiwan Specific", "Others"]
MMLU_IDX2OPT =  {0:"A", 1:"B", 2:"C", 3:"D"}
# GSM8K
GSM8K_SUBJECTS = ["Average", "main", "socratic"]
# ARC Challenge
ARC_CHALLENGE_CHOICES = ["A", "B", "C", "D"]
ARC_CHALLENGE_SUBJECTS = ["Average", "challenge"]
# GPQA
GPQA_SUBJECTS = ["Average", "diamond", "extended", "main"]
# TruthfulQA
TRUTHFULQA_SCORES = ["Average", "generation", "mcqa-mc1", "mcqa-mc2"]
# Big Bench Hard
BIG_BENCH_HARD_SUBJECTS = ["Average", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "object_counting", "penguins_in_a_table", "salient_translation_error_detection", "tracking_shuffled_objects_five_objects",]
# option codes
OPTION_CODES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AI", "AK", "AL", "AM", "AN", "AP", "AR", "AS", "AT", "AU", "AV", "AW", "AX", "BA", "BB", "BC", "BD", "BE", "BF", "BI", "BL", "BO", "BR", "BS", "BU", "BY", "CA", "CB", "CC", "CD", "CE", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO", "CP", "CR", "CS", "CT", "CV", "DA", "DB", "DC", "DD", "DE", "DF", "DI", "DK", "DL", "DM", "DN", "DO", "DP", "DR", "DS", "DT", "DU", "EB", "EC", "ED", "EE", "EF", "EG", "EL", "EM", "EN", "EP", "EQ", "ER", "ES", "ET", "EV", "EX", "EY", "FA", "FB", "FC", "FD", "FE", "FF", "FI", "FL", "FM", "FO", "FP", "FR", "FS", "FT", "FX", "GA", "GB", "GC", "GE", "GG", "GL", "GM", "GN", "GO", "GP", "GR", "GS", "GT", "GV", "HA", "HC", "HD", "HE", "HH", "HI", "HL", "HO", "HP", "HR", "HS", "HT", "IA", "IB", "IC", "ID", "IE", "IF", "IG", "II", "IK", "IL", "IM", "IN", "IO", "IP", "IR", "IS", "IT", "IV", "IX", "IZ", "JB", "JS", "KB", "KE", "KS", "LA", "LC", "LD", "LE", "LI", "LL", "LM", "LO", "LP", "LR", "LS", "LT", "LY", "MA", "MB", "MC", "MD", "ME", "MI", "ML", "MM", "MO", "MP", "MQ", "MR", "MS", "MT", "MW", "MY", "NA", "NB", "NC", "ND", "NE", "NF", "NG", "NI", "NL", "NN", "NO", "NP", "NR", "NS", "NT", "NU", "OB", "OC", "OD", "OF", "OH", "OK", "OL", "OM", "ON", "OP", "OR", "OS", "OT", "OU", "OW", "PA", "PC", "PD", "PE", "PF", "PG", "PH", "PI", "PK", "PL", "PM", "PN", "PO", "PP", "PR", "PS", "PT", "PU", "PY", "QL", "QU", "RA", "RC", "RE", "RI", "RL", "RO", "RS", "RT", "RU", "RY", "SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH", "SI", "SK", "SL", "SM", "SN", "SO", "SP", "SR", "SS", "ST", "SU", "SV", "SW", "SY", "TA", "TB", "TC", "TD", "TE", "TF", "TH", "TI", "TL", "TM", "TO", "TP", "TR", "TS", "TT", "TV", "TW", "TX", "TY", "UB", "UC", "UD", "UE", "UG", "UI", "UK", "UL", "UM", "UN", "UP", "UR", "US", "UT", "VA", "VB", "VC", "VD", "VE", "VF", "VI", "VM", "VO", "VP", "VS", "WA", "WD", "WE", "WF", "WH", "WI", "WM", "WN", "WP", "WR", "WS", "WT", "XT", "XV", "XX", "XY", "YS", "YY", "ZE", "ABC", "ACE", "ACK", "ACT", "ADD", "AGE", "ALL", "AME", "AML", "AMP", "AND", "ANG", "ANT", "API", "APP", "ARD", "ARN", "ART", "ARY", "ASC", "ASE", "ASH", "ASS", "AST", "ATA", "ATE", "ATH", "AUT", "AVA", "AXI", "BER", "BIT", "BUG", "CAA", "CAT", "CCE", "CCN", "CES", "CLA", "CLC", "CLI", "COL", "COM", "CON", "CRE", "CSS", "CUR", "DAT", "DAY", "DBC", "DEF", "DER", "DEX", "DIR", "DIS", "DOC", "DOM", "EAR", "ECK", "ECT", "ELD", "EMA", "END", "ENT", "ENV", "ERE", "ERR", "ERS", "ERT", "ERY", "EXT", "FIG", "FIX", "FLA", "FOR", "GEN", "GER", "GET", "HER", "IAB", "IAL", "ICE", "IDE", "IES", "IGN", "III", "ILL", "IMA", "IME", "IND", "INE", "INF", "ING", "INT", "ION", "IOS", "ISO", "IST", "ITE", "ITH", "ITY", "IVE", "JAX", "KEY", "LAB", "LAY", "LED", "LES", "LIC", "LIN", "LOB", "LOC", "LOG", "LOW", "MAN", "MAP", "MAX", "MIN", "MIT", "MON", "NER", "NET", "NEW", "NOT", "NUM", "OFF", "OIN", "ONE", "ONG", "OPT", "ORD", "ORM", "ORS", "ORT", "ORY", "OST", "OUR", "OUT", "PAR", "PDF", "PER", "PHP", "POS", "PRE", "PRI", "PRO", "PUT", "QUE", "RAM", "RAY", "RED", "REE", "REF", "REG", "RES", "RGB", "RIG", "ROM", "ROP", "ROR", "ROW", "SBN", "SDK", "SEE", "SER", "SET", "SHA", "SON", "SQL", "SSL", "SSN", "STR", "SUB", "SUM", "TAC", "TAG", "TER", "THE", "UES", "UID", "ULL", "ULT", "UMN", "UND", "UNT", "URE", "URI", "URL", "URN", "USA", "USE", "UST", "UTC", "UTE", "UTF", "VAL", "VAR", "VER", "VID", "VIS", "WID", "WIN", "WOR", "XML", "XXX", "YES", "YPE"]

# SPM Model Path
SPM_MODEL_PATH = os.path.join(PROJECT_BASE_PATH, "model", "flores_spm", "flores200_sacrebleu_tokenizer_spm.model")

# GPT NOT SUPPORT PARM MODEL
GPT_NOT_SUPPORT_PARM_MODELS = [
    "gpt-o3-mini"
]

CRITERIA_PROMPT = '''
你是一位專業的翻譯評估員。請根據提供的「原文文本」、「翻譯文本」、「風格範例」及「評估標準」，對翻譯品質進行評估，評分範圍為 1（最差）至 10（最佳），並以 JSON 格式輸出結果。

評估標準：
1. 資訊保留度：
   - 評估翻譯文本是否完整保留了原文的資訊內容，包括關鍵細節、邏輯關係與語義準確性。
   - 允許因風格匹配的需求進行詞句調整，但不可影響核心資訊的傳遞。
   - 例如，若原文提及具體數據、時間、因果關係或條件，翻譯文本應忠實呈現，而非省略或改動這些重要內容。

2. 風格匹配度：
   - 若風格範例為空，則請直接給予 10 分。
   - 評估翻譯文本是否符合給定的「風格範例」，包括語氣、句式、措辭選擇、正式度等。
   - 例如，若風格範例是學術論文，則翻譯文本應使用正式、嚴謹的語言，避免口語化表達；若風格範例是兒童讀物，則應使用簡單易懂、富有親和力的詞彙。
   - 風格匹配度高的翻譯應該讀起來與範例文本的風格一致，而不只是逐字翻譯。
   - 若「原文文本」只回答選擇題的答案 (A, B, C, D)，沒有其他輸出，則不需要考慮風格匹配度，給予 10 分，若有額外輸出解釋，則需考慮風格匹配度。

3. 專有名詞準確度：
   - 專有名詞包括人名、地名、機構名稱、術語、技術詞彙等，應與上下文一致，並符合標準翻譯慣例。
   - 例如，「United Nations」應翻譯為「聯合國」，而非「統一國家」；「Neural Network」應譯為「神經網絡」，而非「神經連接」。
   - 若專有名詞有公認的譯法，則應使用標準譯法，若無固定譯法，則應確保譯法在全文內保持一致。

4. 翻譯品質：
   - 綜合評估翻譯文本的整體品質，包括語法、流暢度與可讀性。
   - 翻譯應避免生硬直譯或機翻痕跡，確保句子通順自然、符合目標語言的語法規範。
   - 例如，若翻譯文本讀起來拗口或不符合語法，應降低分數；若譯文自然流暢，則應提高分數。

---
「原文文本」：
{source_text}

「翻譯文本」：
{target_text}

「風格範例」：
{style_example}
---

請以以下 JSON 格式輸出評估結果，確保 `分數` 為 1-10 之間的整數，`原因` 為簡要但具體的說明：
{
    "資訊保留度": {
        "分數": <1-10 的分數>,
        "原因": "<簡要說明此評分的理由>"
    },
    "風格匹配度": {
        "分數": <1-10 的分數>,
        "原因": "<簡要說明此評分的理由>"
    },
    "專有名詞準確度": {
        "分數": <1-10 的分數>,
        "原因": "<簡要說明此評分的理由>"
    },
    "翻譯品質": {
        "分數": <1-10 的分數>,
        "原因": "<簡要說明此評分的理由>"
    }
}
'''.strip()

# bleow are need for llama-factory code
from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

RUNNING_LOG = "running_log.txt"
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

MOD_SUPPORTED_MODELS = {"bloom", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "starcoder2"}

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"

V_HEAD_WEIGHTS_NAME = "value_head.bin"

LAYERNORM_NAMES = {"norm", "ln"}

SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}

SUPPORTED_CLASS_FOR_BLOCK_DIAG_ATTN = {
    "cohere",
    "falcon",
    "gemma",
    "gemma2",
    "llama",
    "mistral",
    "phi",
    "phi3",
    "qwen2",
    "starcoder2",
}