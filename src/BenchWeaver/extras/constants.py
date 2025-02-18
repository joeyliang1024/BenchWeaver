from pathlib import Path
PROJECT_BASE_PATH = str(Path(__file__).resolve().parents[3])

# MMLU
MMLU_CHOICES = ["A", "B", "C", "D"]
MMLU_SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
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