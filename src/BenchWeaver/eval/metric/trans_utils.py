from typing import Literal, Callable, Optional
import sentencepiece as spm
from ...extras.constants import SPM_MODEL_PATH

sp = spm.SentencePieceProcessor()
sp.Load(SPM_MODEL_PATH)

def get_encode_fnc(output_format: Literal["piece", "id"]) -> Callable:
    if output_format == "piece":
        def encode(l):
            return sp.EncodeAsPieces(l)
        return encode
    elif output_format == "id":
        def encode(l):
            return list(map(str, sp.EncodeAsIds(l)))
        return encode
    else:
        raise NotImplementedError
    
    
def get_valid_fnc(min_len: Optional[int], max_len: Optional[int]) -> Callable[[str], bool]:
    if min_len is not None or max_len is not None:
        def valid(line: str) -> bool:
            return (
                (min_len is None or len(line) >= min_len) and
                (max_len is None or len(line) <= max_len)
            )
    else:
        def valid(line: str) -> bool:
            return True
    return valid

