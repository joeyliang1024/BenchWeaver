from typing import List, Literal
from torch.serialization import add_safe_globals
from collections import Counter
import evaluate
import jieba
from konlpy.tag import Okt
from comet.models.utils import Prediction
from comet import download_model, load_from_checkpoint
from ...extras.lang_detect import detect_language
from ...extras.constants import COMET_MODEL_NAME_OR_PATH
from .trans_utils import get_encode_fnc, get_valid_fnc

# Add Prediction to safe globals for torch serialization
# This error is due to a recent PyTorch update (v2.6+) 
# where the default behavior of torch.load was changed to weights_only=True, 
# which prevents loading arbitrary Python objects.
add_safe_globals([Prediction]) 

bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
okt = Okt()

def eval_bleu(predictions: List[str], references: List[List[str]])->dict:
    '''
    Evaluates BLEU score for a list of predictions and references.
    '''
    # detect language of 10 
    lang = Counter([detect_language(ref[-1]) for ref in references]).most_common(1)[0][0]
    
    if lang == "en":
        return bleu.compute(predictions=predictions, references=references)
    elif lang in ["zh-cn", "zh-tw"]:
        # use jieba for tokenization
        predictions = [" ".join(jieba.cut(pred)) for pred in predictions]
        references = [[" ".join(jieba.cut(ref)) for ref in refs] for refs in references]
        return bleu.compute(predictions=predictions, references=references)
    elif lang == "ko":
        predictions = [" ".join(okt.morphs(pred)) for pred in predictions]
        references = [[" ".join(okt.morphs(ref)) for ref in refs] for refs in references]
        return bleu.compute(predictions=predictions, references=references)
    else:
        return {
            'bleu': 0.0, 
            'precisions': [], 
            'brevity_penalty': 0.0, 
            'length_ratio': 0.0, 
            'translation_length': 0, 
            'reference_length': 0
            }
    
def eval_chrf(predictions: List[str], references: List[List[str]], word_order:int = 2)->dict:
    '''
    Evaluates CHRF score for a list of predictions and references.
    '''
    return chrf.compute(predictions=predictions, references=references, word_order=word_order)

def eval_comet(predictions: List[str],
               references: List[List[str]],
               comet_model = COMET_MODEL_NAME_OR_PATH,
               batch_size:int = 8,
               gpus:int = 2,
               details:bool = False,
               )->dict:
    '''
    Evaluates COMET score for a list of predictions and references.
    '''
    model_path = download_model(comet_model)
    model = load_from_checkpoint(model_path)
    data = [{"src": ref[0], "mt": pred} for ref, pred in zip(references, predictions)]
    model_output: Prediction = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {"comet": model_output.system_score} if not details else {"comet": model_output.system_score, "scores": model_output.scores}
    
def eval_spbleu(predictions: List[str], 
                references: List[List[str]], 
                output_format: Literal["piece", "id"] = "piece",
                min_len: int = None, 
                max_len: int = None
                ) -> (dict | None):
    '''
    Evaluates SP-BLEU score for a list of predictions and references.
    
    Source code are from [Flories repository](https://github.com/facebookresearch/flores/blob/main/previous_releases/floresv1/scripts/spm_encode.py)
    
    Spm model can be found at [here](https://tinyurl.com/flores200sacrebleuspm)
    
    Args:
        predictions: List of hypothesis translations.
        references: List of reference lists (one or more per prediction).
        output_format: Either 'piece' (subword tokens) or 'id' (token IDs as strings).
        min_len: Minimum length after encoding to include.
        max_len: Maximum length after encoding to include.

    Returns:
        BLEU score dictionary.
        
    '''
    encode = get_encode_fnc(output_format)
    valid = get_valid_fnc(min_len=None, max_len=None)
    
    encode = get_encode_fnc(output_format)
    valid = get_valid_fnc(min_len, max_len)

    encoded_preds, encoded_refs = [], []

    for pred, ref_list in zip(predictions, references):
        enc_pred = encode(pred.strip())
        enc_refs = [encode(ref.strip()) for ref in ref_list]

        # Ensure we only keep valid examples
        if valid(enc_pred) and all(valid(ref) for ref in enc_refs):
            encoded_preds.append(" ".join(enc_pred))
            encoded_refs.append([" ".join(r) for r in enc_refs])

    if not encoded_preds:
        return {
            'bleu': 0.0,
            'precisions': [],
            'brevity_penalty': 0.0,
            'length_ratio': 0.0,
            'translation_length': 0,
            'reference_length': 0
        }

    return bleu.compute(predictions=encoded_preds, references=encoded_refs)
