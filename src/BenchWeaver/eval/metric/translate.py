from typing import List
from collections import Counter
import evaluate
import jieba
from konlpy.tag import Okt
from comet.models.utils import Prediction
from comet import load_from_checkpoint
from ...extras.lang_detect import detect_language
from ...extras.constants import COMET_MODEL_NAME_OR_PATH

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
    data = [{"src": ref[0], "mt": pred} for ref, pred in zip(references, predictions)]
    model = load_from_checkpoint(comet_model)
    model_output: Prediction = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {"comet": model_output.system_score} if not details else {"comet": model_output.system_score, "scores": model_output.scores}
    
    