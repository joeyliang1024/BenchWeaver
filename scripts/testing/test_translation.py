import os
os.environ['JAVA_HOME'] = "/usr/lib/java"
from BenchWeaver.eval.metric.translate import eval_bleu, eval_chrf, eval_comet, eval_spbleu  # noqa: E402

# Sample predictions and references
predictions = [
    "The cat sits on the mat.",
    "Dogs are loyal animals.",
    "He likes to play soccer."
]

references = [
    ["The cat is sitting on the mat."],
    ["Dogs are very loyal pets."],
    ["He enjoys playing football."]
]

print("\n====== COMET ======")
comet_result = eval_comet(predictions, references, details=True)
print(comet_result)

print("====== BLEU ======")
bleu_result = eval_bleu(predictions, references)
print(bleu_result)

print("\n====== CHRF ======")
chrf_result = eval_chrf(predictions, references)
print(chrf_result)

print("\n====== SP-BLEU ======")
spbleu_result = eval_spbleu(predictions, references)
print(spbleu_result)

