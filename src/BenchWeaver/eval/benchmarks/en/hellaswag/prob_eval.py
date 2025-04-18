import os
import numpy as np
from tqdm import tqdm, trange
from datasets import load_dataset
from typing import Any, Dict, Optional
from .....extras.constants import MMLU_CHOICES, MMLU_SUBJECTS, PROJECT_BASE_PATH
from ....template import get_hellaswag_eval_template
from ....evaluator import ProbEvaluator

class HellaSwagProbEvaluator(ProbEvaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args=args)
        self.eval_template = get_hellaswag_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in MMLU_CHOICES]
    
    def eval(self) -> None:
        eval_task = self.eval_args.task.split("_")[0]
        eval_split = self.eval_args.task.split("_")[1]
        
        results = {}
        category_corrects = {subj: np.array([], dtype="bool") for subj in MMLU_SUBJECTS}

        pbar = tqdm(self.categories.keys(), desc="Processing subjects", position=0)
        
        for subject in pbar:
            dataset = load_dataset(
                path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, eval_task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.hf_token,
                trust_remote_code=True,
            )
            pbar.set_postfix_str(self.categories[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(len(dataset[eval_split]), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[eval_split][i],
                    choices=MMLU_CHOICES,
                    support_set=support_set,
                    subject_name=self.categories[subject]["name"],
                )

                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])

            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input)
                outputs += preds

            corrects = np.array(outputs) == np.array(labels)
            category_name = self.categories[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        score_info = "\n".join(
            [
                "{:>30}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        self.save_results(category_corrects, results)
