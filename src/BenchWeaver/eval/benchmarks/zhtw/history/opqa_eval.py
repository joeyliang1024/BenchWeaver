import asyncio
import os
import re
from typing import Any, Dict, List
from json_repair import repair_json
import numpy as np
from tqdm import tqdm
from ....evaluator import OPQAEvaluator
from ....template import get_history_eval_template

class HistoryEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_history_eval_template(self.eval_args.lang)
        self.eval_args.system_prompt = (
            "你是一位使用繁體中文、嚴格但公正的高中小論文批改專家。"
            "只根據使用者提供的論文內容輸出，不得臆測或補寫原文不存在的資訊。"
            "回覆必須且只允許為『一個 JSON 物件』，不得包含任何多餘說明或 Markdown。"
            "本任務產出兩個鍵：annotation（批改清單）與 overall（綜合總評）。"
            "annotation 陣列中 page 必須為整數頁碼，應依據文中的 '--- Page N ---' 標記填入；無法定位時填 0。"
            "【關於 annotation.content 的要求】必須提供『具體可操作』的評語：包含優點、缺點，以及明確的修正建議，"
            "若為五個核心部分（題目、前言、文獻探討、研究方法、參考文獻）之一，請在開頭加入「等第：優等/中等/待加強」，其他一般評論不須給等第。"
            "在最後【綜合總評】中，各部份的重視程度大致可參考以下比例：題目10%, 前言30%, 文獻探討40%, 研究方法10%, 參考文獻10%。"
        )
    
    async def process_subjects(
        self,
        server_process: asyncio.subprocess.Process,
        model_name: str,
        data: Dict[str, List[Any]],
        prompt_key: str,
        output_path: str,
        progress_desc: str,
    ) -> Dict[str, List[Any]]:
        """Process subjects using the specified client and data with concurrency control."""
        results = {subj: [] for subj in self.categories.keys()}
        total_progress_bar = tqdm(self.categories.keys(), desc=progress_desc)

        # Define maximum concurrency
        max_concurrency = getattr(self.model_args, "vllm_max_concurrency", 100)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_item(idx: int, messages: Any, subject: str, progress_bar: tqdm):
            """Processes a single item with semaphore-based concurrency control."""
            async with semaphore:
                try:
                    result, origin_idx = await self.generate(
                        model=model_name,
                        system_prompt=getattr(self.eval_args, prompt_key),
                        example=messages,
                        idx=idx,
                        generating_args=self.generating_args,
                    )
                    progress_bar.update(1)
                    return origin_idx, result
                except Exception as e:
                    progress_bar.update(1)
                    print(f"Error processing item {idx} in subject {subject}: {e}")
                    return idx, None

        try:
            for subject in self.categories.keys():
                subject_results = [None] * len(data[subject])

                with tqdm(
                    total=len(data[subject]),
                    desc=self.categories[subject]["name"],
                    dynamic_ncols=True,
                ) as subject_progress_bar:

                    # Create tasks for all items under a subject
                    tasks = [
                        asyncio.create_task(process_single_item(idx, messages, subject, subject_progress_bar))
                        for idx, messages in enumerate(data[subject])
                    ]

                    # Collect results as tasks complete
                    for task in asyncio.as_completed(tasks):
                        origin_idx, result = await task
                        if result is not None:
                            subject_results[origin_idx] = self.retrieve_response(result)

                results[subject] = subject_results
                total_progress_bar.update(1)

        finally:
            # Ensure cleanup and save results
            await self.terminate_server(process=server_process)
            self.client = None
            self.save_data(data=results, output_path=os.path.join(self.save_folder, output_path))
            total_progress_bar.close()

        return results
    
    def comput_score(self, check_results: Dict[str, List[Any]], subjects: List[str], checked_answers=None) -> Dict[str, float]:
        criteria_keys = ['題目評語相似度', '研究動機評語相似度', '研究目標評語相似度', '研究探討評語相似度', '參考文獻評語相似度']
        category_corrects = {
            subj: {
                key: [] for key in criteria_keys
            } 
            for subj in subjects
        }
        category_corrects["Average"] = []
        for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
            category_name = self.categories[subject]["category"]
            for check_result in check_results[subject]:
                retrieved_results = self.retrieve_scores(check_result, keys=criteria_keys)
                for key in criteria_keys:
                    category_corrects[category_name][key].append(retrieved_results.get(key, 0))
                category_corrects["Average"].append(retrieved_results.get('weighted_average', 0))
        # compute average score for each criteria
        final_scores = {}
        for category_name, scores_dict in category_corrects.items():
            if category_name != "Average":
                final_scores[category_name] = {
                    key: round(np.mean(scores), 4) if scores else 0.0
                    for key, scores in scores_dict.items()
                }
            else:
                final_scores[category_name] = round(np.mean(scores_dict), 4) if scores_dict else 0.0
        return final_scores
        
    @staticmethod
    def retrieve_scores(parse_string: str, keys: List[str]) -> Dict[str, float]:
        """
        從文本中提取所有的 rating 分數並組織成字典格式
        使用正則表達式直接匹配評估項目和對應的rating
        """
        # 定義各個評估項目的關鍵字映射
        rating_patterns = {
            '題目評語相似度': r'題目評語相似度.*?Rating:\s*\[\[(\d+)\]\]',
            '研究動機評語相似度': r'研究動機評語相似度.*?Rating:\s*\[\[(\d+)\]\]', 
            '研究目標評語相似度': r'研究目標評語相似度.*?Rating:\s*\[\[(\d+)\]\]',
            '研究探討評語相似度': r'研究探討評語相似度.*?Rating:\s*\[\[(\d+)\]\]',
            '參考文獻評語相似度': r'參考文獻評語相似度.*?Rating:\s*\[\[(\d+)\]\]'
        }
        # 初始化結果字典
        result = {
            key: 0 for key in keys
        }
        weights = [0.2 , 0.2, 0.2, 0.25, 0.15]  # 五個評分項目的權重相等
        # 使用 re.DOTALL 讓 . 也能匹配換行符
        for key, pattern in rating_patterns.items():
            match = re.search(pattern, parse_string, re.DOTALL)
            if match:
                result[key] = int(match.group(1))
        # compute weighted average
        result['weighted_average'] = sum(result[key] * weight for key, weight in zip(keys, weights))
        return result
        
    @staticmethod
    def retrieve_response(parse_string: str) -> str:
        result = repair_json(parse_string)
        if isinstance(result, dict) and "overall" in result:
                return result["overall"]
        return parse_string