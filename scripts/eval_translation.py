from argparse import Namespace
import asyncio
import os
import re
import json
import fire
import logging
import ast
import numpy as np
from tqdm.auto import tqdm
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from BenchWeaver.extras.constants import PROJECT_BASE_PATH, CRITERIA_PROMPT
from BenchWeaver.inference.client import Client

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__) 

def export_data(data: Union[list, dict], output_path:str) -> None:
    """
    Export the data to json format.
    """
    assert isinstance(data, (list, dict)), "Data must be a list or a dictionary."
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Data exported to {output_path}")
    
def load_data(input_path: str) -> Union[list, dict]:
    """
    Load data from json format.
    """
    assert os.path.exists(input_path), f"Input directory {input_path} does not exist."
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Data loaded from {input_path}")
    return data

def load_translation_data(origin_lang_dir, target_lang_dir, data_type: Literal['question', 'answer'], test_mode: bool = False) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """
    Load translation data from the specified directories.
    """
    def maybe_truncate(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        if test_mode:
            return {subj: items[:3] for subj, items in data.items()}
        return data

    if data_type == 'question':
        # file paths for question data
        origin_lang_prompt_file = os.path.join(origin_lang_dir, 'inference_prompts.json')
        target_lang_prompt_file = os.path.join(target_lang_dir, 'translated_question_record.json')
        origin_question_dict = {
            subj: [
                    ''.join(
                        f"- user: {item['content']}\n" if item['role'] == 'user' else f"- assistant: {item['content']}\n"
                        for item in message
                    )
                    for message in message_lists
                ][:min(len(message_lists), 200)] # take at most 200 responses
            for subj, message_lists in load_data(origin_lang_prompt_file).items() 
        }      
        trans_question_dict = {
            subj: [
                    ''.join(
                        f"- user: {item['content']}\n" if item['role'] == 'user' else f"- assistant: {item['content']}\n"
                        for item in message
                    )
                    for message in message_lists
                ][:min(len(message_lists), 200)] # take at most 200 responses
            for subj, message_lists in load_data(target_lang_prompt_file).items() 
        }
        return maybe_truncate(origin_question_dict), maybe_truncate(trans_question_dict)
    
    elif data_type == 'answer':
        # file paths for answer data
        origin_lang_answer_file = os.path.join(origin_lang_dir, 'translated_response_record.json')
        target_lang_answer_file = os.path.join(target_lang_dir, 'inference_results.json')
        origin_answer_dict = {
            subj: [
                response for response in response_list
                ][:min(len(response_list), 200)] # take at most 200 responses
            for subj, response_list in load_data(origin_lang_answer_file).items()        
        }
        trans_answer_dict = {
            subj: [
                response for response in response_list
                ][:min(len(response_list), 200)] # take at most 200 responses
            for subj, response_list in load_data(target_lang_answer_file).items()        
        }
        return maybe_truncate(origin_answer_dict), maybe_truncate(trans_answer_dict)
    
    else :
        raise ValueError("data_type must be either 'question' or 'answer'.")

def clean_inner_keys(data: dict) -> dict:
    """
    Cleans inner dictionary keys by removing colons (： and :) and stripping whitespace.

    Args:
        data (dict): A dictionary with nested dictionaries as values.

    Returns:
        dict: A new dictionary with cleaned inner keys.
    """
    cleaned_data = {}
    for outer_key, inner_dict in data.items():
        cleaned_inner = {}
        for k, v in inner_dict.items():
            cleaned_key = re.sub(r'[：:]', '', k).strip()
            cleaned_inner[cleaned_key] = v
        cleaned_data[outer_key] = cleaned_inner
    return cleaned_data

def parse_score(llm_response: str) -> dict:
    """
    Parse the score from the LLM response.
    """
    try:
        score_dict = ast.literal_eval(re.sub(r'\\|\n', '', llm_response))
        if isinstance(score_dict, dict):
            return clean_inner_keys(score_dict)
        else:
            return score_dict
    
    except (SyntaxError, ValueError):
        
        return {
                  "資訊保留度": {"分數": 1, "原因": ""},
                  "風格匹配度": {"分數": 1, "原因": ""},
                  "專有名詞準確度": {"分數": 1, "原因": ""},
                  "翻譯品質": {"分數": 1, "原因": ""}
        }
        
def retrieve_style_example(trans_prompt:str) -> str:
    pattern = r"(?:For example:|Examples:|Few-shot Examples:)\s*(.*?)\s*(?:Source sentence:|Proper Noun Examples:)"
    match = re.search(pattern, trans_prompt, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        return ""

def format_prompt(source_data_dict: Dict[str, List[Any]], target_data_dict: Dict[str, List[Any]], style_example: str) -> Dict[str, List[Any]]:
    """
    Format the prompt for the LLM.
    return a dict with key subject and value as a list of of messages (list of dict with role and content)
    """
    prompt_dict = {subj: [] for subj in source_data_dict.keys()}
    # sort the source and target data dicts by subject
    with tqdm(total=len(source_data_dict), desc="Formatting prompts") as pbar:
        for subj in source_data_dict.keys():
            pbar.set_postfix_str(subj)
            for source, target in zip(source_data_dict[subj], target_data_dict[subj]):
                prompt_dict[subj].append([
                    {
                        "role": "user",
                        "content": CRITERIA_PROMPT.replace(
                            "{source_text}", source).replace(
                                "{target_text}", target).replace(
                                    "{style_example}", style_example
                                    ).strip()
                    }
                ])
            pbar.update(1)
    pbar.close()
    return prompt_dict

def merge_and_calculate_results(
    question_check_result: Dict[str, List[dict]],
    answer_check_result: Dict[str, List[dict]]
    ):
    score_dict = {
        subj: {
            "question":{
                "資訊保留度": 0,
                "風格匹配度": 0,
                "專有名詞準確度": 0,
                "翻譯品質": 0,
                "Average": 0
            },
            "answer":{
                "資訊保留度": 0,
                "風格匹配度": 0,
                "專有名詞準確度": 0,
                "翻譯品質": 0,
                "Average": 0
            }
        } 
        for subj in question_check_result.keys()
    }
    average_score_dict = {
        "資訊保留度": 0,
        "風格匹配度": 0,
        "專有名詞準確度": 0,
        "翻譯品質": 0,
        "Average": 0,
        "question":{
                "資訊保留度": 0,
                "風格匹配度": 0,
                "專有名詞準確度": 0,
                "翻譯品質": 0,
                "Average": 0
            },
            "answer":{
                "資訊保留度": 0,
                "風格匹配度": 0,
                "專有名詞準確度": 0,
                "翻譯品質": 0,
                "Average": 0
            }
    }
    
    for subj in question_check_result.keys():
        question_record_dict = {
            "資訊保留度": [],
            "風格匹配度": [],
            "專有名詞準確度": [],
            "翻譯品質": []
        }
        answer_record_dict = {
            "資訊保留度": [],
            "風格匹配度": [],
            "專有名詞準確度": [],
            "翻譯品質": []
        }
        for question_result_dict, answer_result_dict in zip(question_check_result[subj], answer_check_result[subj]):
            # append the scores to the record dict
            try:
                question_record_dict["資訊保留度"].append(question_result_dict["資訊保留度"]["分數"])
                question_record_dict["風格匹配度"].append(question_result_dict["風格匹配度"]["分數"])
                question_record_dict["專有名詞準確度"].append(question_result_dict["專有名詞準確度"]["分數"])
                question_record_dict["翻譯品質"].append(question_result_dict["翻譯品質"]["分數"])
                answer_record_dict["資訊保留度"].append(answer_result_dict["資訊保留度"]["分數"])
                answer_record_dict["風格匹配度"].append(answer_result_dict["風格匹配度"]["分數"])
                answer_record_dict["專有名詞準確度"].append(answer_result_dict["專有名詞準確度"]["分數"])
                answer_record_dict["翻譯品質"].append(answer_result_dict["翻譯品質"]["分數"])
                # calculate the average score for each subject
                score_dict[subj]['question']['資訊保留度'] = np.mean(question_record_dict["資訊保留度"])
                score_dict[subj]['question']['風格匹配度'] = np.mean(question_record_dict["風格匹配度"])
                score_dict[subj]['question']['專有名詞準確度'] = np.mean(question_record_dict["專有名詞準確度"])
                score_dict[subj]['question']['翻譯品質'] = np.mean(question_record_dict["翻譯品質"])
                score_dict[subj]['question']['Average'] = np.mean([
                    score_dict[subj]['question']['資訊保留度'],
                    score_dict[subj]['question']['風格匹配度'],
                    score_dict[subj]['question']['專有名詞準確度'],
                    score_dict[subj]['question']['翻譯品質']
                ])
                score_dict[subj]['answer']['資訊保留度'] = np.mean(answer_record_dict["資訊保留度"])
                score_dict[subj]['answer']['風格匹配度'] = np.mean(answer_record_dict["風格匹配度"])
                score_dict[subj]['answer']['專有名詞準確度'] = np.mean(answer_record_dict["專有名詞準確度"])
                score_dict[subj]['answer']['翻譯品質'] = np.mean(answer_record_dict["翻譯品質"])
                score_dict[subj]['answer']['Average'] = np.mean([
                    score_dict[subj]['answer']['資訊保留度'],
                    score_dict[subj]['answer']['風格匹配度'],
                    score_dict[subj]['answer']['專有名詞準確度'],
                    score_dict[subj]['answer']['翻譯品質']
                ])
            except KeyError as e:
                logger.error(f"KeyError: {e} in subject {subj}")
                logger.error(f"Question result: {question_result_dict}")
                logger.error(f"Answer result: {answer_result_dict}")
                continue
            
    # calculate average score for each subject
    average_score_dict['資訊保留度'] = np.mean(
        [score_dict[subj]['question']['資訊保留度'] for subj in score_dict.keys()] + 
        [score_dict[subj]['answer']['資訊保留度'] for subj in score_dict.keys()]
    )
    average_score_dict['question']["資訊保留度"] = np.mean(
        [score_dict[subj]['question']['資訊保留度'] for subj in score_dict.keys()]
    )
    average_score_dict['answer']["資訊保留度"] = np.mean(
        [score_dict[subj]['answer']['資訊保留度'] for subj in score_dict.keys()]
    )
    average_score_dict['風格匹配度'] = np.mean(
        [score_dict[subj]['question']['風格匹配度'] for subj in score_dict.keys()] + 
        [score_dict[subj]['answer']['風格匹配度'] for subj in score_dict.keys()]
    )
    average_score_dict['question']["風格匹配度"] = np.mean(
        [score_dict[subj]['question']['風格匹配度'] for subj in score_dict.keys()]
    )
    average_score_dict['answer']["風格匹配度"] = np.mean(
        [score_dict[subj]['answer']['風格匹配度'] for subj in score_dict.keys()]
    )
    average_score_dict['專有名詞準確度'] = np.mean(
        [score_dict[subj]['question']['專有名詞準確度'] for subj in score_dict.keys()] + 
        [score_dict[subj]['answer']['專有名詞準確度'] for subj in score_dict.keys()]
    )
    average_score_dict['question']["專有名詞準確度"] = np.mean(
        [score_dict[subj]['question']['專有名詞準確度'] for subj in score_dict.keys()]
    )
    average_score_dict['answer']["專有名詞準確度"] = np.mean(
        [score_dict[subj]['answer']['專有名詞準確度'] for subj in score_dict.keys()]
    )
    average_score_dict['翻譯品質'] = np.mean(
        [score_dict[subj]['question']['翻譯品質'] for subj in score_dict.keys()] + 
        [score_dict[subj]['answer']['翻譯品質'] for subj in score_dict.keys()]
    )
    average_score_dict['question']["翻譯品質"] = np.mean(
        [score_dict[subj]['question']['翻譯品質'] for subj in score_dict.keys()]
    )
    average_score_dict['answer']["翻譯品質"] = np.mean(
        [score_dict[subj]['answer']['翻譯品質'] for subj in score_dict.keys()]
    )
    average_score_dict['Average'] = np.mean([
        average_score_dict['資訊保留度'],
        average_score_dict['風格匹配度'],
        average_score_dict['專有名詞準確度'],
        average_score_dict['翻譯品質']
    ])
    average_score_dict['question']["Average"] = np.mean([
        average_score_dict['question']["資訊保留度"],
        average_score_dict['question']["風格匹配度"],
        average_score_dict['question']["專有名詞準確度"],
        average_score_dict['question']["翻譯品質"]
    ])
    average_score_dict['answer']["Average"] = np.mean([
        average_score_dict['answer']["資訊保留度"],
        average_score_dict['answer']["風格匹配度"],
        average_score_dict['answer']["專有名詞準確度"],
        average_score_dict['answer']["翻譯品質"]
    ])
    
    score_dict.update({"Average": average_score_dict})
    return score_dict

async def generate(
    client: Client,
    model: str,
    system_prompt: Optional[str],
    example: List[Dict[str, Any]],
    idx: int, 
    generating_args: Namespace,
) -> Tuple[str, int]:
    return await client.generate(
        model=model,
        system_prompt=system_prompt,
        example=example,
        generating_args=generating_args,
    ), idx
        
async def process_subjects(
    client: Client,
    model_name: str,
    data: Dict[str, List[Any]],
    output_path: str,
    progress_desc: str,
    max_concurrency: int = 100,
) -> Dict[str, List[Any]]:
    """Process subjects using the specified client and data with concurrency control."""
    results = {subj: [] for subj in data.keys()}
    total_progress_bar = tqdm(data.keys(), desc=progress_desc)
    # Define maximum concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    async def process_single_item(idx: int, messages: Any, subject: str, progress_bar: tqdm):
        """Processes a single item with semaphore-based concurrency control."""
        async with semaphore:
            try:
                result, origin_idx = await generate(
                    client=client,
                    model=model_name,
                    system_prompt=None,
                    example=messages,
                    idx=idx,
                    generating_args=None,
                )
                progress_bar.update(1)
                return origin_idx, result
            except Exception as e:
                progress_bar.update(1)
                print(f"Error processing item {idx} in subject {subject}: {e}")
                return idx, None
    try:
        for subject in data.keys():
            subject_results = [None] * len(data[subject])
            with tqdm(
                total=len(data[subject]),
                desc=subject,
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
                        subject_results[origin_idx] = parse_score(result)
            results[subject] = subject_results
            total_progress_bar.update(1)
    finally:
        export_data(data=results, output_path=output_path)
        total_progress_bar.close()
    
    return results
            
async def async_main(
    result_dir: str,
    export_dir: str,
    check_model_name: str = "gpt-4o",
    test_mode: bool = False,
    ):
    os.makedirs(os.path.join(PROJECT_BASE_PATH, export_dir), exist_ok=True)
    client = Client(
        mode='api',
        max_model_len=16384,
        openai_source="azure",
    )
    ########### load data ###########
    origin_question_dict, trans_question_dict = load_translation_data(
        os.path.join(PROJECT_BASE_PATH, result_dir),
        os.path.join(PROJECT_BASE_PATH, result_dir),
        data_type='question',
        test_mode=test_mode,
    )
    origin_answer_dict, trans_answer_dict = load_translation_data(
        os.path.join(PROJECT_BASE_PATH, result_dir),
        os.path.join(PROJECT_BASE_PATH, result_dir),
        data_type='answer',
        test_mode=test_mode,
    )
    style_data_path = os.path.join(PROJECT_BASE_PATH, result_dir, 'ques_trans_prompts.json')
    if os.path.exists(style_data_path):
        style_example_data = load_data(style_data_path)
        subj = list(style_example_data.keys())[0]
        style_example = retrieve_style_example(style_example_data[subj][0][-1]['content'])
    else:
        style_example = ""
    logger.info(f"Style example:\n{style_example}")
    
    ########### format data to prompts ###########
    question_prompt = format_prompt(
        source_data_dict=origin_question_dict,
        target_data_dict=trans_question_dict,
        style_example=style_example
    )
    export_data(
        data=question_prompt, 
        output_path=os.path.join(PROJECT_BASE_PATH, export_dir, "question_prompt.json")
    )
    answer_prompt = format_prompt(
        source_data_dict=origin_answer_dict,
        target_data_dict=trans_answer_dict,
        style_example=style_example
    )
    export_data(
        data=answer_prompt, 
        output_path=os.path.join(PROJECT_BASE_PATH, export_dir, "answer_prompt.json")
    )

    ########### process question checking ###########
    logger.info("Processing question checking...")
    question_check_result = await process_subjects(
        client=client,
        model_name=check_model_name,
        data=question_prompt,
        output_path=os.path.join(PROJECT_BASE_PATH, export_dir, "question_check_result.json"),
        progress_desc="Checking question translation",
        max_concurrency=100,
    )
    ########### process answer checking ###########
    logger.info("Processing answer checking...")
    answer_check_result = await process_subjects(
        client=client,
        model_name=check_model_name,
        data=answer_prompt,
        output_path=os.path.join(PROJECT_BASE_PATH, export_dir, "answer_check_result.json"), 
        progress_desc="Checking answer translation",
        max_concurrency=100,
    )
    ########### scoring ###########
    logger.info("Scoring...")
    score_dict = merge_and_calculate_results(
        question_check_result=question_check_result,
        answer_check_result=answer_check_result
    )
    ########### export results ###########
    logger.info("Exporting results...")
    export_data(
        data=score_dict, 
        output_path=os.path.join(PROJECT_BASE_PATH, export_dir, 'score.json'))

def main(
    result_dir: str,
    export_dir: str,
    check_model_name: str = "gpt-4o",
    test_mode: bool = False,
    ):
    asyncio.run(async_main(result_dir=result_dir, export_dir=export_dir, check_model_name=check_model_name, test_mode=test_mode))
    
if __name__ == "__main__":
    fire.Fire(main)