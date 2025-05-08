# Original Copyright 2021 OpenAI under MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Origin Source Code: https://github.com/amazon-science/mxeval/blob/main/mxeval/evaluation.py
import itertools
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import numpy as np
import tqdm
from .data import read_problems, stream_jsonl, write_jsonl
from ....extras.constants import PROJECT_BASE_PATH
# Amazon modification
# import check correctness for all languages
from .execution import (
    check_correctness,
    check_correctness_cpp,
    check_correctness_csharp,
    check_correctness_go,
    check_correctness_java,
    check_correctness_javascript,
    check_correctness_kotlin,
    check_correctness_perl,
    check_correctness_php,
    check_correctness_ruby,
    check_correctness_scala,
    check_correctness_swift,
    check_correctness_typescript,
)

check_correctness_function_map = {
        "python": check_correctness,
        "java": check_correctness_java,
        "javascript": check_correctness_javascript,
        "typescript": check_correctness_typescript,
        "kotlin": check_correctness_kotlin,
        "ruby": check_correctness_ruby,
        "php": check_correctness_php,
        "cpp": check_correctness_cpp,
        "csharp": check_correctness_csharp,
        "go": check_correctness_go,
        "perl": check_correctness_perl,
        "scala": check_correctness_scala,
        "swift": check_correctness_swift,
    }

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def get_execute_function(lang):
    lang = lang.lower()
    assert lang in check_correctness_function_map, f"Language {lang} is not among the supported languages: {check_correctness_function_map.keys()}"
    return check_correctness_function_map[lang]

def evaluate_functional_correctness(
    sample_file: str | List[dict],
    problem_file: str | List[dict],
    output_dir: str = None,
    k: List[int] = [1, 10, 100],
    n_workers: int = os.cpu_count() - 1,
    timeout: float = 10.0,
):
    """
    Evaluates the functional correctness of generated samples for languages include:
    python, java, javascript, typescript, kotlin, ruby, php, cpp, csharp, go, perl, scala, swift.
    
    """
    # Check if the problem_file is a list of dictionaries or a file path
    if not isinstance(problem_file, list):
        problems = read_problems(problem_file)
    else:
        print("Skip reading problems -- using problem_file (List[dict]) as problems")
        problems = {iterdict["task_id"]: iterdict for iterdict in problem_file}
    # Check if the sample_file is a list of dictionaries or a file path
    if isinstance(sample_file, list):
        iterable = sample_file
        sample_count = len(sample_file)
    else:
        iterable = list(stream_jsonl(sample_file))
        sample_count = sum(1 for _ in iterable)
    # see execution.py for details
    # Check the generated samples against test suites.
    check_correctness_function_map = {
        "python": check_correctness,
        "java": check_correctness_java,
        "javascript": check_correctness_javascript,
        "typescript": check_correctness_typescript,
        "kotlin": check_correctness_kotlin,
        "ruby": check_correctness_ruby,
        "php": check_correctness_php,
        "cpp": check_correctness_cpp,
        "csharp": check_correctness_csharp,
        "go": check_correctness_go,
        "perl": check_correctness_perl,
        "scala": check_correctness_scala,
        "swift": check_correctness_swift,
    }

    seed = int(time.time() * 1000000) % 1000000
    np.random.seed(seed=seed)  # microsecond

    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        # First pass: submit all tasks and track number of samples
        with tqdm.tqdm(total=sample_count, desc="Submitting samples") as pbar_submit:
            for sample in iterable:
                task_id = sample["task_id"]
                completion = sample["completion"]
                pbar_submit.set_postfix_str(task_id)
                args = (problems[task_id], completion, timeout, completion_id[task_id])
                language = sample["language"]
                check_correctness_function = check_correctness_function_map[language]
                future = executor.submit(check_correctness_function, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
                pbar_submit.update(1)

        assert len(completion_id) == len(problems), "Some problems are not attempted.\nCompletion ID: " + str(len(completion_id)) + "\nProblems: " + str(len(problems))

        # Second pass: process completed futures
        with tqdm.tqdm(total=len(futures), desc="Running test suites") as pbar_run:
            for future in as_completed(futures):
                result = future.result()
                pbar_run.set_postfix_str(result["task_id"])
                results[result["task_id"]].append((result["completion_id"], result))
                pbar_run.update(1)

    # common code for all languages
    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    def combine_results():
        for sample in (sample_file if isinstance(sample_file, list) else stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            sample["time_elapsed"] = result[1]["time_elapsed"]
            yield sample
    
    if output_dir:
        out_file = os.path.join(output_dir, "mxeval_results.jsonl")
        print(f"Writing results to {out_file}")
        write_jsonl(
            out_file, 
            tqdm.tqdm(combine_results(), 
            total=n_samples)
        )
    if os.path.exists(os.path.join(PROJECT_BASE_PATH, "mxeval_cache")):
        shutil.rmtree(os.path.join(PROJECT_BASE_PATH, "mxeval_cache"))
    return pass_at_k
