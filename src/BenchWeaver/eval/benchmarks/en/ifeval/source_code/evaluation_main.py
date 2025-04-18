import os
from typing import Optional
from ......extras.logging import get_logger
from .evaluation_lib import (
    read_prompt_list,
    read_prompt_to_response_dict,
    test_instruction_following_strict,
    test_instruction_following_loose,
    write_outputs,
    get_report,
)

logger = get_logger(__name__)

def evaluate_instruction_following(
    input_data: str,
    input_response_data: Optional[str],
    output_dir: str = None,
    disable_output: bool = True,
    from_dir: bool = False,
):
    inputs = read_prompt_list(input_data, from_dir=from_dir)
    prompt_to_response = read_prompt_to_response_dict(input_response_data, from_dir=from_dir)
    scores = {}
    for func, score_name, output_file_name in [
        (test_instruction_following_strict, "strict", "eval_results_strict"),
        (test_instruction_following_loose, "loose", "eval_results_loose"),
    ]:
        logger.info(f"Evaluating with {score_name} mode ...")
        outputs = [func(inp, prompt_to_response) for inp in inputs]
        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        logger.info(f"Accuracy: {accuracy:.2%}")
        
        # Write the outputs to a file if not disabled
        if not disable_output:
            assert output_dir is not None, "Output directory must be specified if disable_output is False."
            output_file_path = os.path.join(output_dir, output_file_name + ".jsonl")
            write_outputs(output_file_path, outputs)
            print(f"Generated: {output_file_path}")
        # Update the scores dictionary with the results
        score_record = {}
        score_record["Average Accuracy"] = round(accuracy, 4)
        score_record.update(get_report(outputs))
        scores[score_name] = score_record
        
    return scores
