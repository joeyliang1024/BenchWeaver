import asyncio
import argparse
import yaml
from ..eval.evaluator.evaluator import Evaluator
from ..eval.benchmarks.configs import BENCHMARK_CONFIG, get_evaluators

async def eval():
    args = parse_args()

    # Load configuration from the provided YAML file
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    task = config['task'].split("_")[0]
    mode = config['benchmark_mode']
    pipeline = config['pipeline']
    # Select the appropriate evaluator
    if task not in BENCHMARK_CONFIG.keys() or mode not in BENCHMARK_CONFIG[task]['mode']:
        raise ValueError(f"Task '{task}' with mode '{mode}' is not supported.")

    evaluator_class = get_evaluators(task)[mode]
    evaluator: Evaluator = evaluator_class(args=config)
    # Run the evaluation
    if mode == "mcqa-prob":
        evaluator.eval()
    else:
        if pipeline == "same":
            # same language evaluation
            await evaluator.same_lang_eval(
                choices=BENCHMARK_CONFIG[task]['mcqa_choices'],
                subjects=BENCHMARK_CONFIG[task]['display_scores'],
            )
        else:
            # different language evaluation
            await evaluator.diff_lang_eval(
                choices=BENCHMARK_CONFIG[task]['mcqa_choices'],
                subjects=BENCHMARK_CONFIG[task]['display_scores'],
            )
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation using a config file.")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="Path to the config.yaml file."
    )
    return parser.parse_args()


def run_eval():
    """
    excute function for cli
    """
    asyncio.run(eval())
    
  
if __name__ == "__main__":
    asyncio.run(eval())
