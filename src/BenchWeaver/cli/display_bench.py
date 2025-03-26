from tabulate import tabulate
from ..eval.benchmarks.configs import BENCHMARK_CONFIG, get_evaluators

def display_benchmark_table():
    table = []
    headers = ["Supported Benchmark", "Language", "Evaluators", "Suggest num shots", "Cot"]
    
    for benchmark, config in BENCHMARK_CONFIG.items():
        table.append([
            benchmark,
            config["language"],
            ", ".join(get_evaluators(benchmark)),
            config["sugguest_num_shots"],
            config["support_chain_of_thought"]
        ])
    
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    display_benchmark_table()