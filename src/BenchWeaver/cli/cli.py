import sys
from tabulate import tabulate
from enum import Enum, unique
import logging
from ..extras.print_env import VERSION, print_env
from .eval_score import run_eval
from .display_bench import display_benchmark_table
# this is a temp cli usage
USAGE = [
    ["Command", "Description"],
    ["bench-weaver-cli eval -h", "Evaluate models"],
    ["bench-weaver-cli webui", "Launch Gradio Webui for eval (planning)"],
    ["bench-weaver-cli version", "Show version info"],
    ["bench-weaver-cli benchmark", "Show supported benchmarks"],
    ["bench-weaver-cli env", "Show dependency info"],
    ["bench-weaver-cli help", "Show CLI usage"]
]

WELCOME = (
    "+" + "-" * 60 + "+"
    + "\n"
    + f"| Welcome to BenchWeaver, version {VERSION}"
    + " " * (27 - len(VERSION))
    + "|\n|"
    + " " * 60
    + "|\n"
    + "| Project page: https://github.com/joeyliang1024/BenchWeaver |\n"
    + "+" + "-" * 60 + "+"
)

def display_usage_table():
    print(tabulate(USAGE, headers="firstrow", tablefmt="grid"))
    
logger = logging.getLogger(__name__)

@unique
class Command(str, Enum):
    ENV = "env"
    VER = "version"
    HELP = "help"
    WEBUI = "webui"
    EVAL = "eval"
    BENCHMARK = "benchmark"
    
def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP

    if command == Command.ENV:
        print_env()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        display_usage_table()
    elif command == Command.WEBUI:
        # not implemented yet
        pass
    elif command == Command.BENCHMARK:
        display_benchmark_table()
    elif command == Command.EVAL:
        run_eval()
    else:
        raise NotImplementedError(f"Unknown command: {command}.")

if __name__ == "__main__":
    main()