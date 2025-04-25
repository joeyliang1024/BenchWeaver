import os
import json
from evaluate import load
from BenchWeaver.extras.constants import PROJECT_BASE_PATH

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
code_eval = load(os.path.join(PROJECT_BASE_PATH, "src/BenchWeaver/eval/metric/code_utils.py"))

# Problem 1: min_cost
test_1 = (
    'assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8\n'
    'assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12\n'
    'assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16'
)
code_1 = """
R = 3
C = 3
def min_cost(cost, m, n): 
    tc = [[0 for x in range(C)] for x in range(R)] 
    tc[0][0] = cost[0][0] 
    for i in range(1, m+1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
    for j in range(1, n+1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] 
    return tc[m][n]
"""

# Problem 2: factorial
test_2 = (
    'assert factorial(5) == 120\n'
    'assert factorial(0) == 1\n'
    'assert factorial(3) == 6'
)
code_2 = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

# Combine
references = [test_1, test_2]
predictions = [[code_1], [code_2]]  # List of lists — 1 candidate per problem

results = code_eval.compute(predictions=predictions, references=references, k=[1])
print(json.dumps(results, indent=2))
