import pandas as pd
import requests
import json
import gzip
import os
import datetime

from typing import Tuple 
import subprocess
from utils import stream_jsonl, execute_code_in_subprocess
from baselines import canonical_solution, openai_with_solution, openai_with_feedback, openai_without_solution

# Define the path to the HumanEval.jsonl.gz file
ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "data", "HumanEval.jsonl.gz")

# Load the problems from the file
problems = list(stream_jsonl(HUMAN_EVAL))

# Select code generation algorithm 
code_generations = {
    'canonical':  canonical_solution,
    'openai_with_solution': openai_with_solution,
    'openai_without_solution': openai_without_solution,
    'openai_with_feedback': openai_with_feedback,
    'openai_without_solution_pass_at_k': openai_without_solution,
}
code_generation_kwargs = {
    'canonical': {'append_prompt': True, 'test_hard_problems_only': True},
    'openai_with_solution': {'append_prompt': False, 'test_hard_problems_only': True},
    'openai_without_solution': {'append_prompt': False, 'model': 'gpt-4', 'test_hard_problems_only': True},
    'openai_with_feedback': {'append_prompt': False, 'pass_in_sol': False, 'total_trials': 20, 'model' : 'gpt-4', 'test_hard_problems_only': True},
    'openai_without_solution_pass_at_k': {'append_prompt': False, 'pass_at_k': 10, 'model' : 'gpt-4', 'test_hard_problems_only': True},
}
generation_algorithm = 'openai_with_feedback'
generation_kwargs = code_generation_kwargs[generation_algorithm]
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tag = f"{date}-{generation_algorithm}"
if generation_kwargs:
    tag += f"-{str(generation_kwargs)}"
test_hard_problems_only = generation_kwargs.pop('test_hard_problems_only', False)
hard_problems = os.path.join(ROOT, "data", "hard_9.txt")
hard_problems = set([x.strip() for x in open(hard_problems, 'r').readlines()])
# Solve problems using code generation algorithm
solutions = {}
full_code_executions = {}
std_errs = {}
std_outs = {}
extra_outputs = {}
pass_rate = 0
num_problems = 0
verbose = True 
CHECKER_BEGIN_CODE = 'def check(candidate)'
append_prompt = generation_kwargs.pop('append_prompt')
total_trials = generation_kwargs.pop('pass_at_k', 1)
for ii, problem in enumerate(problems):
    if problem['task_id'] not in hard_problems:
        continue
    problem_id = problem.get("task_id", f"HumanEval/{ii}")
    checker = CHECKER_BEGIN_CODE + problem['test'].split(CHECKER_BEGIN_CODE)[-1]
    num_trials = 0 
    overall_success = False
    while num_trials < total_trials:
        num_trials += 1
        solution, extra_output = code_generations[generation_algorithm](problem, **generation_kwargs)
        if append_prompt:
            code = problem['prompt'] + '\n' + solution + '\n' + checker + f'\ncheck({problem["entry_point"]})'
        else: 
            code = solution + '\n' + checker + f'\ncheck({problem["entry_point"]})'
        if verbose:
            print(problem_id, f" trial {num_trials}/{total_trials}")
            print(code)
        success, stdout, stderr = execute_code_in_subprocess(code)
        extra_outputs[problem_id] = extra_output
        solutions[problem_id] = solution
        full_code_executions[problem_id] = code
        std_errs[problem_id] = stderr
        std_outs[problem_id] = stdout
        if success:
            overall_success = True
            break
    num_problems += 1
    pass_rate += overall_success
    print(problem_id, "success:", overall_success)
print(f"Success rate: {pass_rate}/{num_problems} = {pass_rate/num_problems*100:.2f}%")

# Save the output
def save_output(tag, solutions, std_errs, std_outs, full_code_executions, pass_rate, extra_outputs):
    os.makedirs(f'output/{tag}', exist_ok=True)
    with open(f"output/{tag}/solution.json", "w") as outfile:
        json.dump(solutions, outfile)
    with open(f'output/{tag}/std_errs.json', 'w') as outfile:
        json.dump(std_errs, outfile)
    with open(f'output/{tag}/std_outs.json', 'w') as outfile:
        json.dump(std_outs, outfile)
    with open(f'output/{tag}/full_code_executions.json', 'w') as outfile:
        json.dump(full_code_executions, outfile)
    with open(f'output/{tag}/pass_rate.txt', 'w') as outfile:
        json.dump({'pass_rate' : pass_rate/len(problems)*100}, outfile)
    with open(f'output/{tag}/extra_outputs.json', 'w') as outfile:
        json.dump({'extra_outputs' : extra_outputs}, outfile)
    # read source code of file and save to output
    with open(__file__, 'r') as f:
        code = f.read()
    with open(f'output/{tag}/source_code.py', 'w') as outfile:
        outfile.write(code)

print(f"Saving output to {tag}")
save_output(tag, solutions, std_errs, std_outs, full_code_executions, pass_rate, extra_outputs)
